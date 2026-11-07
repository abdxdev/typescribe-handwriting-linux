"""
TypeScribe generation backend.

This module has no GUI dependencies. It is used in two ways:

* imported for `process_text` / `ALPHABET`
* run as a worker process (`python generate.py < job.json`) by the GTK GUI,
  inside the conda environment that has TensorFlow installed. Progress is
  reported as one JSON object per line on stdout.
"""

import json
import os
import random
import sys

ALPHABET = [
    "\x00", " ", "!", '"', "#", "'", "(", ")", ",", "-", ".",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";",
    "?", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
    "L", "M", "N", "O", "P", "R", "S", "T", "U", "V", "W", "Y",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
    "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x",
    "y", "z",
]

# Live drawing: lines are sampled in small groups so ink appears early. The first group is
# tiny (ink shows up quickly), later ones are bigger (batched sampling is faster per line).
LIVE_FIRST_CHUNK = 2
LIVE_CHUNK_LINES = 6


def unsupported_chars(text, alphabet=ALPHABET):
    """Characters in `text` the model cannot write (whitespace is fine)."""
    return [c for c in text if c not in alphabet and not c.isspace()]


def line_colors(job, count):
    """Per-line ink colours for the chosen ink mix (single / alternate / random)."""
    first = job["ink_color"]
    second = job.get("ink_color2") or first
    mode = job.get("ink_mode", 0)
    if mode == 1:
        return [first if i % 2 == 0 else second for i in range(count)]
    if mode == 2:
        return [random.choice((first, second)) for _ in range(count)]
    return [first] * count


def layout_pages(input_text, max_line_length, lines_per_page, alphabet=ALPHABET):
    """Sanitize, wrap and paginate text. Returns a list of pages (lists of lines).

    Empty lines are kept as "." which the synthesizer draws as a blank line.
    """
    lines = [line.strip() if line.strip() else "." for line in input_text.split("\n")]
    sanitized = ["".join(c if c in alphabet else " " for c in line) for line in lines]

    # The model rejects lines over 75 characters, so words longer than the
    # limit (URLs, unbroken strings) are split into chunks.
    limit = max(1, min(max_line_length, 75))

    def words_of(line):
        for word in line.split():
            while len(word) > limit:
                yield word[:limit]
                word = word[limit:]
            yield word

    wrapped = []
    for line in sanitized:
        current = ""
        for word in words_of(line):
            if len(current) + len(word) + 1 > max_line_length:
                if current.strip():
                    wrapped.append(current.strip())
                current = word
            else:
                current += " " + word
        if current:
            wrapped.append(current.strip())

    return [wrapped[i : i + lines_per_page] for i in range(0, len(wrapped), lines_per_page)]


def process_text(job, progress=None, hand=None):
    """Generate handwriting SVG pages for `job` (a dict, see gui.py).

    `hand` is a loaded model to reuse; otherwise one is created. When the job has
    `only_pages` (1-based numbers) just those pages are written again.
    """
    def emit(**event):
        if progress:
            progress(event)

    pages = layout_pages(job["text"], job["max_line_length"], job["lines_per_page"])
    if not pages:
        raise ValueError("There is no text to write.")

    if hand is None:
        from handwriting_synthesis import Hand  # heavy import (TensorFlow)
        emit(event="loading")
        hand = Hand()

    output_dir = job["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    wanted = job.get("only_pages") or range(1, len(pages) + 1)
    wanted = [n for n in wanted if 1 <= n <= len(pages)]

    written = []
    # Empty ruled lines left above the text. They are written as blank lines ("."), so the
    # model skips them and everything below keeps its place on the ruling.
    start = max(0, int(job.get("start_line", 0)))
    for n in wanted:
        text_lines = pages[n - 1]
        page_lines = ["."] * start + text_lines
        emit(event="page_start", n=n, total=len(pages), index=len(written), count=len(wanted))
        path = os.path.join(output_dir, f"result_page_{n}.svg")
        count = len(page_lines)
        colors = [job["ink_color"]] * start + line_colors(job, len(text_lines))

        live = {}
        if job.get("live"):
            # Sample a few lines at a time and report each line's ink as soon as it exists,
            # so the GUI can draw the pen strokes while the rest of the page is still being made.
            def on_line(index, pieces, color, n=n):
                emit(
                    event="line", n=n, line=index, color=color,
                    pieces=[
                        {"w": width, "p": [round(float(v), 2) for point in points for v in point]}
                        for points, width in pieces
                    ],
                )

            live = {
                "on_line": on_line,
                "first_chunk": job.get("live_first_chunk", LIVE_FIRST_CHUNK) + start,  # blanks cost nothing
                "chunk_lines": job.get("live_chunk_lines", LIVE_CHUNK_LINES),
            }

        hand.write(
            filename=path,
            lines=page_lines,
            biases=[job["bias"]] * count,
            styles=[job["style"]] * count,
            stroke_colors=colors,
            stroke_widths=[job["pen_width"]] * count,
            page=job["page"],
            pressure=job.get("pressure", 0.0),
            **live,
        )
        written.append(path)
        emit(event="page_done", n=n, total=len(pages), index=len(written), count=len(wanted), path=path)
    return written


def _worker():
    """Serve one JSON job per stdin line, keeping the model loaded between jobs."""
    # The model loads its weights from paths relative to the project root.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.getcwd())

    # TensorFlow and friends print to stdout; keep stdout clean for our protocol.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout = sys.stderr

    def send(event):
        protocol.write(json.dumps(event) + "\n")
        protocol.flush()

    hand = None
    for raw in sys.stdin:
        if not raw.strip():
            continue
        try:
            job = json.loads(raw)
            if hand is None:
                from handwriting_synthesis import Hand  # heavy import (TensorFlow)
                send({"event": "loading"})
                hand = Hand()
            files = process_text(job, progress=send, hand=hand)
            send({"event": "done", "files": files})
        except Exception as exc:  # reported to the GUI, which shows it in a dialog
            send({"event": "error", "message": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    _worker()

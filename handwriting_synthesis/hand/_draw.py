import numpy as np
import svgwrite
import cairosvg

from handwriting_synthesis import drawing

DEFAULT_PAGE = [32, 24, 896, 632, -64, -96, "#fff", "#f00", "#888"]

# Gap between the left margin and the red margin line (72-ppi units). A fixed value, so
# changing the line height never moves the margin. Keep in sync with gui.py.
MARGIN_LINE_OFFSET = 16


def unpack_page(page):
    """Split the `page` list into named values (see gui.py `_build_job`)."""
    page = page or DEFAULT_PAGE
    line_height, total_lines, view_height, view_width, margin_left, margin_top, page_color, margin_color, line_color = page[:9]
    return {
        "line_height": line_height,
        "total_lines": total_lines,
        "view_height": view_height,
        "view_width": view_width,
        "margin_left": margin_left,
        "margin_top": margin_top,
        "page_color": page_color,
        "margin_color": margin_color,
        "line_color": line_color,
        # Optional 10th entry: draw the page background (color, ruled lines, margins).
        "include_background": page[9] if len(page) > 9 else True,
        # Optional 11th entry: export scale (ppi / 72). The layout stays in 72-ppi
        # units (the viewBox); the SVG/PNG are written at the scaled pixel size.
        "export_scale": page[10] if len(page) > 10 else None,
    }


def _pen_strokes(strokes):
    """Split (x, y, end-of-stroke) rows into separate pen strokes (lists of (x, y))."""
    pens, current, prev_eos = [], [], 1.0
    for x, y, eos in zip(*strokes.T):
        if prev_eos == 1.0 and current:
            pens.append(current)
            current = []
        current.append((float(x), float(y)))
        prev_eos = eos
    if current:
        pens.append(current)
    return pens


def line_pieces(strokes, width, pressure, chunk=5):
    """A line's ink as a list of (points, width) pieces.

    Without pressure every pen stroke is one piece. With pressure the strokes are cut
    into short pieces whose width drifts like pen pressure. The SVG export and the live
    preview both draw exactly these pieces, so they always agree.
    """
    pens = _pen_strokes(strokes)
    if pressure <= 0:
        return [(points, width) for points in pens]

    level = 0.0
    pieces = []
    for points in pens:
        for start in range(0, len(points) - 1, chunk - 1):
            part = points[start:start + chunk]
            if len(part) < 2:
                continue
            level = float(np.clip(level + np.random.normal(0.0, 0.35), -1.0, 1.0))
            piece_width = max(width * 0.25, width * (1 + pressure * 0.7 * level))
            pieces.append((part, round(piece_width, 3)))
    return pieces


def page_line_pieces(offsets, index, page, width, pressure):
    """Page-space ink pieces for line number `index` (0-based) of a page."""
    values = unpack_page(page)
    line_height = values["line_height"]
    initial_coord = np.array([
        values["margin_left"],
        values["margin_top"] - line_height / 2 - index * line_height,
    ])

    # Convert offsets to coordinates and adjust them
    strokes = drawing.offsets_to_coords(offsets)
    strokes = drawing.denoise(strokes)
    strokes[:, :2] = drawing.align(strokes[:, :2])
    strokes[:, 1] *= -1
    strokes[:, :2] -= strokes[:, :2].min() + initial_coord
    return line_pieces(strokes, width, pressure)


def _add_line(dwg, pieces, color, pressure):
    if not pieces:
        return
    if pressure > 0:
        for points, width in pieces:
            d = "M{},{} ".format(*points[0]) + " ".join("L{},{}".format(x, y) for x, y in points[1:])
            dwg.add(svgwrite.path.Path(d).stroke(color=color, width=width, linecap='round').fill("none"))
        return

    # One path per line; every pen stroke starts with a move.
    parts = []
    for points, _width in pieces:
        parts.append("M{},{} ".format(*points[0]) + " ".join("L{},{}".format(x, y) for x, y in points[1:]))
    path = svgwrite.path.Path(" ".join(parts))
    dwg.add(path.stroke(color=color, width=pieces[0][1], linecap='round').fill("none"))


def _draw(strokes, lines, filename, stroke_colors=None, stroke_widths=None, page=None, pressure=0.0, pieces=None):
    """Write the page as SVG (and PNG).

    `strokes` are the model's raw offsets, one array per line. When `pieces` is given
    (one list per line, from `page_line_pieces`) those are drawn instead, which lets the
    caller compute the ink line by line and show it while the page is still being made.
    """
    stroke_colors = stroke_colors or ['black'] * len(lines)
    stroke_widths = stroke_widths or [2] * len(lines)

    values = unpack_page(page)
    line_height = values["line_height"]
    total_lines_per_page = values["total_lines"]
    view_height, view_width = values["view_height"], values["view_width"]
    margin_left, margin_top = values["margin_left"], values["margin_top"]
    page_color, margin_color, line_color = values["page_color"], values["margin_color"], values["line_color"]
    include_background = values["include_background"]
    export_scale = values["export_scale"]

    # Initialize the SVG drawing
    if export_scale:
        size = (f"{view_width * export_scale:.2f}px", f"{view_height * export_scale:.2f}px")
        dwg = svgwrite.Drawing(filename=filename, size=size)
    else:
        dwg = svgwrite.Drawing(filename=filename)
    dwg.viewbox(width=view_width, height=view_height)

    if include_background:
        # Tagged so exports can leave the paper and rules out while keeping the ink.
        background = dwg.g(id="page-background")
        background.add(dwg.rect(insert=(0, 0), size=(view_width, view_height), fill=page_color))

        # Draw fixed number of ruled lines
        for i in range(total_lines_per_page):
            y_position = line_height * (i + 1) - margin_top # Adjust as needed to align with text
            background.add(dwg.line(start=(0, y_position), end=(view_width, y_position), stroke=line_color, stroke_width=1))

        margin_x = -margin_left + MARGIN_LINE_OFFSET  # independent of line height
        background.add(dwg.line(start=(margin_x, 0), end=(margin_x, view_height), stroke=margin_color, stroke_width=1))
        background.add(dwg.line(start=(margin_x - 5, 0), end=(margin_x - 5, view_height), stroke=margin_color, stroke_width=1))

        background.add(dwg.line(start=(0, -margin_top), end=(view_width, -margin_top), stroke=margin_color, stroke_width=1))
        background.add(dwg.line(start=(0, -margin_top - 5), end=(view_width, -margin_top - 5), stroke=margin_color, stroke_width=1))

        dwg.add(background)

    for i, line in enumerate(lines):
        # Stop drawing text if lines exceed the fixed page limit
        if i >= total_lines_per_page:
            break

        # Blank lines only move the pen down (the line index fixes the baseline).
        if not line:
            continue

        if pieces is not None and pieces[i] is not None:
            line_ink = pieces[i]
        else:
            line_ink = page_line_pieces(strokes[i], i, page, stroke_widths[i], pressure)
        _add_line(dwg, line_ink, stroke_colors[i], pressure)

    # Save the SVG file and convert to PNG
    dwg.save()
    cairosvg.svg2png(url=filename, write_to=filename + '.png')

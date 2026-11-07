#!/usr/bin/env python3
"""TypeScribe - native GTK4 / libadwaita front end.

The GUI runs on the system Python (PyGObject). Handwriting generation runs in a
separate worker process (generate.py) using the Python environment that has
TensorFlow installed, so the interface never freezes while the model works.

Layout plan - every setting and every action has exactly ONE home
-----------------------------------------------------------------
  Header bar    Whole-app actions: settings toggle, open text file,
                Generate / Cancel, Export..., main menu (About).
  Sidebar       Every setting that shapes the result, scoped to a profile.
                  Profile row    switch, new, rename, reset, delete
                  Handwriting    style, strokes (consistency / thickness /
                                 pressure), ink (color / mix / second color)
                  Page           paper (size / orientation / resolution /
                                 width / height), ruling (line height / ruled
                                 lines / margins), text on the page
                                 (characters per line / lines written / fit),
                                 colors (page / margin / line)
  Text pane     The text that gets written.
  Canvas pane   Layout  - live preview of the ruled page; drag its edges, margin
                          lines and handles to change the page settings directly
                Pages   - the generated result: browse, regenerate one page
  Export dialog Everything about saving: folder, file name, formats, page
                background. Opened from the header, nowhere else.
  Status bar    State, warnings and progress (read-only).

Toasts are feedback only; they never carry actions or settings.
"""

import importlib.util
import io
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

try:
    import gi
except ImportError:
    # PyGObject lives in the distro's Python, not in conda/mise/venv interpreters.
    system_python = "/usr/bin/python3"
    if os.path.realpath(sys.executable) != os.path.realpath(system_python) and os.path.exists(system_python):
        os.execv(system_python, [system_python, *sys.argv])
    sys.exit(
        "TypeScribe needs PyGObject, GTK 4 and libadwaita.\n"
        "Arch:   sudo pacman -S python-gobject gtk4 libadwaita\n"
        "Fedora: sudo dnf install python3-gobject gtk4 libadwaita\n"
        "Debian/Ubuntu: sudo apt install python3-gi gir1.2-gtk-4.0 gir1.2-adw-1"
    )

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, GObject, Gtk, Pango  # noqa: E402

gi.require_version("GdkPixbuf", "2.0")
from gi.repository import GdkPixbuf  # noqa: E402

try:  # librsvg lets the PDF export stay vector; without it pages are embedded as images
    gi.require_version("Rsvg", "2.0")
    from gi.repository import Rsvg  # noqa: E402
except (ValueError, ImportError):
    Rsvg = None

import cairo  # noqa: E402  (pycairo ships with PyGObject)
from generate import layout_pages, unsupported_chars  # noqa: E402

APP_ID = "io.github.typescribe.TypeScribe"
APP_DIR = Path(__file__).resolve().parent
ASSETS = APP_DIR / "assets"
VERSION = "1.2"
MODEL_ENV_NAME = "test_hand"

# Leaving the paper out of an export means re-rendering the SVG, which needs librsvg.
CAN_STRIP_BACKGROUND = Rsvg is not None

DEFAULT_BASE_NAME = "handwriting"
DEFAULT_PROFILE = "Default"
INK_MODES = ["Single color", "Alternate lines", "Random per line"]

# Profile settings that live in the sidebar.
DEFAULTS = {
    "line_length": 60,
    "lines_per_page": 24,
    "total_lines": 24,
    "style": 1,
    "bias": 0.95,
    "pen_width": 1.0,
    "ink_color": "#000000",
    "ink_color2": "#1c4fd8",
    "ink_mode": 0,
    "pressure": 0.0,
    "paper": "A4",
    "ppi": 72,
    "ppi_custom": 200,
    "line_height": 32.0,
    "view_height": 841.89,  # A4 at 72 ppi
    "view_width": 595.28,
    "margin_left": 72.0,
    "margin_top": 72.0,
    "page_color": "#ffffff",
    "margin_color": "#ff0000",
    "line_color": "#d3d3d3",
    "reference_opacity": 50,  # percent; the reference image is for the layout preview only
}

# Profile settings that live in the Export dialog.
EXPORT_DEFAULTS = {
    "name": DEFAULT_BASE_NAME,
    "name_auto": True,  # derive the name from the text until the user edits it
    "format": "svg",  # page files: "svg" or "png" (one choice)
    "pdf": False,     # one PDF instead of separate page files, built from the format above
    "background": True,
}

CSS = """
.style-sample { background: #ffffff; border-radius: 8px; padding: 8px; }
.page-preview { border-radius: 12px; }
.result-paper { background: #f2f2f2; }
"""

# Generated pages are rendered here first; "Export" copies them to the chosen folder.
PREVIEW_DIR = Path(GLib.get_user_cache_dir()) / "typescribe" / "preview"
SETTINGS_FILE = Path(GLib.get_user_config_dir()) / "typescribe" / "settings.json"


def load_settings():
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(settings):
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except OSError:
        pass  # remembering settings is best-effort


def migrate_settings(settings):
    """Normalise the settings dict and move legacy `ui` data into a profile."""
    profiles = settings.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        legacy = settings.get("ui")
        profiles = {DEFAULT_PROFILE: legacy if isinstance(legacy, dict) else {}}
    profiles = {
        str(name): (data if isinstance(data, dict) else {})
        for name, data in profiles.items() if str(name).strip()
    }
    if not profiles:
        profiles = {DEFAULT_PROFILE: {}}
    settings["profiles"] = profiles
    settings.pop("ui", None)
    active = settings.get("active_profile")
    settings["active_profile"] = active if active in profiles else next(iter(profiles))
    return settings


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_rgba(value):
    color = Gdk.RGBA()
    color.parse(value)
    return color


def rgba_to_hex(color):
    return "#%02x%02x%02x" % tuple(
        round(max(0.0, min(1.0, c)) * 255) for c in (color.red, color.green, color.blue)
    )


def derive_base_name(text, limit=48):
    """Build a safe file name from the beginning of the text being handwritten."""
    prefix = text.lstrip()[:limit]
    name = re.sub(r"[^\w-]+", "_", prefix, flags=re.UNICODE).strip("_-. ")
    return name.lower() or DEFAULT_BASE_NAME


def set_source(cr, value):
    color = parse_rgba(value)
    cr.set_source_rgba(color.red, color.green, color.blue, color.alpha)


BACKGROUND_GROUP_RE = re.compile(r'\s*<g id="page-background">.*?</g>', re.S)


def svg_text_size(svg_text):
    """Page size in points (the SVG viewBox is in 72-ppi units), or None."""
    match = re.search(
        r'viewBox="\s*[-\d.]+[\s,]+[-\d.]+[\s,]+([\d.]+)[\s,]+([\d.]+)\s*"', svg_text
    )
    return (float(match.group(1)), float(match.group(2))) if match else None


def strip_page_background(svg_text):
    """Remove the paper color, ruled lines and margins from an SVG document."""
    return BACKGROUND_GROUP_RE.sub("", svg_text, count=1)


REFERENCE_FITS = ["Stretch to page", "Fit inside page", "Fill page"]
REFERENCE_MAX_PX = 1800  # larger images are scaled down once so dragging stays smooth


def load_reference_surface(path):
    """Load an image as a cairo surface for the layout preview (upright, downscaled if huge)."""
    pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(str(path), REFERENCE_MAX_PX, REFERENCE_MAX_PX, True)
    pixbuf = pixbuf.apply_embedded_orientation() or pixbuf  # phone photos carry a rotation tag
    # Hand the pixels to cairo through an in-memory PNG (fast settings; cairo handles alpha).
    ok, data = pixbuf.save_to_bufferv("png", ["compression"], ["1"])
    if not ok:
        raise ValueError("could not decode the image")
    return cairo.ImageSurface.create_from_png(io.BytesIO(data))


def page_image(svg_path, include_background=True):
    """The page as a cairo image surface, optionally without paper and ruled lines."""
    if include_background:
        return cairo.ImageSurface.create_from_png(f"{svg_path}.png")
    if Rsvg is None:
        raise RuntimeError("Excluding the page background from a PNG needs librsvg.")
    text = strip_page_background(Path(svg_path).read_text(encoding="utf-8", errors="replace"))
    size = svg_text_size(text)
    if size is None:
        raise RuntimeError("Could not read the page size from the SVG.")
    rendered = cairo.ImageSurface.create_from_png(f"{svg_path}.png")
    width, height = rendered.get_width(), rendered.get_height()
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    cr = cairo.Context(surface)
    cr.scale(width / size[0], height / size[1])
    viewport = Rsvg.Rectangle()
    viewport.x, viewport.y, viewport.width, viewport.height = 0, 0, size[0], size[1]
    Rsvg.Handle.new_from_data(text.encode("utf-8")).render_document(cr, viewport)
    return surface


def write_png(svg_path, png_path, include_background=True):
    """Write a page as PNG, optionally leaving out the paper and ruled lines."""
    if include_background:
        shutil.copy2(f"{svg_path}.png", png_path)
    else:
        page_image(svg_path, False).write_to_png(str(png_path))


def write_pdf(svg_pages, pdf_path, include_background=True, from_png=False):
    """Combine the pages into one multi-page PDF.

    Built from the SVG pages (needs librsvg) the strokes stay vector: sharp at any
    zoom and independent of the chosen ppi. Built from the PNG pages (`from_png`)
    each page is embedded as an image at its exported resolution. Leaving out the
    background requires librsvg either way.
    """
    if not include_background and Rsvg is None:
        raise RuntimeError("Excluding the page background from a PDF needs librsvg.")
    pdf = None
    try:
        for svg in svg_pages:
            text = Path(svg).read_text(encoding="utf-8", errors="replace")
            if not include_background:
                text = strip_page_background(text)
            size, image = svg_text_size(text), None
            if from_png or Rsvg is None or size is None:
                image = page_image(svg, include_background)
                size = size or (image.get_width(), image.get_height())
            width, height = size
            if pdf is None:
                pdf = cairo.PDFSurface(str(pdf_path), width, height)
            pdf.set_size(width, height)
            cr = cairo.Context(pdf)
            if image is None:
                viewport = Rsvg.Rectangle()
                viewport.x, viewport.y, viewport.width, viewport.height = 0, 0, width, height
                Rsvg.Handle.new_from_data(text.encode("utf-8")).render_document(cr, viewport)
            else:
                cr.scale(width / image.get_width(), height / image.get_height())
                cr.set_source_surface(image, 0, 0)
                cr.paint()
            pdf.show_page()
    finally:
        if pdf is not None:
            pdf.finish()


def lines_overflow(v):
    """True if the last ruled line falls below the bottom of the page."""
    return v["line_height"] * v["total_lines"] + v["margin_top"] > v["view_height"]


def find_model_python():
    """Locate a Python interpreter that can run the TensorFlow model."""
    candidates = []
    override = os.environ.get("TYPESCRIBE_PYTHON")
    if override:
        candidates.append(Path(override))

    prefix = os.environ.get("CONDA_PREFIX")
    if prefix and Path(prefix).name == MODEL_ENV_NAME:
        candidates.append(Path(prefix) / "bin" / "python")

    home = Path.home()
    roots = [
        home / "miniforge3", home / "mambaforge", home / "miniconda3",
        home / "anaconda3", home / ".conda", home / ".local/share/mamba",
        Path("/opt/miniforge3"), Path("/opt/miniconda3"), Path("/opt/conda"),
    ]
    for root in roots:
        candidates.append(root / "envs" / MODEL_ENV_NAME / "bin" / "python")

    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)

    # Last resort: the interpreter running the GUI already has TensorFlow.
    if importlib.util.find_spec("tensorflow"):
        return sys.executable
    return None


# --------------------------------------------------------------------------- #
# Page geometry helpers (shared by "Fit to page" and the draggable preview)
# --------------------------------------------------------------------------- #
CHAR_WIDTH_UNITS = 8.3  # handwriting is roughly this wide per character (72-ppi units)
TEXT_SLACK_UNITS = 24   # unused space kept at the right of the text block


def fit_lines(height, margin_top, line_height):
    """Ruled lines that fit between the top margin and the bottom of the page."""
    return max(1, min(200, int((height - margin_top) // line_height)))  # 200 = spinner maximum


def fit_chars(width, margin_left, ppi):
    """Characters per line that fit between the left margin and the right edge."""
    scale = 72 / ppi  # px -> the model's 72-based units
    chars = int((width * scale - margin_left * scale - TEXT_SLACK_UNITS) / CHAR_WIDTH_UNITS)
    return max(20, min(75, chars))


def layout_drag_changes(handle, s, dx, dy):
    """Setting changes for dragging `handle` by (dx, dy) page pixels.

    `s` is a snapshot of the values taken when the drag began. Ruled lines and
    characters per line keep following the page while they matched what
    "Fit to page" would pick; otherwise they are only capped to what still fits.
    The dict is ordered so that ruled lines are set before lines written.
    """
    ppi = s["ppi"]
    width, height = s["view_width"], s["view_height"]
    left, top, line = s["margin_left"], s["margin_top"], s["line_height"]
    lines0, written0, chars0 = s["total_lines"], s["lines_per_page"], s["line_length"]

    # Knobs that change one text setting only.
    if handle == "total_lines":
        lines = max(1, min(fit_lines(height, top, line), round(lines0 + dy / line)))
        return {
            "total_lines": lines,
            "lines_per_page": lines if written0 == lines0 else min(written0, lines),
        }
    if handle == "lines_per_page":
        return {"lines_per_page": max(1, min(lines0, round(written0 + dy / line)))}
    if handle == "line_length":
        per_char = CHAR_WIDTH_UNITS * ppi / 72
        most = max(20, fit_chars(width, left, ppi))
        return {"line_length": max(20, min(most, round(chars0 + dx / per_char)))}

    # Edges and margin lines change the geometry; the text settings then reflow.
    def snap(value, step=1.0):
        return round(value / step) * step

    new_w, new_h, new_left, new_top, new_line = width, height, left, top, line
    if handle in ("width", "size"):
        new_w = max(50.0, snap(width + dx))
    if handle in ("height", "size"):
        new_h = max(50.0, snap(height + dy))
    if handle == "margin_left":
        new_left = max(0.0, snap(left + dx, 0.5))
    if handle == "margin_top":
        new_top = max(0.0, snap(top + dy, 0.5))
    if handle == "line_height":
        new_line = max(4.0, snap(line + dy, 0.5))

    was_full = lines0 == fit_lines(height, top, line)
    fits = fit_lines(new_h, new_top, new_line)
    lines = fits if was_full else min(lines0, fits)
    written = lines if written0 == lines0 else min(written0, lines)

    chars_followed = chars0 == fit_chars(width, left, ppi)
    chars_fit = fit_chars(new_w, new_left, ppi)
    chars = chars_fit if chars_followed else min(chars0, chars_fit)

    return {
        "view_width": new_w, "view_height": new_h,
        "margin_left": new_left, "margin_top": new_top, "line_height": new_line,
        "total_lines": lines, "lines_per_page": written, "line_length": chars,
    }


def accent_rgb():
    """The desktop accent color (libadwaita 1.6+), or a neutral blue."""
    try:
        color = Adw.StyleManager.get_default().get_accent_color_rgba()
        return color.red, color.green, color.blue
    except AttributeError:
        return 0.21, 0.52, 0.89


def rounded_rect(cr, x, y, w, h, r):
    r = min(r, w / 2, h / 2)
    cr.new_sub_path()
    cr.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    cr.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    cr.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    cr.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    cr.close_path()


# --------------------------------------------------------------------------- #
# Page layout preview - also the interactive layout editor
# --------------------------------------------------------------------------- #
class PagePreview(Gtk.DrawingArea):
    """Draws the ruled page exactly as the generated SVG will lay it out.

    Handles (all draggable):
      right edge / bottom edge / corner   page width, height, both
      red top-margin line                 top margin
      red left-margin line                left margin
      knob on the first ruled line        line height
      knob on the last ruled line         ruled lines per page
      knob on the last written line       lines written per page
      knob in the header band             characters per line (approximate)
    """

    PAD = 28
    FIT = 0.92        # leave some headroom so edges can be dragged outwards
    EDGE_HIT = 7
    KNOB_R = 7

    CURSORS = {
        "width": "ew-resize", "height": "ns-resize", "size": "nwse-resize",
        "margin_left": "ew-resize", "margin_top": "ns-resize",
        "line_height": "ns-resize", "total_lines": "ns-resize",
        "lines_per_page": "ns-resize", "line_length": "ew-resize",
    }
    TIPS = {
        "width": "Drag to change the page width",
        "height": "Drag to change the page height",
        "size": "Drag to resize the page",
        "margin_left": "Drag to move the left margin",
        "margin_top": "Drag to move the top margin",
        "line_height": "Drag up or down to change the line height",
        "total_lines": "Drag to change the number of ruled lines",
        "lines_per_page": "Drag to change how many lines are written",
        "line_length": "Drag to change the characters per line (approximate)",
    }

    def __init__(self, get_values, on_drag):
        super().__init__(hexpand=True, vexpand=True)
        self.get_values = get_values
        self.on_drag = on_drag          # on_drag(handle, start_values, dx_px, dy_px)
        self._geom = None               # (scale, ox, oy) of the last draw
        self._frozen = None             # geometry pinned while a drag runs
        self._hover = None
        self._active = None
        self._start = None
        self._origin = (0.0, 0.0)
        self._pointer = (0.0, 0.0)
        self.reference = None           # cairo surface shown behind the ruling; never exported
        self.set_content_width(240)
        self.set_content_height(300)
        self.add_css_class("page-preview")
        self.set_draw_func(self._draw)

        motion = Gtk.EventControllerMotion()
        motion.connect("motion", self._on_motion)
        motion.connect("leave", self._on_leave)
        self.add_controller(motion)
        drag = Gtk.GestureDrag()
        drag.connect("drag-begin", self._on_drag_begin)
        drag.connect("drag-update", self._on_drag_update)
        drag.connect("drag-end", self._on_drag_end)
        self.add_controller(drag)

    def set_reference(self, surface):
        self.reference = surface
        self.queue_draw()

    # ---- geometry --------------------------------------------------------- #
    def _layout(self, width, height, v):
        """(scale, ox, oy): widget pixels per page pixel and the page's top-left."""
        if self._frozen is not None:
            return self._frozen
        scale = min(
            (width - 2 * self.PAD) / v["view_width"], (height - 2 * self.PAD) / v["view_height"]
        ) * self.FIT
        ox = (width - v["view_width"] * scale) / 2
        oy = (height - v["view_height"] * scale) / 2
        return scale, ox, oy

    def _knobs(self, v, scale, ox, oy):
        """Anchor points of the knobs, in widget coordinates."""
        pw, ph = v["view_width"] * scale, v["view_height"] * scale
        lh, mt, ml = v["line_height"], v["margin_top"], v["margin_left"]
        per_char = CHAR_WIDTH_UNITS * v["ppi"] / 72

        def on_page(y):
            return min(max(y, oy + 8), oy + ph - 8)

        wrap_x = min(ox + (ml + v["line_length"] * per_char) * scale, ox + pw - 10)
        return {
            "line_height": (ox + pw / 2, on_page(oy + (mt + lh) * scale)),
            "total_lines": (ox + pw - 22, on_page(oy + (mt + lh * v["total_lines"]) * scale)),
            "lines_per_page": (ox + 22, on_page(oy + (mt + lh * v["lines_per_page"]) * scale)),
            "line_length": (wrap_x, oy + min(max(12.0, mt * scale / 2), ph - 12)),
        }

    def _guides(self, name, v, scale, ox, oy):
        """Line segments that show what the hovered or dragged handle controls."""
        pw, ph = v["view_width"] * scale, v["view_height"] * scale
        lh, mt, ml = v["line_height"], v["margin_top"], v["margin_left"]
        per_char = CHAR_WIDTH_UNITS * v["ppi"] / 72

        def row(lines):
            y = oy + (mt + lh * lines) * scale
            return (ox, y, ox + pw, y)

        segments = []
        if name in ("width", "size"):
            segments.append((ox + pw, oy, ox + pw, oy + ph))
        if name in ("height", "size"):
            segments.append((ox, oy + ph, ox + pw, oy + ph))
        if name == "margin_top":
            y = oy + mt * scale
            segments.append((ox, y, ox + pw, y))
        if name == "margin_left":
            x = ox + (ml + lh / 2) * scale
            segments.append((x, oy, x, oy + ph))
        if name == "line_height":
            segments.append(row(1))
        if name == "total_lines":
            segments.append(row(v["total_lines"]))
        if name == "lines_per_page":
            segments.append(row(v["lines_per_page"]))
        if name == "line_length":
            x = min(ox + (ml + v["line_length"] * per_char) * scale, ox + pw - 10)
            segments.append((x, oy, x, oy + ph))
        return segments

    def _hit(self, x, y):
        """Which handle is under the pointer, or None."""
        if self._geom is None or not self.get_sensitive():
            return None
        scale, ox, oy = self._geom
        v = self.get_values()
        pw, ph = v["view_width"] * scale, v["view_height"] * scale
        lh, mt, ml = v["line_height"], v["margin_top"], v["margin_left"]

        # Knobs first: they sit on top of lines and edges.
        best, best_d = None, None
        for name, (kx, ky) in self._knobs(v, scale, ox, oy).items():
            d = math.hypot(x - kx, y - ky)
            if d <= self.KNOB_R + 5 and (best_d is None or d < best_d):
                best, best_d = name, d
        if best:
            return best
        if math.hypot(x - (ox + pw), y - (oy + ph)) <= 12:
            return "size"

        e = self.EDGE_HIT
        in_x = ox - e <= x <= ox + pw + e
        in_y = oy - e <= y <= oy + ph + e
        margin_x = ox + (ml + lh / 2) * scale  # the margin is a double line, 5 px apart
        margin_y = oy + mt * scale
        candidates = []
        if in_y:
            candidates.append(("width", abs(x - (ox + pw))))
            candidates.append(("margin_left", min(abs(x - margin_x), abs(x - (margin_x - 5 * scale)))))
        if in_x:
            candidates.append(("height", abs(y - (oy + ph))))
            candidates.append(("margin_top", min(abs(y - margin_y), abs(y - (margin_y - 5 * scale)))))
        if not candidates:
            return None
        name, distance = min(candidates, key=lambda item: item[1])
        return name if distance <= e else None

    # ---- pointer ----------------------------------------------------------- #
    def _on_motion(self, _controller, x, y):
        if self._active:
            return
        name = self._hit(x, y)
        if name != self._hover:
            self._hover = name
            self.set_cursor_from_name(self.CURSORS.get(name))
            self.set_tooltip_text(self.TIPS.get(name))
            self.queue_draw()

    def _on_leave(self, *_):
        if self._active:
            return
        self._hover = None
        self.set_cursor_from_name(None)
        self.set_tooltip_text(None)
        self.queue_draw()

    def _on_drag_begin(self, _gesture, x, y):
        self._active = self._hit(x, y)
        if not self._active:
            return
        self._start = dict(self.get_values())
        self._frozen = self._geom  # keep the zoom steady so edges follow the pointer
        self._origin = self._pointer = (x, y)
        self.set_tooltip_text(None)
        self.queue_draw()

    def _on_drag_update(self, _gesture, dx, dy):
        if not self._active:
            return
        scale = self._frozen[0]
        self._pointer = (self._origin[0] + dx, self._origin[1] + dy)
        self.on_drag(self._active, self._start, dx / scale, dy / scale)
        self.queue_draw()

    def _on_drag_end(self, *_):
        self._active = self._start = self._frozen = None
        self.queue_draw()  # re-fits the page to the widget again

    # ---- drawing ----------------------------------------------------------- #
    def _draw(self, area, cr, width, height):
        cr.rectangle(0, 0, width, height)
        cr.clip()
        v = self.get_values()
        scale, ox, oy = self._layout(width, height, v)
        if scale <= 0:
            return
        self._geom = (scale, ox, oy)
        pw, ph = v["view_width"] * scale, v["view_height"] * scale
        hairline = max(1.0, scale)

        # Paper with a soft shadow.
        cr.set_source_rgba(0, 0, 0, 0.18)
        cr.rectangle(ox + 2, oy + 3, pw, ph)
        cr.fill()
        set_source(cr, v["page_color"])
        cr.rectangle(ox, oy, pw, ph)
        cr.fill()

        cr.save()
        cr.rectangle(ox, oy, pw, ph)
        cr.clip()

        lh, mt, ml = v["line_height"], v["margin_top"], v["margin_left"]

        # Reference image (layout aid only): over the paper, under the ruling.
        image = self.reference
        if image is not None and v["reference_opacity"] > 0:
            iw, ih = image.get_width(), image.get_height()
            sx, sy = pw / iw, ph / ih  # "Stretch to page"
            if v["reference_fit"] == 1:
                sx = sy = min(sx, sy)
            elif v["reference_fit"] == 2:
                sx = sy = max(sx, sy)
            cr.save()
            cr.translate(ox + (pw - iw * sx) / 2, oy + (ph - ih * sy) / 2)
            cr.scale(sx, sy)
            cr.set_source_surface(image, 0, 0)
            cr.get_source().set_filter(cairo.FILTER_GOOD)
            cr.paint_with_alpha(v["reference_opacity"])
            cr.restore()

        # Ruled lines (same geometry as handwriting_synthesis/hand/_draw.py).
        set_source(cr, v["line_color"])
        cr.set_line_width(hairline)
        for i in range(v["total_lines"]):
            y = oy + (lh * (i + 1) + mt) * scale
            cr.move_to(ox, y)
            cr.line_to(ox + pw, y)
        cr.stroke()

        # Double margin lines.
        set_source(cr, v["margin_color"])
        cr.set_line_width(hairline)
        x = ox + (ml + lh / 2) * scale
        for dx in (0, -5 * scale):
            cr.move_to(x + dx, oy)
            cr.line_to(x + dx, oy + ph)
        y = oy + mt * scale
        for dy in (0, -5 * scale):
            cr.move_to(ox, y + dy)
            cr.line_to(ox + pw, y + dy)
        cr.stroke()
        cr.restore()

        if self.get_sensitive():  # no handles while a page is being written
            self._draw_handles(cr, v, scale, ox, oy)
        self._draw_labels(cr, v, ox, oy, pw, ph)
        if self._active:
            self._draw_value_pill(cr, v, width, height)

    def _draw_handles(self, cr, v, scale, ox, oy):
        ar, ag, ab = accent_rgb()
        pw, ph = v["view_width"] * scale, v["view_height"] * scale
        focus = self._active or self._hover

        # Guide for the handle that is hovered or being dragged.
        if focus:
            cr.set_source_rgba(ar, ag, ab, 0.95)
            cr.set_line_width(2.0)
            if focus in ("line_height", "total_lines", "lines_per_page", "line_length"):
                cr.set_dash([6, 4])
            for x1, y1, x2, y2 in self._guides(focus, v, scale, ox, oy):
                cr.move_to(x1, y1)
                cr.line_to(x2, y2)
            cr.stroke()
            cr.set_dash([])

        def grip(cx, cy, w, h, strong):
            rounded_rect(cr, cx - w / 2, cy - h / 2, w, h, min(w, h) / 2)
            cr.set_source_rgba(ar, ag, ab, 1.0 if strong else 0.8)
            cr.fill_preserve()
            cr.set_source_rgba(1, 1, 1, 0.95)
            cr.set_line_width(1.2)
            cr.stroke()

        grip(ox + pw, oy + ph / 2, 8, 36, focus in ("width", "size"))
        grip(ox + pw / 2, oy + ph, 36, 8, focus in ("height", "size"))
        grip(ox + pw, oy + ph, 14, 14, focus == "size")

        for name, (kx, ky) in self._knobs(v, scale, ox, oy).items():
            cr.new_sub_path()
            cr.arc(kx, ky, self.KNOB_R + (2 if name == focus else 0), 0, 2 * math.pi)
            cr.set_source_rgba(ar, ag, ab, 1.0)
            cr.fill_preserve()
            cr.set_source_rgba(1, 1, 1, 0.95)
            cr.set_line_width(1.8)
            cr.stroke()

    def _value_text(self, name, v):
        texts = {
            "width": f"Width {v['view_width']:.0f} px",
            "height": f"Height {v['view_height']:.0f} px",
            "size": f"{v['view_width']:.0f} × {v['view_height']:.0f} px",
            "margin_left": f"Left margin {v['margin_left']:g} px",
            "margin_top": f"Top margin {v['margin_top']:g} px",
            "line_height": f"Line height {v['line_height']:g} px",
            "total_lines": f"{v['total_lines']} ruled lines",
            "lines_per_page": f"{v['lines_per_page']} lines written",
            "line_length": f"{v['line_length']} characters per line",
        }
        return texts[name]

    def _draw_value_pill(self, cr, v, width, height):
        text = self._value_text(self._active, v)
        cr.select_font_face("Sans")
        cr.set_font_size(12)
        ext = cr.text_extents(text)
        w, h = ext.width + 16, 24
        px, py = self._pointer
        x = min(max(px + 14, 4), max(4, width - w - 4))
        y = min(max(py - 34, 4), max(4, height - h - 4))
        rounded_rect(cr, x, y, w, h, 8)
        cr.set_source_rgba(0, 0, 0, 0.8)
        cr.fill()
        cr.set_source_rgb(1, 1, 1)
        cr.move_to(x + 8, y + 16)
        cr.show_text(text)

    def _draw_labels(self, cr, v, ox, oy, pw, ph):
        # Dimension labels use the theme's foreground colour.
        fg = self.get_color()
        cr.set_source_rgba(fg.red, fg.green, fg.blue, 0.7)
        cr.select_font_face("Sans")
        cr.set_font_size(11)

        ppi = v["ppi"]
        label = f"{v['view_width']:.0f} px ({v['view_width'] / ppi:.2f} in)"
        ext = cr.text_extents(label)
        cr.move_to(ox + (pw - ext.width) / 2, oy - 8)
        cr.show_text(label)

        label = f"{v['view_height']:.0f} px ({v['view_height'] / ppi:.2f} in)"
        ext = cr.text_extents(label)
        cr.save()
        cr.translate(ox - 8, oy + (ph + ext.width) / 2)
        cr.rotate(-math.pi / 2)
        cr.move_to(0, 0)
        cr.show_text(label)
        cr.restore()

        label = f"{v['total_lines']} lines · {v['ppi']} ppi"
        if not v["background"]:
            label += " · transparent export"
        ext = cr.text_extents(label)
        cr.move_to(ox + (pw - ext.width) / 2, oy + ph + 22)
        cr.show_text(label)

        if lines_overflow(v):
            label = "Too many lines for this page"
            cr.set_font_size(12)
            ext = cr.text_extents(label)
            bx, by = ox + (pw - ext.width) / 2 - 10, oy + ph / 2 - 14
            cr.set_source_rgba(1, 1, 1, 0.92)
            cr.rectangle(bx, by, ext.width + 20, 28)
            cr.fill()
            cr.set_source_rgb(0.75, 0.1, 0.1)
            cr.move_to(bx + 10, by + 19)
            cr.show_text(label)


# --------------------------------------------------------------------------- #
# Color button
# --------------------------------------------------------------------------- #
class ColorButton(Gtk.MenuButton):
    """Color swatch that opens a popover chooser.

    Unlike Gtk.ColorDialogButton this does not open a separate top-level window,
    so compositor window rules (forced floating size, etc.) cannot clip it.
    """

    def __init__(self, color):
        super().__init__(valign=Gtk.Align.CENTER)
        self._rgba = color
        self._syncing = False

        self.swatch = Gtk.DrawingArea(content_width=40, content_height=20)
        self.swatch.set_draw_func(self._draw_swatch)
        self.set_child(self.swatch)

        self.chooser = Gtk.ColorChooserWidget(use_alpha=False, show_editor=False)
        self.chooser.set_rgba(color)
        self.chooser.connect("notify::rgba", self._on_chooser_changed)
        self.chooser.connect("color-activated", lambda *_: self.popdown())
        popover = Gtk.Popover(child=self.chooser)
        self.set_popover(popover)

    @GObject.Property(type=Gdk.RGBA)
    def rgba(self):
        return self._rgba

    @rgba.setter
    def rgba(self, value):
        self._rgba = value
        self.swatch.queue_draw()

    def get_rgba(self):
        return self._rgba

    def set_rgba(self, value):
        self._syncing = True
        try:
            self.chooser.set_rgba(value)
        finally:
            self._syncing = False
        self.rgba = value

    def _on_chooser_changed(self, chooser, _pspec):
        if not self._syncing:
            self.rgba = chooser.get_rgba()

    def _draw_swatch(self, area, cr, width, height):
        r = 5
        cr.new_sub_path()
        cr.arc(width - r, r, r, -math.pi / 2, 0)
        cr.arc(width - r, height - r, r, 0, math.pi / 2)
        cr.arc(r, height - r, r, math.pi / 2, math.pi)
        cr.arc(r, r, r, math.pi, 3 * math.pi / 2)
        cr.close_path()
        c = self._rgba
        cr.set_source_rgba(c.red, c.green, c.blue, 1)
        cr.fill_preserve()
        cr.set_source_rgba(0.5, 0.5, 0.5, 0.6)
        cr.set_line_width(1)
        cr.stroke()


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
# name, width mm, height mm (portrait)
PAPER_SIZES = [
    ("A3", 297, 420), ("A4", 210, 297), ("A5", 148, 210), ("A6", 105, 148),
    ("B5", 176, 250), ("Letter", 215.9, 279.4), ("Legal", 215.9, 355.6),
    ("Tabloid", 279.4, 431.8),
]
PAPER_NAMES = [p[0] for p in PAPER_SIZES] + ["Custom"]
PPI_CHOICES = [72, 96, 150, 300]
PPI_LABELS = ["72 ppi (standard)", "96 ppi (screen)", "150 ppi", "300 ppi (print)", "Custom…"]
DIM_KEYS = ("view_width", "view_height", "line_height", "margin_left", "margin_top")


class TypeScribeWindow(Adw.ApplicationWindow):
    def __init__(self, app):
        super().__init__(application=app, title="TypeScribe", default_width=1240, default_height=800)
        self._ready = False    # no refreshes until every widget exists
        self._loading = True   # no settings writes until startup is done
        self.reference_path = None  # layout-preview-only image (see _set_reference)
        self._updating = False  # suppress change handlers during programmatic updates
        self._batching = False  # several settings change at once; refresh when done
        self._save_id = 0
        self._ppi = DEFAULTS["ppi"]
        self.spins = {}
        self.colors = {}
        self.worker = None     # long-lived generator process (keeps the model loaded)
        self.worker_stdin = None
        self.run = None        # the job currently being generated, if any
        self.last_job = None   # the job behind the pages on screen (for regenerating a page)
        self.progress_fraction = None  # None = indeterminate
        self.log_path = Path(GLib.get_user_cache_dir()) / "typescribe" / "last-run.log"
        self.pulse_id = 0
        self.settings = migrate_settings(load_settings())
        self.profiles = self.settings["profiles"]
        self.active_profile = self.settings["active_profile"]
        remembered_dir = self.settings.get("output_dir")
        self.output_dir = remembered_dir if isinstance(remembered_dir, str) and remembered_dir else None
        self.export_opts = dict(EXPORT_DEFAULTS)  # edited only in the Export dialog
        self.result_pages = []
        self.result_index = 0

        self._build_sidebar()
        self._build_content()
        self._build_shell()
        self._install_actions()
        self._sync_profile_row()
        self._apply_ui_settings(self.profiles[self.active_profile])
        self._on_style_changed()
        self.connect("close-request", self._on_close)
        self._loading = False
        self._ready = True
        self._refresh()

    # ====================================================================== #
    # Sidebar: profile row + Handwriting tab + Page tab
    # ====================================================================== #
    def _spin_row(self, key, title, subtitle, lo, hi, step, digits):
        row = Adw.SpinRow.new_with_range(lo, hi, step)
        row.set_title(title)
        if subtitle:
            row.set_subtitle(subtitle)
        row.set_digits(digits)
        row.set_value(DEFAULTS[key])
        row.connect("notify::value", self._refresh)
        self.spins[key] = row
        return row

    def _color_row(self, key, title):
        button = ColorButton(parse_rgba(DEFAULTS[key]))
        button.connect("notify::rgba", self._refresh)
        row = Adw.ActionRow(title=title)
        row.add_suffix(button)
        row.set_activatable_widget(button)
        self.colors[key] = button
        return row

    def _build_sidebar(self):
        self.view_stack = Adw.ViewStack(vexpand=True)
        self.view_stack.add_titled_with_icon(
            self._build_handwriting_tab(), "handwriting", "Handwriting", "document-edit-symbolic",
        )
        self.view_stack.add_titled_with_icon(
            self._build_page_tab(), "page", "Page", "x-office-document-symbolic",
        )
        switcher = Adw.ViewSwitcher(policy=Adw.ViewSwitcherPolicy.WIDE, stack=self.view_stack)
        header = Adw.HeaderBar(show_end_title_buttons=False)
        header.set_title_widget(switcher)

        # The profile is the scope of everything below it, so it sits above both tabs.
        self.profile_row = Adw.ComboRow(
            title="Profile", tooltip_text="Handwriting and page settings are saved per profile",
        )
        self.profile_row.connect("notify::selected", self._on_profile_selected)
        profile_menu = Gio.Menu()
        edit_section = Gio.Menu()
        edit_section.append("New Profile…", "win.profile-new")
        edit_section.append("Rename…", "win.profile-rename")
        edit_section.append("Reset to Defaults…", "win.profile-reset")
        delete_section = Gio.Menu()
        delete_section.append("Delete Profile…", "win.profile-delete")
        profile_menu.append_section(None, edit_section)
        profile_menu.append_section(None, delete_section)
        manage_btn = Gtk.MenuButton(
            icon_name="view-more-symbolic", menu_model=profile_menu,
            valign=Gtk.Align.CENTER, tooltip_text="Manage profiles",
        )
        manage_btn.add_css_class("flat")
        self.profile_row.add_suffix(manage_btn)
        profile_group = Adw.PreferencesGroup(margin_top=12, margin_bottom=0, margin_start=12, margin_end=12)
        profile_group.add(self.profile_row)

        body = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        body.append(profile_group)
        body.append(self.view_stack)
        self.sidebar_view = Adw.ToolbarView(content=body)
        self.sidebar_view.add_top_bar(header)

    def _build_handwriting_tab(self):
        page = Adw.PreferencesPage()

        style = Adw.PreferencesGroup(title="Writing style")
        self.style_row = Adw.ComboRow(
            title="Style", model=Gtk.StringList.new([f"Style {i}" for i in range(1, 13)]),
        )
        self.style_row.connect("notify::selected", self._on_style_changed)
        style.add(self.style_row)
        self.style_sample = Gtk.Picture(can_shrink=False, halign=Gtk.Align.CENTER, width_request=176, height_request=56)
        sample_box = Gtk.Box(halign=Gtk.Align.CENTER, margin_top=6, margin_bottom=6)
        sample_box.add_css_class("style-sample")
        sample_box.append(self.style_sample)
        style.add(sample_box)
        page.add(style)

        strokes = Adw.PreferencesGroup(title="Strokes")
        strokes.add(self._spin_row("bias", "Consistency", "Higher is neater, lower is looser", 0.0, 2.0, 0.05, 2))
        strokes.add(self._spin_row("pen_width", "Pen thickness", None, 0.1, 10.0, 0.1, 1))
        strokes.add(self._spin_row("pressure", "Pen pressure", "Variation in line weight", 0.0, 1.0, 0.05, 2))
        page.add(strokes)

        ink = Adw.PreferencesGroup(title="Ink")
        ink.add(self._color_row("ink_color", "Ink color"))
        self.ink_mode_row = Adw.ComboRow(
            title="Ink mix", subtitle="Also write with a second color", model=Gtk.StringList.new(INK_MODES),
        )
        self.ink_mode_row.connect("notify::selected", self._on_ink_mode_changed)
        ink.add(self.ink_mode_row)
        self.ink2_row = self._color_row("ink_color2", "Second ink color")
        self.ink2_row.set_visible(False)
        ink.add(self.ink2_row)
        page.add(ink)
        return page

    def _build_page_tab(self):
        page = Adw.PreferencesPage()

        # Paper: what the sheet is.
        paper = Adw.PreferencesGroup(title="Paper")
        self.paper_row = Adw.ComboRow(title="Paper size", model=Gtk.StringList.new(PAPER_NAMES))
        self.paper_row.set_selected(PAPER_NAMES.index(DEFAULTS["paper"]))
        self.paper_row.connect("notify::selected", self._on_paper_changed)
        paper.add(self.paper_row)

        self.orientation_row = Adw.ComboRow(
            title="Orientation", model=Gtk.StringList.new(["Portrait", "Landscape"]),
        )
        self.orientation_row.connect("notify::selected", self._on_orientation_changed)
        paper.add(self.orientation_row)

        self.ppi_row = Adw.ComboRow(
            title="Resolution", subtitle="Pixel density of every size below",
            model=Gtk.StringList.new(PPI_LABELS),
        )
        self.ppi_row.set_selected(PPI_CHOICES.index(DEFAULTS["ppi"]))
        self.ppi_row.connect("notify::selected", self._on_ppi_changed)
        paper.add(self.ppi_row)

        ppi_custom = self._spin_row("ppi_custom", "Custom resolution", "ppi", 10, 1200, 1, 0)
        ppi_custom.set_visible(False)
        ppi_custom.connect("notify::value", self._on_ppi_changed)
        paper.add(ppi_custom)

        for key, title in (("view_width", "Page width"), ("view_height", "Page height")):
            row = self._spin_row(key, title, "px", 50, 30000, 1, 2)
            row.connect("notify::value", self._on_dims_edited)
            paper.add(row)
        page.add(paper)

        # Ruling: where the lines and margins are drawn.
        ruling = Adw.PreferencesGroup(title="Ruling")
        ruling.add(self._spin_row("line_height", "Line height", "px", 4, 3000, 0.5, 2))
        ruling.add(self._spin_row("total_lines", "Ruled lines per page", None, 1, 200, 1, 0))
        ruling.add(self._spin_row("margin_left", "Left margin", "px", 0, 10000, 0.5, 2))
        ruling.add(self._spin_row("margin_top", "Top margin", "px", 0, 10000, 0.5, 2))
        page.add(ruling)

        # Text on the page: how the text is wrapped onto the ruling.
        text = Adw.PreferencesGroup(title="Text on the page")
        text.add(self._spin_row("line_length", "Characters per line", "Wrap limit", 20, 75, 1, 0))
        text.add(self._spin_row("lines_per_page", "Lines written per page", "Up to the ruled lines", 1, 200, 1, 0))
        fit_row = Adw.ActionRow(
            title="Fit to page", subtitle="Fill the page with ruled lines and set the wrap limit",
        )
        fit_btn = Gtk.Button(label="Fit", valign=Gtk.Align.CENTER)
        fit_btn.connect("clicked", self._on_fit)
        fit_row.add_suffix(fit_btn)
        fit_row.set_activatable_widget(fit_btn)
        text.add(fit_row)
        page.add(text)
        # You can never write more lines than are ruled.
        self.spins["total_lines"].connect("notify::value", self._sync_line_limits)

        colors = Adw.PreferencesGroup(title="Colors")
        colors.add(self._color_row("page_color", "Page color"))
        colors.add(self._color_row("margin_color", "Margin color"))
        colors.add(self._color_row("line_color", "Line color"))
        page.add(colors)

        # A picture of the real paper to line the layout up against.
        reference = Adw.PreferencesGroup(
            title="Reference image",
            description="Shown behind the layout preview to help you set the page up. "
                        "It is never used when generating or exporting.",
        )
        self.reference_row = Adw.ActionRow(title="Image", subtitle="None chosen", subtitle_lines=1)
        self.reference_clear_btn = Gtk.Button(
            icon_name="edit-clear-symbolic", valign=Gtk.Align.CENTER, visible=False,
            tooltip_text="Remove the reference image",
        )
        self.reference_clear_btn.add_css_class("flat")
        self.reference_clear_btn.connect("clicked", lambda *_: self._set_reference(None))
        choose_reference = Gtk.Button(label="Choose…", valign=Gtk.Align.CENTER)
        choose_reference.connect("clicked", self._on_choose_reference)
        self.reference_row.add_suffix(self.reference_clear_btn)
        self.reference_row.add_suffix(choose_reference)
        reference.add(self.reference_row)
        self.reference_fit_row = Adw.ComboRow(
            title="Fit", model=Gtk.StringList.new(REFERENCE_FITS), visible=False,
        )
        self.reference_fit_row.connect("notify::selected", self._refresh)
        reference.add(self.reference_fit_row)
        self.reference_opacity_row = self._spin_row("reference_opacity", "Opacity", "%", 5, 100, 5, 0)
        self.reference_opacity_row.set_visible(False)
        reference.add(self.reference_opacity_row)
        page.add(reference)
        return page

    # ====================================================================== #
    # Content: header, text pane, canvas pane, status bar
    # ====================================================================== #
    def _build_content(self):
        self.sidebar_toggle = Gtk.ToggleButton(icon_name="sidebar-show-symbolic", tooltip_text="Settings (F9)")
        open_btn = Gtk.Button(
            icon_name="document-open-symbolic", tooltip_text="Open text file (Ctrl+O)",
            action_name="win.open-text",
        )

        menu = Gio.Menu()
        menu.append("About TypeScribe", "win.about")
        menu_btn = Gtk.MenuButton(icon_name="open-menu-symbolic", menu_model=menu, primary=True,
                                  tooltip_text="Main menu")

        self.generate_btn = Gtk.Button(label="Generate", action_name="win.generate")
        self.generate_btn.add_css_class("suggested-action")
        self.export_btn = Gtk.Button(label="Export…", action_name="win.export")

        self.window_title = Adw.WindowTitle(title="TypeScribe", subtitle="")
        header = Adw.HeaderBar()
        header.set_title_widget(self.window_title)
        header.pack_start(self.sidebar_toggle)
        header.pack_start(open_btn)
        header.pack_end(menu_btn)
        header.pack_end(self.export_btn)
        header.pack_end(self.generate_btn)

        self.status = Gtk.Label(xalign=0, hexpand=True, ellipsize=Pango.EllipsizeMode.END)
        self.status.add_css_class("dim-label")
        self.progress = Gtk.ProgressBar(valign=Gtk.Align.CENTER, visible=False, width_request=180)
        bar = Gtk.Box(spacing=12, margin_top=8, margin_bottom=8, margin_start=14, margin_end=14)
        bar.append(self.status)
        bar.append(self.progress)

        paned = Gtk.Paned(
            orientation=Gtk.Orientation.HORIZONTAL,
            start_child=self._build_text_pane(), end_child=self._build_canvas_pane(),
            resize_start_child=True, resize_end_child=True,
            shrink_start_child=False, shrink_end_child=False, position=440,
        )
        self.content_view = Adw.ToolbarView(content=paned)
        self.content_view.add_top_bar(header)
        self.content_view.add_bottom_bar(bar)

    def _pane(self, header_widget, body):
        """A pane = a fixed-height header strip, a separator, then the body."""
        header = Gtk.Box(spacing=8, margin_start=14, margin_end=14, height_request=48)
        header.append(header_widget)
        pane = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, hexpand=True, vexpand=True)
        pane.append(header)
        pane.append(Gtk.Separator())
        pane.append(body)
        return pane

    def _build_text_pane(self):
        self.buffer = Gtk.TextBuffer()
        self.buffer.connect("changed", self._on_text_changed)
        self.text_view = Gtk.TextView(
            buffer=self.buffer, wrap_mode=Gtk.WrapMode.WORD_CHAR,
            top_margin=14, bottom_margin=14, left_margin=18, right_margin=18,
        )
        scroller = Gtk.ScrolledWindow(child=self.text_view, hexpand=True, vexpand=True)

        self.placeholder = Gtk.Label(
            label="Type or paste some text to begin",
            xalign=0, yalign=0, margin_top=14, margin_start=22, can_target=False,
        )
        self.placeholder.add_css_class("dim-label")
        overlay = Gtk.Overlay(child=scroller)
        overlay.add_overlay(self.placeholder)

        title = Gtk.Label(label="Text", xalign=0, hexpand=True)
        title.add_css_class("heading")
        return self._pane(title, overlay)

    def _build_canvas_pane(self):
        self.preview = PagePreview(self.values, self._on_layout_drag)
        hint = Gtk.Label(
            label="Drag the page edges, margin lines and handles to adjust the layout",
            wrap=True, justify=Gtk.Justification.CENTER,
            margin_bottom=10, margin_start=14, margin_end=14,
        )
        hint.add_css_class("dim-label")
        hint.add_css_class("caption")
        layout_view = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        layout_view.append(self.preview)
        layout_view.append(hint)

        self.canvas_stack = Gtk.Stack(
            vexpand=True, hexpand=True, transition_type=Gtk.StackTransitionType.CROSSFADE,
        )
        self.canvas_stack.add_named(layout_view, "layout")
        self.canvas_stack.add_named(self._build_result_view(), "pages")

        # Two-way tab control; "Pages" stays disabled until something was generated.
        self.layout_tab = Gtk.ToggleButton(label="Layout", active=True,
                                           tooltip_text="Live preview of the ruled page")
        self.pages_tab = Gtk.ToggleButton(label="Pages", group=self.layout_tab, sensitive=False,
                                          tooltip_text="Generate to see the written pages")
        self.layout_tab.connect("toggled", self._on_canvas_tab)
        self.pages_tab.connect("toggled", self._on_canvas_tab)
        tabs = Gtk.Box(valign=Gtk.Align.CENTER)
        tabs.add_css_class("linked")
        tabs.append(self.layout_tab)
        tabs.append(self.pages_tab)
        return self._pane(tabs, self.canvas_stack)

    def _build_result_view(self):
        self.picture = Gtk.Picture(
            can_shrink=True, content_fit=Gtk.ContentFit.CONTAIN,
            vexpand=True, hexpand=True, margin_top=12, margin_bottom=6, margin_start=12, margin_end=12,
        )
        self.prev_btn = Gtk.Button(icon_name="go-previous-symbolic", tooltip_text="Previous page")
        self.next_btn = Gtk.Button(icon_name="go-next-symbolic", tooltip_text="Next page")
        self.prev_btn.connect("clicked", lambda *_: self._show_page(self.result_index - 1))
        self.next_btn.connect("clicked", lambda *_: self._show_page(self.result_index + 1))
        self.page_label = Gtk.Label(label="")
        self.page_label.add_css_class("dim-label")
        self.regen_btn = Gtk.Button(
            child=Adw.ButtonContent(icon_name="view-refresh-symbolic", label="Regenerate page"),
            tooltip_text="Write this page again",
        )
        self.regen_btn.connect("clicked", self._on_regenerate)

        nav = Gtk.Box(spacing=8, margin_bottom=10, margin_start=12, margin_end=12, halign=Gtk.Align.CENTER)
        nav.append(self.prev_btn)
        nav.append(self.page_label)
        nav.append(self.next_btn)
        nav.append(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL, margin_start=6, margin_end=6))
        nav.append(self.regen_btn)

        # Light backdrop so ink stays visible on transparent exports in dark mode.
        paper = Gtk.Box(vexpand=True, hexpand=True)
        paper.add_css_class("result-paper")
        paper.append(self.picture)

        pages = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        pages.append(paper)
        pages.append(nav)
        return pages

    def _build_shell(self):
        self.split = Adw.OverlaySplitView(
            sidebar=self.sidebar_view, content=self.content_view,
            min_sidebar_width=340, max_sidebar_width=420, sidebar_width_fraction=0.3,
        )
        self.split.bind_property(
            "show-sidebar", self.sidebar_toggle, "active",
            GObject.BindingFlags.BIDIRECTIONAL | GObject.BindingFlags.SYNC_CREATE,
        )
        breakpoint_ = Adw.Breakpoint.new(Adw.BreakpointCondition.parse("max-width: 900sp"))
        breakpoint_.add_setter(self.split, "collapsed", True)
        self.add_breakpoint(breakpoint_)

        self.toasts = Adw.ToastOverlay(child=self.split)
        self.set_content(self.toasts)

    def _install_actions(self):
        for name, handler in (
            ("generate", self._on_generate),
            ("export", self._on_export),
            ("open-text", self._on_open_text),
            ("toggle-sidebar", lambda *_: self.split.set_show_sidebar(not self.split.get_show_sidebar())),
            ("profile-new", self._on_profile_new),
            ("profile-rename", self._on_profile_rename),
            ("profile-reset", self._on_profile_reset),
            ("profile-delete", self._on_profile_delete),
            ("about", self._on_about),
        ):
            action = Gio.SimpleAction.new(name, None)
            action.connect("activate", handler)
            self.add_action(action)

    def _set_action_enabled(self, name, enabled):
        action = self.lookup_action(name)
        if action:
            action.set_enabled(enabled)

    # ====================================================================== #
    # State
    # ====================================================================== #
    def get_text(self):
        start, end = self.buffer.get_bounds()
        return self.buffer.get_text(start, end, False)

    # ---- reference image (layout preview only) --------------------------------- #
    def _on_choose_reference(self, *_):
        dialog = Gtk.FileDialog(title="Choose Reference Image")
        images = Gtk.FileFilter(name="Images")
        images.add_pixbuf_formats()
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(images)
        dialog.set_filters(filters)
        if self.reference_path:
            dialog.set_initial_file(Gio.File.new_for_path(self.reference_path))
        dialog.open(self, None, self._on_reference_chosen)

    def _on_reference_chosen(self, dialog, result):
        try:
            path = dialog.open_finish(result).get_path()
        except GLib.Error:
            return  # dismissed
        if path and not self._set_reference(path):
            self.toast("Could not open that image")

    def _set_reference(self, path):
        """Show `path` behind the layout preview (None removes it); False if it can't be read."""
        surface = None
        if path:
            try:
                surface = load_reference_surface(path)
            except (GLib.Error, OSError, ValueError, cairo.Error):
                return False  # keep whatever was showing
        self.reference_path = path if surface else None
        self.reference_row.set_subtitle(GLib.markup_escape_text(Path(path).name) if surface else "None chosen")
        self.reference_row.set_tooltip_text(self.reference_path)
        for widget in (self.reference_clear_btn, self.reference_fit_row, self.reference_opacity_row):
            widget.set_visible(surface is not None)
        if hasattr(self, "preview"):
            self.preview.set_reference(surface)
        self._queue_save()
        return True

    def values(self):
        s = self.spins
        v = {
            "line_length": int(s["line_length"].get_value()),
            "lines_per_page": int(s["lines_per_page"].get_value()),
            "total_lines": int(s["total_lines"].get_value()),
            "bias": s["bias"].get_value(),
            "pen_width": s["pen_width"].get_value(),
            "line_height": float(s["line_height"].get_value()),
            "view_height": s["view_height"].get_value(),
            "view_width": s["view_width"].get_value(),
            "margin_left": float(s["margin_left"].get_value()),
            "margin_top": float(s["margin_top"].get_value()),
            "style": self.style_row.get_selected() + 1,
            "ink_mode": self.ink_mode_row.get_selected(),
            "pressure": s["pressure"].get_value(),
            "ppi": self.current_ppi(),
            # Layout-preview aid: read only by the preview, never put into a job or export.
            "reference_opacity": s["reference_opacity"].get_value() / 100,
            "reference_fit": self.reference_fit_row.get_selected(),
            # An export option (set in the Export dialog); the layout preview mirrors it.
            "background": self.export_opts["background"] or not CAN_STRIP_BACKGROUND,
        }
        for key, button in self.colors.items():
            v[key] = rgba_to_hex(button.get_rgba())
        return v

    # ---- paper size / resolution ------------------------------------------ #
    def current_ppi(self):
        index = self.ppi_row.get_selected()
        if index < len(PPI_CHOICES):
            return PPI_CHOICES[index]
        return max(1, int(self.spins["ppi_custom"].get_value()))

    def _is_custom_paper(self):
        return self.paper_row.get_selected() == len(PAPER_NAMES) - 1

    def _apply_paper_dims(self):
        """Set page width/height (px at the current ppi) from the paper preset."""
        name = PAPER_NAMES[self.paper_row.get_selected()]
        for preset, w_mm, h_mm in PAPER_SIZES:
            if preset == name:
                if self.orientation_row.get_selected() == 1:
                    w_mm, h_mm = h_mm, w_mm
                ppi = self.current_ppi()
                self.spins["view_width"].set_value(round(w_mm / 25.4 * ppi, 2))
                self.spins["view_height"].set_value(round(h_mm / 25.4 * ppi, 2))
                return

    def _autofit(self):
        """Fit ruled lines and characters per line to the current page."""
        s = self.spins
        fit = fit_lines(s["view_height"].get_value(), s["margin_top"].get_value(), s["line_height"].get_value())
        s["total_lines"].set_value(fit)
        s["lines_per_page"].set_value(fit)
        s["line_length"].set_value(
            fit_chars(s["view_width"].get_value(), s["margin_left"].get_value(), self.current_ppi())
        )

    def _on_fit(self, *_):
        self._autofit()
        self._refresh()

    def _on_layout_drag(self, handle, start, dx, dy):
        """The layout preview was dragged: update the same settings the sidebar edits."""
        self._apply_layout_edit(layout_drag_changes(handle, start, dx, dy))

    def _apply_layout_edit(self, changes):
        self._batching = True
        try:
            for key, value in changes.items():  # ordered: ruled lines before lines written
                self.spins[key].set_value(value)
        finally:
            self._batching = False
        self._refresh()

    def _sync_line_limits(self, *_):
        total = max(1, int(self.spins["total_lines"].get_value()))
        self.spins["lines_per_page"].set_range(1, total)

    def _on_paper_changed(self, *_):
        if self._updating:
            return
        if not self._is_custom_paper():
            self._updating = True
            self._apply_paper_dims()
            self._autofit()
            self._updating = False
        self._refresh()

    def _on_orientation_changed(self, *_):
        if self._updating:
            return
        if self._is_custom_paper():
            self._updating = True
            width, height = self.spins["view_width"].get_value(), self.spins["view_height"].get_value()
            self.spins["view_width"].set_value(height)
            self.spins["view_height"].set_value(width)
            self._updating = False
            self._refresh()
        else:
            self._on_paper_changed()

    def _on_dims_edited(self, *_):
        """Typing a page size by hand switches the paper to Custom."""
        if self._updating or self._is_custom_paper():
            return
        self._updating = True
        self.paper_row.set_selected(len(PAPER_NAMES) - 1)
        self._updating = False

    def _on_ppi_changed(self, *_):
        self.spins["ppi_custom"].set_visible(self.ppi_row.get_selected() == len(PPI_CHOICES))
        if self._updating:
            return
        new = self.current_ppi()
        if new == self._ppi:
            return
        # Keep the physical layout identical: rescale every px value.
        factor = new / self._ppi
        self._updating = True
        for key in DIM_KEYS:
            spin = self.spins[key]
            spin.set_value(round(spin.get_value() * factor, spin.get_digits()))
        self._ppi = new
        if not self._is_custom_paper():
            self._apply_paper_dims()
        self._updating = False
        self._refresh()

    def _on_style_changed(self, *_):
        sample = ASSETS / f"font{self.style_row.get_selected() + 1}.png"
        self.style_sample.set_file(Gio.File.new_for_path(str(sample)))
        self._refresh()

    def _on_ink_mode_changed(self, *_):
        self.ink2_row.set_visible(self.ink_mode_row.get_selected() != 0)
        self._refresh()

    def _on_text_changed(self, *_):
        self.placeholder.set_visible(self.buffer.get_char_count() == 0)
        if self.export_opts.get("name_auto", True):
            self.export_opts["name"] = derive_base_name(self.get_text())
        self._refresh()

    # ---- derived UI state -------------------------------------------------- #
    def _refresh(self, *_):
        """Re-derive everything that depends on settings or text."""
        if not self._ready or self._batching:
            return
        self._queue_save()
        self.preview.queue_draw()
        ppi = self.current_ppi()
        mm = lambda key: self.spins[key].get_value() / ppi * 25.4  # noqa: E731
        self.paper_row.set_subtitle(f"{mm('view_width'):.0f} × {mm('view_height'):.0f} mm")
        for key in DIM_KEYS:  # every px value also shows its size in inches
            px = self.spins[key].get_value()
            self.spins[key].set_subtitle(f"px  ({px / ppi:.2f} in)")
        self._update_actions()
        self._update_status()

    def _can_generate(self, v, text):
        return bool(text.strip()) and not lines_overflow(v)

    def _is_stale(self, v, text):
        return bool(self.result_pages) and self.last_job is not None and self._build_job(text, v) != self.last_job

    def _update_actions(self):
        running = self.run is not None
        v, text = self.values(), self.get_text()
        self._set_action_enabled("generate", running or self._can_generate(v, text))
        self._set_action_enabled("export", bool(self.result_pages) and not running)
        self._set_action_enabled("open-text", not running)
        self.generate_btn.set_tooltip_text(
            "Stop writing" if running else "Write the text by hand (Ctrl+Enter)"
        )
        self.export_btn.set_tooltip_text(
            "Save the generated pages (Ctrl+E)" if self.result_pages else "Generate pages first"
        )
        self.regen_btn.set_sensitive(bool(self.result_pages) and self.last_job is not None and not running)

    def _set_status(self, text, warn=False):
        self.status.set_text(text)
        if warn:
            self.status.remove_css_class("dim-label")
            self.status.add_css_class("warning")
        else:
            self.status.remove_css_class("warning")
            self.status.add_css_class("dim-label")

    def _update_status(self):
        if self.run:  # the worker's events own the status line while writing
            return
        v, text = self.values(), self.get_text()
        parts, warn = [], False
        if text.strip():
            pages = layout_pages(text, v["line_length"], v["lines_per_page"])
            lines = sum(len(p) for p in pages)
            parts.append(f"{len(pages)} page{'s' if len(pages) != 1 else ''} · {lines} lines")
            skipped = unsupported_chars(text)
            if skipped:
                shown = " ".join(dict.fromkeys(skipped[:40]).keys())[:20].strip()
                parts.append(f"{len(skipped)} unsupported character{'s' if len(skipped) != 1 else ''} "
                             f"will be skipped ({shown})")
                warn = True
        else:
            parts.append("Type or paste some text to begin")
        if lines_overflow(v):
            parts.append("Ruled lines run past the bottom of the page")
            warn = True
        if self._is_stale(v, text):
            parts.append("Generated pages are out of date")
        self._set_status(" · ".join(parts), warn)

    # ====================================================================== #
    # Canvas tabs and page browser
    # ====================================================================== #
    def _on_canvas_tab(self, button):
        if button.get_active():
            self.canvas_stack.set_visible_child_name("pages" if button is self.pages_tab else "layout")

    def _show_canvas(self, name):
        (self.pages_tab if name == "pages" else self.layout_tab).set_active(True)

    def _show_page(self, index):
        if not self.result_pages:
            return
        index = max(0, min(index, len(self.result_pages) - 1))
        self.result_index = index
        svg = self.result_pages[index]
        preview = svg + ".png" if os.path.exists(svg + ".png") else svg
        self.picture.set_file(None)  # drop the cached image so a rewritten page reloads
        self.picture.set_file(Gio.File.new_for_path(preview))
        self.page_label.set_text(f"Page {index + 1} of {len(self.result_pages)}")
        self.prev_btn.set_sensitive(index > 0)
        self.next_btn.set_sensitive(index < len(self.result_pages) - 1)

    # ====================================================================== #
    # Header actions: open text, generate
    # ====================================================================== #
    def _on_open_text(self, *_):
        dialog = Gtk.FileDialog(title="Open Text File")
        text_filter = Gtk.FileFilter(name="Text files")
        text_filter.add_mime_type("text/plain")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(text_filter)
        dialog.set_filters(filters)
        dialog.open(self, None, self._on_text_opened)

    def _on_text_opened(self, dialog, result):
        try:
            gfile = dialog.open_finish(result)
            ok, data, _etag = gfile.load_contents(None)
        except GLib.Error:
            return
        self.buffer.set_text(data.decode("utf-8", "replace"))
        self._show_canvas("layout")

    def _on_generate(self, *_):
        if self.run:  # the same action cancels a running job
            self._cancel()
            return
        text, v = self.get_text(), self.values()
        if not self._can_generate(v, text):
            return  # the action is disabled in this state; the status bar says why
        self._start(self._build_job(text, v))

    def _build_job(self, text, v):
        to_units = 72 / v["ppi"]
        return {
            "text": text,
            "output_dir": str(PREVIEW_DIR),
            "max_line_length": v["line_length"],
            "lines_per_page": v["lines_per_page"],
            "bias": v["bias"],
            "style": v["style"],
            "ink_color": v["ink_color"],
            "ink_color2": v["ink_color2"],
            "ink_mode": v["ink_mode"],
            "pressure": v["pressure"],
            "pen_width": v["pen_width"],
            # The model lays out in 72-ppi units; the export is scaled to the chosen ppi.
            # The paper is always rendered; "transparent export" strips it at export time,
            # so that choice never costs a regeneration.
            "page": [
                v["line_height"] * to_units, v["total_lines"], v["view_height"] * to_units,
                v["view_width"] * to_units, -v["margin_left"] * to_units, -v["margin_top"] * to_units,
                v["page_color"], v["margin_color"], v["line_color"],
                True, v["ppi"] / 72,
            ],
        }

    def _on_regenerate(self, *_):
        if self.run or not self.result_pages or not self.last_job:
            return
        self._start(self.last_job, only_page=self.result_index + 1)

    # ---- worker process ---------------------------------------------------- #
    def _ensure_worker(self):
        """Start the generator once; it then stays up so the model is loaded only once."""
        if self.worker:
            return True
        python = find_model_python()
        if not python:
            self.alert(
                "Model environment not found",
                f"TypeScribe needs a Python environment with TensorFlow. Create it with\n"
                f"conda env create -f environment.yml\n(environment name: {MODEL_ENV_NAME}), "
                f"or point the TYPESCRIBE_PYTHON variable at its python executable.",
            )
            return False
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        launcher = Gio.SubprocessLauncher.new(
            Gio.SubprocessFlags.STDIN_PIPE | Gio.SubprocessFlags.STDOUT_PIPE
        )
        launcher.set_cwd(str(APP_DIR))
        launcher.set_stderr_file_path(str(self.log_path))
        launcher.setenv("PYTHONUNBUFFERED", "1", True)
        launcher.setenv("TF_CPP_MIN_LOG_LEVEL", "2", True)
        try:
            worker = launcher.spawnv([python, str(APP_DIR / "generate.py")])
        except GLib.Error as err:
            self.alert("Could not start the generator", err.message)
            return False
        self.worker = worker
        self.worker_stdin = worker.get_stdin_pipe()
        reader = Gio.DataInputStream.new(worker.get_stdout_pipe())
        reader.read_line_async(GLib.PRIORITY_DEFAULT, None, self._on_line, worker)
        return True

    def _kill_worker(self):
        worker, self.worker = self.worker, None
        if worker:
            worker.force_exit()

    def _start(self, job, only_page=None):
        if not self._ensure_worker():
            return
        if only_page is None:
            PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
            for path in PREVIEW_DIR.iterdir():
                if path.is_file():
                    path.unlink()
            message = job
        else:  # write just this page again; the others stay as they are
            message = {**job, "only_pages": [only_page]}

        self.run = {"job": job, "only_page": only_page}
        self.progress_fraction = None
        self._set_running(True)
        self._set_status("Starting…")
        try:
            self.worker_stdin.write_all((json.dumps(message) + "\n").encode("utf-8"), None)
        except GLib.Error as err:
            self._kill_worker()
            self._finish_run(error=f"Could not talk to the generator: {err.message}")

    def _cancel(self):
        # Killing the worker is the only way to stop the model mid-page; the next
        # Generate starts a fresh one (and reloads the model).
        self._kill_worker()
        self._finish_run(cancelled=True)

    def _set_running(self, running):
        self.sidebar_view.set_sensitive(not running)
        self.preview.set_sensitive(not running)  # no dragging while a page is being written
        self.preview.queue_draw()
        self.text_view.set_editable(not running)
        self.progress.set_visible(running)
        self.generate_btn.set_label("Cancel" if running else "Generate")
        if running:
            self.generate_btn.remove_css_class("suggested-action")
            self.generate_btn.add_css_class("destructive-action")
            if not self.pulse_id:
                self.pulse_id = GLib.timeout_add(120, self._pulse)
        else:
            self.progress.set_fraction(0)
            self.generate_btn.remove_css_class("destructive-action")
            self.generate_btn.add_css_class("suggested-action")
        self._update_actions()

    def _pulse(self):
        if not self.run:
            self.pulse_id = 0
            return False
        if self.progress_fraction is None:  # model loading / first page: no measurable progress yet
            self.progress.pulse()
        return True

    def _on_line(self, stream, result, worker):
        try:
            line, _length = stream.read_line_finish_utf8(result)
        except GLib.Error:
            line = None
        if line is None:  # EOF: this worker has exited
            worker.wait_async(None, self._on_worker_exited)
            return

        if worker is self.worker and self.run:  # ignore anything from a worker we killed
            try:
                event = json.loads(line)
            except ValueError:
                event = {}
            kind = event.get("event")
            if kind == "loading":
                self.progress_fraction = None
                self._set_status("Loading the handwriting model…")
            elif kind == "page_start":
                self._set_status(f"Writing page {event.get('n')} of {event.get('total')}…")
            elif kind == "page_done":
                self.progress_fraction = event.get("index", 0) / max(1, event.get("count", 1))
                self.progress.set_fraction(self.progress_fraction)
            elif kind == "done":
                self._finish_run(files=event.get("files", []))
            elif kind == "error":
                self._finish_run(error=event.get("message", "Unknown error"))
        stream.read_line_async(GLib.PRIORITY_DEFAULT, None, self._on_line, worker)

    def _on_worker_exited(self, proc, result):
        try:
            proc.wait_finish(result)
        except GLib.Error:
            pass
        if proc is not self.worker:  # one we killed on purpose
            return
        self.worker = None
        if self.run:
            self._finish_run(error="The generator exited unexpectedly.")

    def _finish_run(self, files=None, error=None, cancelled=False):
        run, self.run = self.run, None
        succeeded = not cancelled and not error and bool(files)
        index = 0
        if succeeded:
            if run["only_page"] is None:
                self.result_pages = files
                self.last_job = run["job"]
            else:
                index = run["only_page"] - 1
        self._set_running(False)
        self._refresh()

        if cancelled:
            self.toast("Generation cancelled")
        elif not succeeded:
            tail = ""
            try:
                tail = "\n".join(self.log_path.read_text(errors="replace").splitlines()[-15:])
            except OSError:
                pass
            self.alert("Generation failed", error or "The generator produced no pages.", tail or None)
        else:
            self.pages_tab.set_sensitive(True)
            self._show_page(index)
            self._show_canvas("pages")
            count = len(self.result_pages)
            if run["only_page"] is None:
                self.toast(f"Generated {count} page{'s' if count != 1 else ''}")
            else:
                self.toast(f"Page {run['only_page']} written again")

    # ====================================================================== #
    # Export dialog - the only place that touches folder, name, formats
    # ====================================================================== #
    def _base_name(self):
        name = self.export_opts["name"]
        if self.export_opts.get("name_auto", True):
            name = derive_base_name(self.get_text())
        name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name.strip()).strip(". ")
        return name or derive_base_name(self.get_text())

    def _set_export_name(self, value):
        """Remember a typed file name; clearing it goes back to following the text."""
        self.export_opts["name"] = value
        typed = value.strip()
        self.export_opts["name_auto"] = not typed or typed == derive_base_name(self.get_text())
        self._queue_save()

    def _on_export(self, *_):
        if not self.result_pages or self.run:
            return
        count = len(self.result_pages)
        opts = self.export_opts

        dialog = Adw.Dialog(title="Export", content_width=460)
        cancel_btn = Gtk.Button(label="Cancel")
        cancel_btn.connect("clicked", lambda *_: dialog.close())
        export_btn = Gtk.Button(label="Export")
        export_btn.add_css_class("suggested-action")
        header = Adw.HeaderBar(show_start_title_buttons=False, show_end_title_buttons=False)
        header.set_title_widget(
            Adw.WindowTitle(title="Export", subtitle=f"{count} page{'s' if count != 1 else ''}")
        )
        header.pack_start(cancel_btn)
        header.pack_end(export_btn)

        # Destination: where and under which name.
        destination = Adw.PreferencesGroup(
            title="Destination",
            description="Pages are saved as name_page_1, name_page_2 …  or, as a PDF, one name.pdf.",
        )
        folder_row = Adw.ActionRow(title="Folder", subtitle_lines=1)
        open_folder_btn = Gtk.Button(icon_name="folder-open-symbolic", valign=Gtk.Align.CENTER,
                                     tooltip_text="Open this folder")
        open_folder_btn.add_css_class("flat")
        open_folder_btn.connect("clicked", self._open_output_folder)
        choose_btn = Gtk.Button(label="Choose…", valign=Gtk.Align.CENTER)
        folder_row.add_suffix(open_folder_btn)
        folder_row.add_suffix(choose_btn)
        destination.add(folder_row)
        name_row = Adw.EntryRow(title="File name")
        name_row.set_text(opts["name"])
        destination.add(name_row)

        # Formats.
        formats = Adw.PreferencesGroup(title="Formats")
        # Page files: choose one format (radio-style rows).
        format_radios, format_rows = {}, {}
        for key, title in (("svg", "SVG"), ("png", "PNG")):
            radio = Gtk.CheckButton(valign=Gtk.Align.CENTER)
            if format_radios:
                radio.set_group(next(iter(format_radios.values())))
            row = Adw.ActionRow(title=title, activatable_widget=radio)
            row.add_prefix(radio)
            format_radios[key], format_rows[key] = radio, row
            formats.add(row)
        format_radios[opts["format"]].set_active(True)

        # Instead of separate pages, save them all as a single PDF.
        pdf_check = Gtk.CheckButton(active=opts["pdf"], valign=Gtk.Align.CENTER)
        pdf_row = Adw.ActionRow(title="Combine all pages into one PDF", activatable_widget=pdf_check)
        pdf_row.add_prefix(pdf_check)
        formats.add(pdf_row)

        # Page background (the single home of this option).
        paper = Adw.PreferencesGroup(title="Paper")
        background_row = Adw.SwitchRow(
            title="Include page background",
            subtitle="Paper color, ruled lines and margins. Turn off for a transparent export.",
            active=opts["background"] or not CAN_STRIP_BACKGROUND,
        )
        if not CAN_STRIP_BACKGROUND:
            background_row.set_sensitive(False)
            background_row.set_subtitle("Always included: leaving it out needs librsvg")
        paper.add(background_row)

        def sync(*_):
            folder_row.set_subtitle(GLib.markup_escape_text(self.output_dir or "Choose where to save the files"))
            folder_row.set_tooltip_text(self.output_dir)
            open_folder_btn.set_sensitive(bool(self.output_dir))
            if opts["pdf"]:  # the choice above then decides what the PDF is made from
                format_rows["svg"].set_subtitle("PDF built from the vector pages: sharp at any zoom")
                format_rows["png"].set_subtitle("PDF built from the image pages, at the exported resolution")
                pdf_row.set_subtitle("Saves a single PDF instead of separate page files")
            else:
                format_rows["svg"].set_subtitle("Vector, one file per page")
                format_rows["png"].set_subtitle("Image, one file per page")
                pdf_row.set_subtitle("Save one PDF instead of one file per page")
            export_btn.set_sensitive(bool(self.output_dir))

        def on_name(row):
            self._set_export_name(row.get_text())

        def on_format(button, key):
            if button.get_active():  # a radio also fires when it is deselected
                opts["format"] = key
                self._queue_save()
                sync()

        def on_pdf(button):
            opts["pdf"] = button.get_active()
            self._queue_save()
            sync()

        def on_background(row, _pspec):
            opts["background"] = row.get_active()
            self._refresh()  # the layout preview mirrors this option

        name_row.connect("changed", on_name)
        for key, radio in format_radios.items():
            radio.connect("toggled", on_format, key)
        pdf_check.connect("toggled", on_pdf)
        background_row.connect("notify::active", on_background)
        choose_btn.connect("clicked", lambda *_: self._choose_folder(sync))
        export_btn.connect("clicked", lambda *_: (dialog.close(), self._export_results()))
        sync()

        form = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18,
                       margin_top=12, margin_bottom=18, margin_start=12, margin_end=12)
        for group in (destination, formats, paper):
            form.append(group)
        scroller = Gtk.ScrolledWindow(
            child=form, propagate_natural_height=True, hscrollbar_policy=Gtk.PolicyType.NEVER,
        )
        view = Adw.ToolbarView(content=scroller)
        view.add_top_bar(header)
        dialog.set_child(view)
        dialog.set_default_widget(export_btn)
        dialog.present(self)

    def _choose_folder(self, on_done=None):
        dialog = Gtk.FileDialog(title="Choose Output Folder")
        if self.output_dir:
            dialog.set_initial_folder(Gio.File.new_for_path(self.output_dir))
        dialog.select_folder(self, None, self._on_folder_chosen, on_done)

    def _on_folder_chosen(self, dialog, result, on_done=None):
        try:
            folder = dialog.select_folder_finish(result)
        except GLib.Error:
            return  # dismissed
        self.output_dir = folder.get_path()
        self.settings["output_dir"] = self.output_dir
        save_settings(self.settings)
        if on_done:
            on_done()

    def _open_output_folder(self, *_):
        if not self.output_dir:
            return
        try:
            Gio.AppInfo.launch_default_for_uri(Gio.File.new_for_path(self.output_dir).get_uri(), None)
        except GLib.Error as err:
            self.toast(f"Could not open folder: {err.message}")

    def _export_results(self):
        if not self.output_dir or not self.result_pages:
            return
        opts = self.export_opts
        fmt = opts["format"]

        base, destination = self._base_name(), Path(self.output_dir)
        include_background = opts["background"] or not CAN_STRIP_BACKGROUND
        targets = []  # (kind, source file, destination file); PDF has no single source
        if opts["pdf"]:  # one PDF replaces the individual pages
            targets.append(("pdf", None, destination / f"{base}.pdf"))
        else:
            for number, name in enumerate(self.result_pages, start=1):
                source = Path(name)
                if fmt == "svg":
                    targets.append(("svg", source, destination / f"{base}_page_{number}.svg"))
                png = Path(f"{source}.png")
                if fmt == "png" and png.is_file():
                    targets.append(("png", source, destination / f"{base}_page_{number}.png"))

        existing = [t for _kind, _src, t in targets if t.exists()]
        if not existing:
            self._write_targets(targets, include_background)
            return
        dialog = Adw.AlertDialog(
            heading="Replace existing files?",
            body=f"{len(existing)} file{'s' if len(existing) != 1 else ''} in {destination} "
                 f"will be overwritten. Change the file name in the Export dialog to keep both.",
        )
        dialog.add_response("cancel", "Cancel")
        dialog.add_response("replace", "Replace")
        dialog.set_response_appearance("replace", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.set_default_response("cancel")
        dialog.set_close_response("cancel")
        dialog.connect("response", lambda _d, response: response == "replace"
                       and self._write_targets(targets, include_background))
        dialog.present(self)

    def _write_targets(self, targets, include_background=True):
        try:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            for kind, source, target in targets:
                if kind == "pdf":
                    write_pdf(self.result_pages, target, include_background,
                              from_png=self.export_opts["format"] == "png")
                elif kind == "png":
                    write_png(source, target, include_background)
                elif include_background:
                    shutil.copy2(source, target)
                else:
                    text = Path(source).read_text(encoding="utf-8", errors="replace")
                    Path(target).write_text(strip_page_background(text), encoding="utf-8")
        except (OSError, RuntimeError, cairo.Error, GLib.Error) as err:
            self.alert("Could not export pages", str(err))
            return
        count = len(self.result_pages)
        self.toast(f"Exported {count} page{'s' if count != 1 else ''} to {self.output_dir}")

    # ====================================================================== #
    # Profiles (managed only from the sidebar's profile row)
    # ====================================================================== #
    def _sync_profile_row(self):
        names = list(self.profiles)
        index = names.index(self.active_profile) if self.active_profile in names else 0
        self._updating = True
        self.profile_row.set_model(Gtk.StringList.new(names))
        self.profile_row.set_selected(index)
        self._updating = False
        self.window_title.set_subtitle(self.active_profile)
        self._set_action_enabled("profile-delete", len(self.profiles) > 1)

    def _flush_settings(self):
        if self._save_id:
            GLib.source_remove(self._save_id)
            self._save_id = 0
        self._save_ui_settings()

    def _activate_profile(self, name):
        if name not in self.profiles:
            return
        self.active_profile = name
        self._apply_ui_settings(self.profiles[name])
        self.settings["profiles"] = self.profiles
        self.settings["active_profile"] = name
        save_settings(self.settings)
        self._sync_profile_row()
        self._refresh()

    def _on_profile_selected(self, row, _param):
        if self._updating or self._loading:
            return
        names = list(self.profiles)
        index = row.get_selected()
        if not 0 <= index < len(names):
            return
        name = names[index]
        if name == self.active_profile:
            return
        self._flush_settings()  # store edits made to the profile we are leaving
        self._activate_profile(name)

    def _ask_profile_name(self, heading, body, initial, accept, rename_of=None):
        dialog = Adw.AlertDialog(heading=heading, body=body)
        dialog.add_response("cancel", "Cancel")
        dialog.add_response("save", "Save")
        dialog.set_response_appearance("save", Adw.ResponseAppearance.SUGGESTED)
        dialog.set_default_response("save")
        dialog.set_close_response("cancel")
        entry = Adw.EntryRow(title="Profile name", activates_default=True)
        entry.set_text(initial)
        holder = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE)
        holder.add_css_class("boxed-list")
        holder.append(entry)
        dialog.set_extra_child(holder)

        def valid(name):
            name = name.strip()
            return bool(name) and (name == rename_of or name not in self.profiles)

        def on_changed(row):
            ok = valid(row.get_text())
            dialog.set_response_enabled("save", ok)
            if ok:
                row.remove_css_class("error")
            else:
                row.add_css_class("error")

        entry.connect("changed", on_changed)
        on_changed(entry)
        dialog.connect("response", lambda _d, response: response == "save" and accept(entry.get_text().strip()))
        dialog.present(self)
        entry.grab_focus()

    def _on_profile_new(self, *_):
        number = len(self.profiles) + 1
        while f"Profile {number}" in self.profiles:
            number += 1
        self._ask_profile_name(
            "New Profile", "The current settings are copied into the new profile.",
            f"Profile {number}", self._create_profile,
        )

    def _create_profile(self, name):
        self._flush_settings()
        self.active_profile = name
        self.profiles[name] = self._collect_ui_settings()
        self.settings["profiles"] = self.profiles
        self.settings["active_profile"] = name
        save_settings(self.settings)
        self._sync_profile_row()
        self.toast(f"Created “{name}”")

    def _on_profile_rename(self, *_):
        self._ask_profile_name(
            "Rename Profile", f"Rename “{self.active_profile}”.", self.active_profile,
            self._rename_profile, rename_of=self.active_profile,
        )

    def _rename_profile(self, new_name):
        if new_name == self.active_profile:
            return
        old_name = self.active_profile
        self.profiles = {
            new_name if key == old_name else key: value for key, value in self.profiles.items()
        }
        self.active_profile = new_name
        self.settings["profiles"] = self.profiles
        self.settings["active_profile"] = new_name
        save_settings(self.settings)
        self._sync_profile_row()
        self.toast(f"Renamed to “{new_name}”")

    def _on_profile_reset(self, *_):
        dialog = Adw.AlertDialog(
            heading="Reset Profile",
            body=f"Reset “{self.active_profile}” to the default handwriting, page and export settings?",
        )
        dialog.add_response("cancel", "Cancel")
        dialog.add_response("reset", "Reset")
        dialog.set_response_appearance("reset", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.set_close_response("cancel")
        dialog.connect("response", lambda _d, response: response == "reset" and self._reset_profile())
        dialog.present(self)

    def _reset_profile(self):
        self._apply_ui_settings({
            "spins": {key: DEFAULTS[key] for key in self.spins},
            "colors": {key: DEFAULTS[key] for key in self.colors},
            "style": DEFAULTS["style"] - 1,
            "ink_mode": DEFAULTS["ink_mode"],
            "paper": PAPER_NAMES.index(DEFAULTS["paper"]),
            "orientation": 0,
            "ppi": PPI_CHOICES.index(DEFAULTS["ppi"]),
            "formats": {
                "format": EXPORT_DEFAULTS["format"], "pdf": EXPORT_DEFAULTS["pdf"],
                "page_background": EXPORT_DEFAULTS["background"],
            },
            "base_name": EXPORT_DEFAULTS["name"],
            "name_auto": True,
        })
        self._on_style_changed()
        self._refresh()
        self.toast(f"Reset “{self.active_profile}”")

    def _on_profile_delete(self, *_):
        if len(self.profiles) <= 1:
            return
        dialog = Adw.AlertDialog(
            heading="Delete Profile",
            body=f"Delete “{self.active_profile}” and its settings? This cannot be undone.",
        )
        dialog.add_response("cancel", "Cancel")
        dialog.add_response("delete", "Delete")
        dialog.set_response_appearance("delete", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.set_close_response("cancel")
        dialog.connect("response", lambda _d, response: response == "delete" and self._delete_active_profile())
        dialog.present(self)

    def _delete_active_profile(self):
        deleted = self.active_profile
        names = list(self.profiles)
        self.profiles.pop(deleted, None)
        fallback = names[0] if names[0] != deleted else names[1]
        self.active_profile = fallback
        self.settings["profiles"] = self.profiles
        self.settings["active_profile"] = fallback
        save_settings(self.settings)
        self._apply_ui_settings(self.profiles[fallback])
        self._sync_profile_row()
        self._refresh()
        self.toast(f"Deleted “{deleted}”")

    # ---- remembered settings ------------------------------------------------ #
    def _collect_ui_settings(self):
        return {
            "spins": {key: row.get_value() for key, row in self.spins.items()},
            "colors": {key: rgba_to_hex(button.get_rgba()) for key, button in self.colors.items()},
            "style": self.style_row.get_selected(),
            "ink_mode": self.ink_mode_row.get_selected(),
            "paper": self.paper_row.get_selected(),
            "orientation": self.orientation_row.get_selected(),
            "ppi": self.ppi_row.get_selected(),
            "reference": {"path": self.reference_path, "fit": self.reference_fit_row.get_selected()},
            "formats": {
                "format": self.export_opts["format"], "pdf": self.export_opts["pdf"],
                "page_background": self.export_opts["background"],
            },
            "base_name": self.export_opts["name"],
            "name_auto": self.export_opts.get("name_auto", True),
        }

    def _apply_ui_settings(self, data):
        if not isinstance(data, dict):
            return
        self._loading = self._updating = True
        try:
            def number(value):
                return isinstance(value, (int, float)) and not isinstance(value, bool)

            def select(row, value):
                if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < row.get_model().get_n_items():
                    row.set_selected(value)

            # Lift the "written <= ruled" cap first so saved values are not clamped
            # by the previous profile's ruled-line count.
            self.spins["lines_per_page"].set_range(1, 200)
            for key, value in (data.get("spins") or {}).items():
                if key in self.spins and number(value):
                    self.spins[key].set_value(value)
            for key, value in (data.get("colors") or {}).items():
                rgba = Gdk.RGBA()
                if key in self.colors and isinstance(value, str) and rgba.parse(value):
                    self.colors[key].set_rgba(rgba)
            select(self.style_row, data.get("style"))
            select(self.ink_mode_row, data.get("ink_mode"))
            select(self.paper_row, data.get("paper"))
            select(self.orientation_row, data.get("orientation"))
            select(self.ppi_row, data.get("ppi"))

            ref = data.get("reference") if isinstance(data.get("reference"), dict) else {}
            fit = ref.get("fit")
            select(self.reference_fit_row, fit if isinstance(fit, int) else 0)
            path = ref.get("path")
            if not (isinstance(path, str) and Path(path).is_file() and self._set_reference(path)):
                self._set_reference(None)  # none saved, or the file has gone away

            formats = data.get("formats") if isinstance(data.get("formats"), dict) else {}
            opts = dict(EXPORT_DEFAULTS)
            if isinstance(formats.get("page_background"), bool):
                opts["background"] = formats["page_background"]
            fmt, pdf = formats.get("format"), formats.get("pdf") is True
            if fmt == "pdf":  # briefly stored as a third format, with its own source
                pdf, fmt = True, formats.get("pdf_source")
            if fmt not in ("svg", "png"):  # older settings: separate svg / png switches
                fmt = "png" if formats.get("png") is True and formats.get("svg") is not True else "svg"
            opts["format"], opts["pdf"] = fmt, pdf
            auto = data.get("name_auto")
            if not isinstance(auto, bool):  # older profiles: "handwriting" was the untouched default
                auto = not data.get("base_name") or data.get("base_name") == DEFAULT_BASE_NAME
            opts["name_auto"] = auto
            if isinstance(data.get("base_name"), str):
                opts["name"] = data["base_name"]
            if auto:
                opts["name"] = derive_base_name(self.get_text())
            self.export_opts = opts

            self._ppi = self.current_ppi()
            self.spins["ppi_custom"].set_visible(self.ppi_row.get_selected() == len(PPI_CHOICES))
            self.ink2_row.set_visible(self.ink_mode_row.get_selected() != 0)
            self._sync_line_limits()
        finally:
            self._loading = self._updating = False

    def _queue_save(self, *_):
        if self._loading or self._save_id:
            return
        self._save_id = GLib.timeout_add(600, self._save_ui_settings)

    def _save_ui_settings(self):
        self._save_id = 0
        self.profiles[self.active_profile] = self._collect_ui_settings()
        self.settings["profiles"] = self.profiles
        self.settings["active_profile"] = self.active_profile
        save_settings(self.settings)
        return False

    def _on_close(self, *_):
        if self._save_id:
            GLib.source_remove(self._save_id)
        self._save_ui_settings()
        self._kill_worker()
        return False

    # ====================================================================== #
    # Feedback: toasts (no actions) and alerts (errors, About)
    # ====================================================================== #
    def toast(self, message):
        toast = Adw.Toast(title=message)
        toast.set_use_markup(False)
        self.toasts.add_toast(toast)

    def alert(self, heading, body, details=None):
        dialog = Adw.AlertDialog(heading=heading, body=body)
        dialog.add_response("ok", "OK")
        if details:
            label = Gtk.Label(label=details, xalign=0, wrap=True, selectable=True)
            label.add_css_class("monospace")
            scroller = Gtk.ScrolledWindow(child=label, min_content_height=140, max_content_height=240)
            scroller.add_css_class("card")
            dialog.set_extra_child(scroller)
        dialog.present(self)

    def _on_about(self, *_):
        about = Adw.AboutDialog(
            application_name="TypeScribe", application_icon=APP_ID, version=VERSION,
            developer_name="Rudraneel Dutta",
            website="https://github.com/rudyoactiv/typescribe-handwriting",
            issue_url="https://github.com/rudyoactiv/typescribe-handwriting/issues",
            comments="Turn typed text into realistic handwriting. A recurrent neural network "
                     "writes every stroke, so no two pages are the same.",
            license_type=Gtk.License.BSD_3,
        )
        about.add_acknowledgement_section(
            "Built on", ["sjvasquez/handwriting-synthesis", "Alex Graves, Generating Sequences with RNNs"]
        )
        about.present(self)


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
class TypeScribeApp(Adw.Application):
    def __init__(self):
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.DEFAULT_FLAGS)

    def do_startup(self):
        Adw.Application.do_startup(self)
        provider = Gtk.CssProvider()
        provider.load_from_string(CSS)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        quit_action = Gio.SimpleAction.new("quit", None)
        quit_action.connect("activate", lambda *_: self.quit())
        self.add_action(quit_action)
        self.set_accels_for_action("app.quit", ["<Primary>q"])
        self.set_accels_for_action("win.generate", ["<Primary>Return"])
        self.set_accels_for_action("win.open-text", ["<Primary>o"])
        self.set_accels_for_action("win.export", ["<Primary>e"])
        self.set_accels_for_action("win.toggle-sidebar", ["F9"])

    def do_activate(self):
        window = self.props.active_window or TypeScribeWindow(self)
        window.present()


def main():
    GLib.set_application_name("TypeScribe")
    GLib.set_prgname(APP_ID)
    return TypeScribeApp().run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
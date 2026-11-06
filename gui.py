#!/usr/bin/env python3
"""TypeScribe - native GTK4 / libadwaita front end.

The GUI runs on the system Python (PyGObject). Handwriting generation runs in a
separate worker process (generate.py) using the Python environment that has
TensorFlow installed, so the interface never freezes while the model works.
"""

import importlib.util
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

import cairo  # noqa: E402  (pycairo ships with PyGObject)
from generate import layout_pages  # noqa: E402

APP_ID = "io.github.typescribe.TypeScribe"
APP_DIR = Path(__file__).resolve().parent
ASSETS = APP_DIR / "assets"
VERSION = "1.1"
MODEL_ENV_NAME = "test_hand"

DEFAULTS = {
    "line_length": 60,
    "lines_per_page": 24,
    "total_lines": 24,
    "style": 1,
    "bias": 0.95,
    "pen_width": 1.0,
    "ink_color": "#000000",
    "paper": "A4",
    "ppi": 72,
    "ppi_custom": 200,
    "line_height": 32,
    "view_height": 841.89,  # A4 at 72 ppi
    "view_width": 595.28,
    "margin_left": 72,
    "margin_top": 72,
    "page_color": "#ffffff",
    "margin_color": "#ff0000",
    "line_color": "#d3d3d3",
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
        pass  # remembering the folder is best-effort


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


def set_source(cr, value):
    color = parse_rgba(value)
    cr.set_source_rgba(color.red, color.green, color.blue, color.alpha)


def svg_page_size(svg_path, fallback):
    """Page size in points: the SVG viewBox is in 72-ppi units, i.e. points."""
    head = Path(svg_path).read_text(errors="replace")[:2000]
    match = re.search(r'viewBox="\s*[-\d.]+[\s,]+[-\d.]+[\s,]+([\d.]+)[\s,]+([\d.]+)\s*"', head)
    return (float(match.group(1)), float(match.group(2))) if match else fallback


def write_pdf(svg_pages, pdf_path):
    """Combine the rendered page images into one multi-page PDF, one page each."""
    pdf = None
    try:
        for svg in svg_pages:
            image = cairo.ImageSurface.create_from_png(f"{svg}.png")
            width, height = svg_page_size(svg, (image.get_width(), image.get_height()))
            if pdf is None:
                pdf = cairo.PDFSurface(str(pdf_path), width, height)
            pdf.set_size(width, height)
            cr = cairo.Context(pdf)
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
# Page layout preview
# --------------------------------------------------------------------------- #
class PagePreview(Gtk.DrawingArea):
    """Draws the ruled page exactly as the generated SVG will lay it out."""

    def __init__(self, get_values):
        super().__init__(hexpand=True, vexpand=True)
        self.get_values = get_values
        self.set_content_width(240)
        self.set_content_height(300)
        self.add_css_class("page-preview")
        self.set_draw_func(self._draw)

    def _draw(self, area, cr, width, height):
        v = self.get_values()
        view_w, view_h = v["view_width"], v["view_height"]
        pad = 28
        scale = min((width - 2 * pad) / view_w, (height - 2 * pad) / view_h)
        if scale <= 0:
            return
        pw, ph = view_w * scale, view_h * scale
        ox, oy = (width - pw) / 2, (height - ph) / 2
        hairline = max(1.0, scale)

        # Paper with a soft shadow.
        cr.set_source_rgba(0, 0, 0, 0.18)
        cr.rectangle(ox + 2, oy + 3, pw, ph)
        cr.fill()
        if v["background"]:
            set_source(cr, v["page_color"])
        else:
            cr.set_source_rgb(0.9, 0.9, 0.9)  # transparent export: neutral stand-in
        cr.rectangle(ox, oy, pw, ph)
        cr.fill()

        cr.save()
        cr.rectangle(ox, oy, pw, ph)
        cr.clip()

        lh, mt, ml = v["line_height"], v["margin_top"], v["margin_left"]
        if not v["background"]:
            cr.restore()
            cr.set_source_rgba(0.4, 0.4, 0.4, 0.8)
            cr.set_dash([5, 4])
            cr.set_line_width(1)
            cr.rectangle(ox + 0.5, oy + 0.5, pw - 1, ph - 1)
            cr.stroke()
            cr.set_dash([])
            self._draw_labels(cr, v, ox, oy, pw, ph)
            return

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
        self._draw_labels(cr, v, ox, oy, pw, ph)

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
        ext = cr.text_extents(label)
        cr.move_to(ox + (pw - ext.width) / 2, oy + ph + 18)
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
# key, title, subtitle, min, max, step, digits
SPIN_GROUPS = [
    ("Text Layout", [
        ("line_length", "Characters per line", "Wrap limit", 20, 75, 1, 0),
        ("lines_per_page", "Lines written per page", None, 1, 200, 1, 0),
        ("total_lines", "Ruled lines per page", "At least written lines", 1, 200, 1, 0),
    ]),
]
PAGE_SPINS = [
    ("view_width", "Page width", "px", 50, 30000, 1, 2),
    ("view_height", "Page height", "px", 50, 30000, 1, 2),
    ("line_height", "Line height", "px", 4, 3000, 1, 0),
    ("margin_left", "Left margin", "px", 0, 10000, 1, 0),
    ("margin_top", "Top margin", "px", 0, 10000, 1, 0),
]

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
        super().__init__(application=app, title="TypeScribe", default_width=1200, default_height=780)
        self.spins = {}
        self.colors = {}
        self._updating = False  # suppress change handlers during programmatic updates
        self._ppi = DEFAULTS["ppi"]
        self.proc = None
        self.run = None
        self.pulse_id = 0
        self.settings = load_settings()
        remembered_dir = self.settings.get("output_dir")
        self.output_dir = remembered_dir if isinstance(remembered_dir, str) and remembered_dir else None
        self.result_pages = []
        self.result_index = 0

        self._build_sidebar()
        self._build_content()
        self._build_shell()
        self._install_actions()
        self._on_style_changed()
        self._refresh()

    # ---- construction ----------------------------------------------------- #
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
        self.prefs = Adw.PreferencesPage()

        for title, specs in SPIN_GROUPS:
            group = Adw.PreferencesGroup(title=title)
            for spec in specs:
                group.add(self._spin_row(*spec))
            self.prefs.add(group)

        hand = Adw.PreferencesGroup(title="Handwriting")
        self.style_row = Adw.ComboRow(
            title="Writing style",
            model=Gtk.StringList.new([f"Style {i}" for i in range(1, 13)]),
        )
        self.style_row.connect("notify::selected", self._on_style_changed)
        hand.add(self.style_row)

        self.style_sample = Gtk.Picture(can_shrink=False, halign=Gtk.Align.CENTER, width_request=176, height_request=56)
        sample_box = Gtk.Box(halign=Gtk.Align.CENTER, margin_top=6, margin_bottom=6)
        sample_box.add_css_class("style-sample")
        sample_box.append(self.style_sample)
        hand.add(sample_box)

        hand.add(self._spin_row("bias", "Consistency", "Higher is neater, lower is looser", 0.0, 2.0, 0.05, 2))
        hand.add(self._spin_row("pen_width", "Pen thickness", None, 0.1, 10.0, 0.1, 1))
        hand.add(self._color_row("ink_color", "Ink color"))
        self.prefs.add(hand)

        page = Adw.PreferencesGroup(title="Page")
        self.paper_row = Adw.ComboRow(title="Paper size", model=Gtk.StringList.new(PAPER_NAMES))
        self.paper_row.set_selected(PAPER_NAMES.index(DEFAULTS["paper"]))
        self.paper_row.connect("notify::selected", self._on_paper_changed)
        page.add(self.paper_row)

        self.orientation_row = Adw.ComboRow(
            title="Orientation", model=Gtk.StringList.new(["Portrait", "Landscape"])
        )
        self.orientation_row.connect("notify::selected", self._on_orientation_changed)
        page.add(self.orientation_row)

        self.ppi_row = Adw.ComboRow(
            title="Resolution", subtitle="Export pixel density",
            model=Gtk.StringList.new(PPI_LABELS),
        )
        self.ppi_row.set_selected(PPI_CHOICES.index(DEFAULTS["ppi"]))
        self.ppi_row.connect("notify::selected", self._on_ppi_changed)
        page.add(self.ppi_row)

        ppi_custom = self._spin_row("ppi_custom", "Custom resolution", "ppi", 10, 1200, 1, 0)
        ppi_custom.set_visible(False)
        ppi_custom.connect("notify::value", self._on_ppi_changed)
        page.add(ppi_custom)

        for key, title, unit, lo, hi, step, digits in PAGE_SPINS:
            row = self._spin_row(key, title, unit, lo, hi, step, digits)
            if key in ("view_width", "view_height"):
                row.connect("notify::value", self._on_dims_edited)
            page.add(row)
        self.background_row = Adw.SwitchRow(
            title="Include page background",
            subtitle="Transparent page",
            active=DEFAULTS["background"],
        )
        self.background_row.connect("notify::active", self._refresh)
        page.add(self.background_row)
        page.add(self._color_row("page_color", "Page color"))
        page.add(self._color_row("margin_color", "Margin color"))
        page.add(self._color_row("line_color", "Line color"))
        self.prefs.add(page)

        header = Adw.HeaderBar(show_end_title_buttons=False)
        header.set_title_widget(Adw.WindowTitle(title="Settings", subtitle=""))
        self.sidebar_view = Adw.ToolbarView(content=self.prefs)
        self.sidebar_view.add_top_bar(header)

    def _build_editor(self):
        self.buffer = Gtk.TextBuffer()
        self.buffer.connect("changed", self._on_text_changed)
        self.text_view = Gtk.TextView(
            buffer=self.buffer, wrap_mode=Gtk.WrapMode.WORD_CHAR,
            top_margin=14, bottom_margin=14, left_margin=18, right_margin=18,
        )
        scroller = Gtk.ScrolledWindow(child=self.text_view, hexpand=True, vexpand=True)

        self.placeholder = Gtk.Label(
            label="Type or paste the text you want written by hand…",
            xalign=0, yalign=0, margin_top=14, margin_start=22, can_target=False,
        )
        self.placeholder.add_css_class("dim-label")
        overlay = Gtk.Overlay(child=scroller)
        overlay.add_overlay(self.placeholder)

        self.preview = PagePreview(self.values)
        preview_title = Gtk.Label(label="Page preview", xalign=0, margin_start=14, margin_top=10)
        preview_title.add_css_class("heading")
        preview_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        preview_box.append(preview_title)
        preview_box.append(self.preview)

        paned = Gtk.Paned(
            orientation=Gtk.Orientation.HORIZONTAL, start_child=overlay, end_child=preview_box,
            resize_start_child=True, resize_end_child=True,
            shrink_start_child=False, shrink_end_child=False, position=620,
        )
        return paned

    def _build_result(self):
        empty = Adw.StatusPage(
            icon_name="image-x-generic-symbolic", title="Nothing generated yet",
            description="Enter text and press Generate.",
        )

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
        open_btn = Gtk.Button(label="Open Folder", icon_name="folder-open-symbolic")
        open_btn.connect("clicked", self._on_open_folder)

        nav = Gtk.Box(spacing=8, margin_bottom=10, margin_start=12, margin_end=12, halign=Gtk.Align.CENTER)
        nav.append(self.prev_btn)
        nav.append(self.page_label)
        nav.append(self.next_btn)
        nav.append(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL, margin_start=6, margin_end=6))
        nav.append(open_btn)

        # Light backdrop so ink stays visible on transparent exports in dark mode.
        paper = Gtk.Box(vexpand=True, hexpand=True)
        paper.add_css_class("result-paper")
        paper.append(self.picture)

        pages = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        pages.append(paper)
        pages.append(nav)

        self.result_stack = Gtk.Stack()
        self.result_stack.add_named(empty, "empty")
        self.result_stack.add_named(pages, "pages")
        return self.result_stack

    def _build_content(self):
        self.view_stack = Adw.ViewStack()
        self.view_stack.add_titled_with_icon(self._build_editor(), "write", "Write", "document-edit-symbolic")
        self.view_stack.add_titled_with_icon(self._build_result(), "result", "Result", "image-x-generic-symbolic")

        switcher = Adw.ViewSwitcher(stack=self.view_stack, policy=Adw.ViewSwitcherPolicy.WIDE)

        self.sidebar_toggle = Gtk.ToggleButton(icon_name="sidebar-show-symbolic", tooltip_text="Toggle settings")
        open_btn = Gtk.Button(icon_name="document-open-symbolic", tooltip_text="Open text file (Ctrl+O)")
        open_btn.set_action_name("win.open-text")

        menu = Gio.Menu()
        menu.append("Open Text File…", "win.open-text")
        menu.append("Export As…", "win.export-as")
        menu.append("Reset to Defaults", "win.reset")
        menu.append("About TypeScribe", "win.about")
        menu_btn = Gtk.MenuButton(icon_name="open-menu-symbolic", menu_model=menu, primary=True)

        self.generate_btn = Gtk.Button(label="Generate")
        self.generate_btn.add_css_class("suggested-action")
        self.generate_btn.set_action_name("win.generate")

        # Export popover: where to save, which formats, then one clear Export button.
        export_popover = Gtk.Popover()
        export_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12, width_request=340,
                             margin_top=12, margin_bottom=12, margin_start=12, margin_end=12)

        def section(title):
            label = Gtk.Label(label=title, xalign=0)
            label.add_css_class("heading")
            return label

        def boxed_list(*rows):
            listbox = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE)
            listbox.add_css_class("boxed-list")
            for row in rows:
                listbox.append(row)
            return listbox

        self.folder_row = Adw.ActionRow(title="Folder", subtitle_lines=1)
        choose_btn = Gtk.Button(label="Choose…", valign=Gtk.Align.CENTER)
        choose_btn.connect("clicked", lambda *_: (export_popover.popdown(), self._export_as()))
        self.folder_row.add_suffix(choose_btn)

        def format_row(title, subtitle, active):
            check = Gtk.CheckButton(active=active, valign=Gtk.Align.CENTER)
            row = Adw.ActionRow(title=title, subtitle=subtitle, activatable_widget=check)
            row.add_prefix(check)
            return row, check

        svg_row, self.svg_check = format_row("SVG", "Vector, one file per page", True)
        png_row, self.png_check = format_row("PNG", "Image, one file per page", True)
        pdf_row, self.pdf_check = format_row("PDF", "A single file with all pages", False)

        export_now = Gtk.Button(label="Export")
        export_now.add_css_class("suggested-action")
        export_now.add_css_class("pill")
        export_now.connect("clicked", lambda *_: (export_popover.popdown(), self._on_export()))

        for widget in (
            section("Save to"), boxed_list(self.folder_row),
            section("Formats"), boxed_list(svg_row, png_row, pdf_row),
            export_now,
        ):
            export_box.append(widget)
        export_popover.set_child(export_box)

        self.export_btn = Adw.SplitButton(label="Export", popover=export_popover)
        self.export_btn.set_action_name("win.export")
        self._update_folder_label()

        header = Adw.HeaderBar()
        header.set_title_widget(switcher)
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

        self.content_view = Adw.ToolbarView(content=self.view_stack)
        self.content_view.add_top_bar(header)
        self.content_view.add_bottom_bar(bar)

    def _build_shell(self):
        self.split = Adw.OverlaySplitView(
            sidebar=self.sidebar_view, content=self.content_view,
            min_sidebar_width=320, max_sidebar_width=420, sidebar_width_fraction=0.3,
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
            ("generate", self._on_generate_action),
            ("export", self._on_export),
            ("export-as", self._export_as),
            ("reset", self._on_reset),
            ("open-text", self._on_open_text),
            ("about", self._on_about),
        ):
            action = Gio.SimpleAction.new(name, None)
            action.connect("activate", handler)
            self.add_action(action)
        self._update_save_state()

    def _update_folder_label(self):
        self.folder_row.set_subtitle(GLib.markup_escape_text(self.output_dir or "No folder chosen yet"))
        self.folder_row.set_tooltip_text(self.output_dir)

    def _update_save_state(self):
        # "generate" stays enabled while running: the same action cancels the job.
        can_export = bool(self.result_pages) and self.proc is None
        for name in ("export", "export-as"):
            action = self.lookup_action(name)
            if action:
                action.set_enabled(can_export)

    def _export_as(self, *_):
        dialog = Gtk.FileDialog(title="Choose Output Folder")
        if self.output_dir:
            dialog.set_initial_folder(Gio.File.new_for_path(self.output_dir))
        dialog.select_folder(self, None, self._on_folder_chosen)

    def _on_export(self, *_):
        if self.output_dir:
            self._export_results()
        else:
            self._export_as()

    # ---- state ------------------------------------------------------------- #
    def get_text(self):
        start, end = self.buffer.get_bounds()
        return self.buffer.get_text(start, end, False)

    def values(self):
        s = self.spins
        v = {
            "line_length": int(s["line_length"].get_value()),
            "lines_per_page": int(s["lines_per_page"].get_value()),
            "total_lines": int(s["total_lines"].get_value()),
            "bias": s["bias"].get_value(),
            "pen_width": s["pen_width"].get_value(),
            "line_height": int(s["line_height"].get_value()),
            "view_height": s["view_height"].get_value(),
            "view_width": s["view_width"].get_value(),
            "margin_left": int(s["margin_left"].get_value()),
            "margin_top": int(s["margin_top"].get_value()),
            "style": self.style_row.get_selected() + 1,
            "background": self.background_row.get_active(),
            "ppi": self.current_ppi(),
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
        scale = 72 / self.current_ppi()  # px -> the model's 72-based units
        width, height = s["view_width"].get_value(), s["view_height"].get_value()
        line_height, margin_left, margin_top = (
            s["line_height"].get_value(), s["margin_left"].get_value(), s["margin_top"].get_value(),
        )
        fit = max(1, int((height - margin_top) // line_height))
        s["total_lines"].set_value(fit)
        s["lines_per_page"].set_value(fit)
        # Handwriting is roughly 8.3 units wide per character.
        chars = int((width * scale - margin_left * scale - 24) / 8.3)
        s["line_length"].set_value(max(20, min(75, chars)))

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

    def _refresh(self, *_):
        if not hasattr(self, "preview"):
            return
        self.preview.queue_draw()
        ppi = self.current_ppi()
        mm = lambda key: self.spins[key].get_value() / ppi * 25.4  # noqa: E731
        self.paper_row.set_subtitle(f"{mm('view_width'):.0f} × {mm('view_height'):.0f} mm")
        for key in DIM_KEYS:  # every px value also shows its size in inches
            px = self.spins[key].get_value()
            self.spins[key].set_subtitle(f"px  ({px / ppi:.2f} in)")
        if self.proc:
            return

        v = self.values()
        parts = []
        text = self.get_text()
        if text.strip():
            pages = layout_pages(text, v["line_length"], v["lines_per_page"])
            lines = sum(len(p) for p in pages)
            parts.append(f"{len(pages)} page{'s' if len(pages) != 1 else ''} · {lines} lines")
        if v["total_lines"] < v["lines_per_page"]:
            parts.append("Ruled lines per page is fewer than lines written")
        if lines_overflow(v):
            parts.append("Ruled lines run past the bottom of the page")
        self.status.set_text(" · ".join(parts) if parts else "Ready")

    def _on_text_changed(self, *_):
        self.placeholder.set_visible(self.buffer.get_char_count() == 0)
        self._refresh()

    def _on_style_changed(self, *_):
        sample = ASSETS / f"font{self.style_row.get_selected() + 1}.png"
        self.style_sample.set_file(Gio.File.new_for_path(str(sample)))

    def _on_reset(self, *_):
        self._updating = True
        for key, row in self.spins.items():
            row.set_value(DEFAULTS[key])
        for key, button in self.colors.items():
            button.set_rgba(parse_rgba(DEFAULTS[key]))
        self.style_row.set_selected(DEFAULTS["style"] - 1)
        self.background_row.set_active(DEFAULTS["background"])
        self.paper_row.set_selected(PAPER_NAMES.index(DEFAULTS["paper"]))
        self.orientation_row.set_selected(0)
        self.ppi_row.set_selected(PPI_CHOICES.index(DEFAULTS["ppi"]))
        self.spins["ppi_custom"].set_visible(False)
        self._ppi = DEFAULTS["ppi"]
        self._updating = False
        self._refresh()

    # ---- dialogs ----------------------------------------------------------- #
    def toast(self, message):
        self.toasts.add_toast(Adw.Toast(title=message))

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
        self.view_stack.set_visible_child_name("write")

    # ---- generation -------------------------------------------------------- #
    def _on_generate_action(self, *_):
        if self.proc:
            self.run["cancelled"] = True
            self.proc.force_exit()
            return

        text = self.get_text()
        if not text.strip():
            self.toast("Enter some text first")
            return
        v = self.values()
        if v["total_lines"] < v["lines_per_page"]:
            self.alert("Check the page settings",
                       "Ruled lines per page must be at least as many as the lines written per page.")
            return
        if lines_overflow(v):
            self.alert("Too many lines",
                       "The ruled lines run past the bottom of the page. Increase the page height, "
                       "reduce the line height, or use fewer ruled lines.")
            return
        python = find_model_python()
        if not python:
            self.alert(
                "Model environment not found",
                f"TypeScribe needs a Python environment with TensorFlow. Create it with\n"
                f"conda env create -f environment.yml\n(environment name: {MODEL_ENV_NAME}), "
                f"or point the TYPESCRIBE_PYTHON variable at its python executable.",
            )
            return

        self._start(text, v, python)

    def _on_folder_chosen(self, dialog, result):
        try:
            folder = dialog.select_folder_finish(result)
        except GLib.Error:
            return  # dismissed
        self.output_dir = folder.get_path()
        self.settings["output_dir"] = self.output_dir
        save_settings(self.settings)
        self._update_folder_label()
        self._export_results()

    def _start(self, text, v, python):
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        for path in PREVIEW_DIR.iterdir():
            if path.is_file():
                path.unlink()

        to_units = 72 / v["ppi"]
        job = {
            "text": text,
            "output_dir": str(PREVIEW_DIR),
            "max_line_length": v["line_length"],
            "lines_per_page": v["lines_per_page"],
            "bias": v["bias"],
            "style": v["style"],
            "ink_color": v["ink_color"],
            "pen_width": v["pen_width"],
            # The model lays out in 72-ppi units; the export is scaled to the chosen ppi.
            "page": [
                v["line_height"] * to_units, v["total_lines"], v["view_height"] * to_units,
                v["view_width"] * to_units, -v["margin_left"] * to_units, -v["margin_top"] * to_units,
                v["page_color"], v["margin_color"], v["line_color"],
                v["background"], v["ppi"] / 72,
            ],
        }

        log = Path(GLib.get_user_cache_dir()) / "typescribe" / "last-run.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        launcher = Gio.SubprocessLauncher.new(
            Gio.SubprocessFlags.STDIN_PIPE | Gio.SubprocessFlags.STDOUT_PIPE
        )
        launcher.set_cwd(str(APP_DIR))
        launcher.set_stderr_file_path(str(log))
        launcher.setenv("PYTHONUNBUFFERED", "1", True)
        launcher.setenv("TF_CPP_MIN_LOG_LEVEL", "2", True)
        try:
            self.proc = launcher.spawnv([python, str(APP_DIR / "generate.py")])
        except GLib.Error as err:
            self.proc = None
            self.alert("Could not start the generator", err.message)
            return

        stdin = self.proc.get_stdin_pipe()
        stdin.write_all(json.dumps(job).encode("utf-8"), None)
        stdin.close(None)

        self.run = {"files": [], "error": None, "cancelled": False, "log": log}
        self._set_running(True)
        self.status.set_text("Starting…")
        self.reader = Gio.DataInputStream.new(self.proc.get_stdout_pipe())
        self.reader.read_line_async(GLib.PRIORITY_DEFAULT, None, self._on_line)

    def _set_running(self, running):
        self.sidebar_view.set_sensitive(not running)
        self.text_view.set_editable(not running)
        self.progress.set_visible(running)
        self._update_save_state()
        self.generate_btn.set_label("Cancel" if running else "Generate")
        if running:
            self.generate_btn.remove_css_class("suggested-action")
            self.generate_btn.add_css_class("destructive-action")
            self.pulse_id = GLib.timeout_add(120, self._pulse)
        else:
            self.generate_btn.remove_css_class("destructive-action")
            self.generate_btn.add_css_class("suggested-action")

    def _pulse(self):
        if not self.proc:
            self.pulse_id = 0
            return False
        self.progress.pulse()
        return True

    def _on_line(self, stream, result):
        try:
            line, _length = stream.read_line_finish_utf8(result)
        except GLib.Error:
            line = None
        if line is None:  # EOF: the worker has exited
            self.proc.wait_async(None, self._on_exited)
            return

        try:
            event = json.loads(line)
        except ValueError:
            event = {}
        kind = event.get("event")
        if kind == "loading":
            self.status.set_text("Loading the handwriting model…")
        elif kind == "page_start":
            self.status.set_text(f"Writing page {event['n']} of {event['total']}…")
        elif kind == "done":
            self.run["files"] = event.get("files", [])
        elif kind == "error":
            self.run["error"] = event.get("message", "Unknown error")
        stream.read_line_async(GLib.PRIORITY_DEFAULT, None, self._on_line)

    def _on_exited(self, proc, result):
        try:
            proc.wait_finish(result)
        except GLib.Error:
            pass
        run, status = self.run, proc.get_exit_status() if proc.get_if_exited() else -1
        self.proc, self.run = None, None
        self._set_running(False)
        self._refresh()

        if run["cancelled"]:
            self.toast("Generation cancelled")
        elif run["error"] or status != 0 or not run["files"]:
            tail = ""
            try:
                tail = "\n".join(run["log"].read_text(errors="replace").splitlines()[-15:])
            except OSError:
                pass
            self.alert("Generation failed", run["error"] or "The generator exited unexpectedly.", tail or None)
        else:
            self.result_pages = run["files"]
            self._show_page(0)
            self.view_stack.set_visible_child_name("result")
            count = len(self.result_pages)
            self._update_save_state()
            self.toast(f"Generated {count} page{'s' if count != 1 else ''}. Choose Export to export.")

    def _export_results(self):
        if not self.output_dir or not self.result_pages:
            return
        formats = [name for name, check in (
            ("svg", self.svg_check), ("png", self.png_check), ("pdf", self.pdf_check),
        ) if check.get_active()]
        if not formats:
            self.toast("Select SVG, PNG or PDF before exporting")
            return
        try:
            destination = Path(self.output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            for source_name in self.result_pages:
                source = Path(source_name)
                if "svg" in formats:
                    shutil.copy2(source, destination / source.name)
                if "png" in formats:
                    png = Path(f"{source}.png")
                    if png.is_file():
                        shutil.copy2(png, destination / png.name)
            if "pdf" in formats:
                write_pdf(self.result_pages, destination / "handwriting.pdf")
        except (OSError, cairo.Error) as err:
            self.alert("Could not export pages", str(err))
            return
        count = len(self.result_pages)
        self.toast(f"Exported {count} page{'s' if count != 1 else ''} to {self.output_dir}")

    # ---- results ----------------------------------------------------------- #
    def _show_page(self, index):
        if not self.result_pages:
            self.result_stack.set_visible_child_name("empty")
            return
        index = max(0, min(index, len(self.result_pages) - 1))
        self.result_index = index
        svg = self.result_pages[index]
        preview = svg + ".png" if os.path.exists(svg + ".png") else svg
        self.picture.set_file(Gio.File.new_for_path(preview))
        self.page_label.set_text(f"Page {index + 1} of {len(self.result_pages)}")
        self.prev_btn.set_sensitive(index > 0)
        self.next_btn.set_sensitive(index < len(self.result_pages) - 1)
        self.result_stack.set_visible_child_name("pages")

    def _on_open_folder(self, *_):
        if not self.output_dir:
            return
        try:
            Gio.AppInfo.launch_default_for_uri(Gio.File.new_for_path(self.output_dir).get_uri(), None)
        except GLib.Error as err:
            self.toast(f"Could not open folder: {err.message}")


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
        self.set_accels_for_action("win.export-as", ["<Primary><Shift>e"])

    def do_activate(self):
        window = self.props.active_window or TypeScribeWindow(self)
        window.present()


def main():
    GLib.set_application_name("TypeScribe")
    GLib.set_prgname(APP_ID)
    return TypeScribeApp().run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())

import numpy as np
import svgwrite
import cairosvg

from handwriting_synthesis import drawing


def _pressure_paths(strokes, color, width, pressure, chunk=5):
    """Pen strokes drawn as short pieces whose width drifts like pen pressure."""
    level = 0.0
    pieces, current, prev_eos = [], [], 1.0
    for x, y, eos in zip(*strokes.T):
        if prev_eos == 1.0 and current:
            pieces.append(current)
            current = []
        current.append((x, y))
        prev_eos = eos
    if current:
        pieces.append(current)

    for points in pieces:
        for start in range(0, len(points) - 1, chunk - 1):
            part = points[start:start + chunk]
            if len(part) < 2:
                continue
            level = float(np.clip(level + np.random.normal(0.0, 0.35), -1.0, 1.0))
            piece_width = max(width * 0.25, width * (1 + pressure * 0.7 * level))
            d = "M{},{} ".format(*part[0]) + " ".join("L{},{}".format(x, y) for x, y in part[1:])
            path = svgwrite.path.Path(d)
            yield path.stroke(color=color, width=round(piece_width, 3), linecap='round').fill("none")


def _draw(strokes, lines, filename, stroke_colors=None, stroke_widths=None, page=None, pressure=0.0):
    stroke_colors = stroke_colors or ['black'] * len(lines)
    stroke_widths = stroke_widths or [2] * len(lines)

    page = page or [32, 24, 896, 632, -64, -96, "#fff", "#f00", "#888"]
    line_height, total_lines_per_page, view_height, view_width, margin_left, margin_top, page_color, margin_color, line_color = page[:9]
    # Optional 10th entry: draw the page background (color, ruled lines, margins).
    include_background = page[9] if len(page) > 9 else True


    # Optional 11th entry: export scale (ppi / 72). The layout stays in 72-ppi
    # units (the viewBox); the SVG/PNG are written at the scaled pixel size.
    export_scale = page[10] if len(page) > 10 else None

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

        background.add(dwg.line(start=(-margin_left + line_height/2, 0), end=(-margin_left + line_height/2, view_height), stroke=margin_color, stroke_width=1))
        background.add(dwg.line(start=(-margin_left + line_height/2 - 5, 0), end=(-margin_left + line_height/2 - 5, view_height), stroke=margin_color, stroke_width=1))

        background.add(dwg.line(start=(0, -margin_top), end=(view_width, -margin_top), stroke=margin_color, stroke_width=1))
        background.add(dwg.line(start=(0, -margin_top - 5), end=(view_width, -margin_top - 5), stroke=margin_color, stroke_width=1))

        dwg.add(background)


    initial_coord = np.array([margin_left, margin_top - line_height/2])


    for i, (offsets, line, color, width) in enumerate(zip(strokes, lines, stroke_colors, stroke_widths)):
        # Stop drawing text if lines exceed the fixed page limit
        if i >= total_lines_per_page:
            break

        if not line:
            initial_coord[1] -= line_height
            continue

        # Convert offsets to coordinates and adjust them
        strokes = drawing.offsets_to_coords(offsets)
        strokes = drawing.denoise(strokes)
        strokes[:, :2] = drawing.align(strokes[:, :2])
        strokes[:, 1] *= -1
        strokes[:, :2] -= strokes[:, :2].min() + initial_coord

        if pressure > 0:
            for path in _pressure_paths(strokes, color, width, pressure):
                dwg.add(path)
        else:
            # Create the path for handwriting strokes
            prev_eos = 1.0
            p = "M{},{} ".format(0, 0)
            for x, y, eos in zip(*strokes.T):
                p += '{}{},{} '.format('M' if prev_eos == 1.0 else 'L', x, y)
                prev_eos = eos
            path = svgwrite.path.Path(p)
            path = path.stroke(color=color, width=width, linecap='round').fill("none")
            dwg.add(path)

        initial_coord[1] -= line_height

    # Save the SVG file and convert to PNG
    dwg.save()
    cairosvg.svg2png(url=filename, write_to=filename + '.png')

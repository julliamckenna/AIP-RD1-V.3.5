"""Legend marker glyphs as small vector outlines - shape, direction, fill and colors.

For every legend entry the extractor has already isolated the symbol's pixels.  This module
turns that symbol into a *glyph*:

    {"shape": "triangle_left", "fill": "open", "color_hex": "#1f4e9c", "fill_hex": null,
     "edge_hex": "#1f4e9c", "polygon": [[-0.8, 0.0], [0.9, -1.0], [0.9, 1.0]], "size_px": 5.2,
     "vertices": 3, "source": "raster", "confidence": 0.9}

``polygon`` is the outline in units of the marker's equivalent radius, centred on the marker's
centroid, x right / y down - a resolution-free vector the overlay scales onto every detected
point, so the drawn outline is the real legend glyph (star points, triangle direction, rotated
squares ...), not a generic shape picked from a name.

When the figure came from vector PDF drawings, the exact path of the legend symbol is used
instead (``source: "pdf_vector"``): exact vertices and exact fill / stroke colors.

The declared marker (agent 01 / agent 02) is compared with the measured glyph; mismatches are
reported to agent 02 so it can correct the declaration in the next check round.
"""

from __future__ import annotations

import json
import math
import pathlib

import cv2
import numpy as np

SHAPES = (
    "circle", "square", "diamond", "triangle_up", "triangle_down", "triangle_left", "triangle_right",
    "pentagon", "hexagon", "star", "plus", "cross", "none",
)
UPSCALE = 6          # contour precision on small raster glyphs
MIN_GLYPH_PX = 4     # below this a raster glyph is too small to classify reliably


def _hex(bgr) -> str:
    b, g, r = (int(round(float(v))) for v in bgr)
    return f"#{max(0, min(255, r)):02x}{max(0, min(255, g)):02x}{max(0, min(255, b)):02x}"


def _lab_to_bgr(lab):
    return cv2.cvtColor(np.uint8([[[round(float(v)) for v in lab]]]), cv2.COLOR_LAB2BGR)[0, 0]


def _hex_distance(a: str, b: str) -> float:
    """Lab distance between two #rrggbb colors (inf when either is missing)."""
    try:
        bgr = [[[int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16)] for h in (a, b)]]
    except (TypeError, ValueError, IndexError):
        return math.inf
    lab = cv2.cvtColor(np.uint8(bgr), cv2.COLOR_BGR2LAB).astype(float)[0]
    return float(np.linalg.norm(lab[0] - lab[1]))


def _normalise(points: np.ndarray, centre, radius) -> list:
    return [[round(float((x - centre[0]) / radius), 3), round(float((y - centre[1]) / radius), 3)] for x, y in points]


# Canonical outlines = matplotlib's own marker paths (most chart software uses the same geometry).
_MPL_MARKERS = {
    "circle": "o", "square": "s", "diamond": "D", "triangle_up": "^", "triangle_down": "v",
    "triangle_left": "<", "triangle_right": ">", "pentagon": "p", "hexagon": "h", "star": "*",
    "plus": "P", "cross": "X",
}
_CANVAS, _RADIUS = 64, 16.0   # silhouettes are compared on a 64x64 canvas, equal area radius 16 px


def _canonical_polygons() -> dict:
    from matplotlib.markers import MarkerStyle

    polygons = {}
    for name, code in _MPL_MARKERS.items():
        style = MarkerStyle(code)
        path = style.get_path().transformed(style.get_transform())
        polygon = max(path.to_polygons(), key=len)
        polygon[:, 1] *= -1  # matplotlib y points up, images y points down
        polygons[name] = polygon
    return polygons


_CANONICAL = None


def _silhouette_canvas(points: np.ndarray) -> np.ndarray:
    """Fill an outline on the comparison canvas with its centroid centred and its area normalised."""
    area = abs(cv2.contourArea(points.astype(np.float32)))
    moments = cv2.moments(points.astype(np.float32))
    if area <= 0 or moments["m00"] == 0:
        return np.zeros((_CANVAS, _CANVAS), np.uint8)
    centre = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
    scale = _RADIUS / math.sqrt(area / math.pi)
    placed = (points - centre) * scale + _CANVAS / 2
    canvas = np.zeros((_CANVAS, _CANVAS), np.uint8)
    cv2.fillPoly(canvas, [np.round(placed * 16).astype(np.int32)], 1, cv2.LINE_8, shift=4)
    return canvas


def classify_outline(contour: np.ndarray) -> tuple[str, float, int]:
    """Best-fitting canonical marker for one closed outline: (shape, confidence, vertex count).

    Confidence is the overlap (IoU) with the winner, reduced when the runner-up fits almost as well.
    """
    global _CANONICAL
    if _CANONICAL is None:
        _CANONICAL = {name: _silhouette_canvas(poly) for name, poly in _canonical_polygons().items()}
    points = contour.reshape(-1, 2).astype(np.float64)
    if len(points) < 3 or cv2.contourArea(points.astype(np.float32)) <= 0:
        return "none", 0.0, 0
    measured = _silhouette_canvas(points)
    scores = []
    for name, template in _CANONICAL.items():
        union = np.logical_or(measured, template).sum()
        scores.append((float(np.logical_and(measured, template).sum()) / max(union, 1), name))
    scores.sort(reverse=True)
    (best, shape), (second, _) = scores[0], scores[1]
    confidence = best * min(1.0, 0.5 + 5 * (best - second))
    hull = cv2.convexHull(points.astype(np.float32))
    vertices = len(cv2.approxPolyDP(hull, 0.04 * cv2.arcLength(hull, True), True))
    return shape, round(confidence, 3), vertices


def _locate(crop_bgr, box, mask, color_lab):
    """The crop pixels under ``mask`` inside the symbol box ``box`` = (x0, y0, x1, y1), aligned by ink."""
    x0, y0, x1, y1 = (int(v) for v in box)
    area = crop_bgr[max(0, y0):y1, max(0, x0):x1]
    h, w = mask.shape
    if area.shape[0] < h or area.shape[1] < w:
        return None
    if color_lab is not None:
        lab = cv2.cvtColor(area, cv2.COLOR_BGR2LAB).astype(np.float32)
        ink = (np.linalg.norm(lab - np.float32(color_lab), axis=2) < 40).astype(np.float32)
    else:
        ink = (cv2.cvtColor(area, cv2.COLOR_BGR2GRAY) < 128).astype(np.float32)
    score = cv2.matchTemplate(ink, mask.astype(np.float32), cv2.TM_CCORR)
    _, _, _, (dx, dy) = cv2.minMaxLoc(score)
    return area[dy:dy + h, dx:dx + w]


def _trim_line_stub(glyph: np.ndarray) -> np.ndarray:
    """Remove the legend line running through or beside a marker glyph.

    The line's thickness is measured at the blob's outer columns (where only the line is);
    columns no taller than that (+1 px) are line.  Line runs at either end are cut when they
    are long compared with the glyph; a short taper such as a triangle tip is kept.
    """
    filled = glyph.copy()
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(filled, contours, -1, 1, thickness=-1)
    heights = filled.sum(axis=0).astype(float)
    inked = np.nonzero(heights)[0]
    if len(inked) < 6 or heights.max() < 4:
        return glyph
    span = heights[inked.min():inked.max() + 1]  # empty padding columns are not the line
    edge = max(2, len(span) // 10)
    line = float(np.median(np.concatenate([span[:edge], span[-edge:]])))
    line = line if line < 0.5 * heights.max() else 0.0  # ends as tall as the glyph: no line
    thin = heights <= max(line + 1.0, min(3.0, 0.25 * heights.max()))  # empty columns count as thin
    keep = np.ones_like(thin)
    for side in (range(len(thin)), range(len(thin) - 1, -1, -1)):
        run = []
        for column in side:
            if not thin[column]:
                break
            run.append(column)
        body_width = int((~thin).sum())
        if len(run) > max(3, 0.35 * body_width, 0.75 * heights.max()):
            keep[run] = False
    return glyph * keep[np.newaxis, :].astype(glyph.dtype)


def ink_mask(area_bgr, color_lab, min_share=0.35, max_off_axis=30.0, background_lab=None) -> np.ndarray:
    """Pixels that contain at least ``min_share`` of the series color mixed with the background.

    Anti-aliased outlines are blends of marker color and background; measuring the share of the
    color along the background->color line keeps a 1 px ring connected where a plain color
    distance test breaks it into fragments.  ``max_off_axis`` rejects other series' colors.
    """
    lab = cv2.cvtColor(area_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    if background_lab is None:
        border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        background = np.median(border, axis=0)
    else:
        background = np.float32(background_lab)
    direction = np.float32(color_lab) - background
    length2 = float(direction @ direction)
    if length2 < 100:  # series color ~ background: fall back to a plain distance test
        return (np.linalg.norm(lab - np.float32(color_lab), axis=2) < 25).astype(np.uint8)
    offset = lab - background
    share = offset @ direction / length2
    off_axis = np.linalg.norm(offset - share[..., None] * direction, axis=2)
    return ((share >= min_share) & (off_axis <= max_off_axis)).astype(np.uint8)


def cut_glyph(crop_bgr, box, color_lab):
    """Cut the complete marker glyph out of its legend symbol box.

    Ink = pixels carrying the series color (``ink_mask``); the glyph is the ink blob
    nearest the box centre (open outlines stay whole because nothing is eroded), and a legend
    line stub running through it is trimmed off.
    """
    if color_lab is None:
        return None
    x0, y0, x1, y1 = (int(v) for v in box)
    area = crop_bgr[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    if area.size == 0:
        return None
    ink = ink_mask(area, color_lab)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    candidates = [i for i in range(1, count) if stats[i, 4] >= 9]
    if not candidates:
        return None
    centre = np.array([area.shape[1] / 2, area.shape[0] / 2])
    best = min(candidates, key=lambda i: np.linalg.norm(centroids[i] - centre) - 0.3 * np.sqrt(stats[i, 4]))
    glyph = _trim_line_stub((labels == best).astype(np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(glyph, connectivity=8)
    if count < 2:
        return None
    glyph = (labels == 1 + int(np.argmax(stats[1:, 4]))).astype(np.uint8)
    ys, xs = np.nonzero(glyph)
    return glyph[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def describe_raster(crop_bgr: np.ndarray, glyph_mask: np.ndarray | None, box, color_lab) -> dict:
    # the extractor's own symbol mask is trimmed for template matching (it drops short columns,
    # i.e. triangle tips); the glyph is re-cut from the symbol box, the given mask is only a fallback
    fallback_mask = glyph_mask
    """Glyph from a raster legend symbol: ``glyph_mask`` = its pixel mask, ``box`` = the symbol box in the panel."""
    series_bgr = _lab_to_bgr(color_lab) if color_lab is not None else None
    glyph_mask = cut_glyph(crop_bgr, box, color_lab) if box is not None else None
    if glyph_mask is None:
        glyph_mask = fallback_mask
    if glyph_mask is None or min(glyph_mask.shape) < MIN_GLYPH_PX:
        return {"shape": "none", "fill": "filled", "color_hex": _hex(series_bgr) if series_bgr is not None else "",
                "fill_hex": None, "edge_hex": None, "polygon": [], "size_px": 0.0, "vertices": 0,
                "source": "raster", "confidence": 0.0}
    mask = (glyph_mask > 0).astype(np.uint8)
    big = cv2.resize(mask.astype(np.float32), None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_LINEAR) > 0.5
    big = big.astype(np.uint8)
    contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    silhouette = np.zeros_like(big)
    cv2.drawContours(silhouette, [contour], -1, 1, thickness=-1)
    # open vs filled: estimate the stroke width (ink area / outline length) and look only at the
    # part of the silhouette deeper than that stroke - hollow for open markers, inked for filled ones
    depth = cv2.distanceTransform(silhouette, cv2.DIST_L2, 5)
    stroke = float(big.sum()) / max(cv2.arcLength(contour, True), 1.0)
    interior = depth > stroke + 0.5 * UPSCALE
    fill_ratio = (float(big[interior].mean()) if interior.sum() > UPSCALE**2
                  else float(big.sum()) / max(float(silhouette.sum()), 1.0))
    shape, confidence, vertices = classify_outline(contour)
    area = cv2.contourArea(contour)
    moments = cv2.moments(contour)
    centre = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    radius = math.sqrt(area / math.pi)
    outline = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True).reshape(-1, 2)
    if shape == "circle":
        outline = np.array([[centre[0] + radius * math.cos(t), centre[1] + radius * math.sin(t)]
                            for t in np.linspace(0, 2 * math.pi, 16, endpoint=False)])

    # colors: find where the glyph mask sits inside the symbol box, then the stroke is the outer
    # band of the glyph and the face is what the band encloses
    edge_hex = fill_hex = None
    fill = "filled" if fill_ratio > 0.6 else "open"
    region = _locate(crop_bgr, box, mask, color_lab)
    if region is not None:
        small_sil = cv2.resize(silhouette, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        inner = cv2.erode(small_sil, np.ones((3, 3), np.uint8))
        band = (small_sil > 0) & (inner == 0) & (mask > 0)
        face = inner > 0
        if band.any():
            edge_hex = _hex(np.median(region[band], axis=0))
        if fill == "filled" and face.any():
            fill_hex = _hex(np.median(region[face], axis=0))
    series_hex = _hex(series_bgr) if series_bgr is not None else (fill_hex or edge_hex or "")
    if fill == "filled" and not fill_hex:
        fill_hex = series_hex
    if not edge_hex:
        edge_hex = series_hex
    return {
        "shape": shape, "fill": fill, "color_hex": series_hex, "fill_hex": fill_hex, "edge_hex": edge_hex,
        "polygon": _normalise(outline, centre, radius), "size_px": round(radius / UPSCALE, 2),
        "vertices": int(vertices), "source": "raster", "confidence": round(float(confidence), 2),
    }


#  Vector PDF figures: exact legend paths


def _path_points(items, scale, origin) -> list[tuple[float, float]]:
    pts = []
    for item in items:
        kind = item[0]
        if kind == "l":
            pts += [item[1], item[2]]
        elif kind == "c":  # cubic Bezier: sample it
            p0, p1, p2, p3 = item[1:5]
            for t in np.linspace(0, 1, 6):
                u = 1 - t
                pts.append(type(p0)(u**3 * p0.x + 3 * u * u * t * p1.x + 3 * u * t * t * p2.x + t**3 * p3.x,
                                    u**3 * p0.y + 3 * u * u * t * p1.y + 3 * u * t * t * p2.y + t**3 * p3.y))
        elif kind == "re":
            r = item[1]
            pts += [r.tl, r.tr, r.br, r.bl]
        elif kind == "qu":
            q = item[1]
            pts += [q.ul, q.ur, q.lr, q.ll]
    return [((p.x - origin[0]) * scale, (p.y - origin[1]) * scale) for p in pts]


def pdf_vector_marks(page, clip, dpi, max_size_pt=20.0) -> list[dict]:
    """Small closed/filled drawings (marker glyphs) of a vector figure, in the rendered image's pixels."""
    scale = dpi / 72.0
    marks = []
    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if rect is None or not rect.intersects(clip) or max(rect.width, rect.height) > max_size_pt:
            continue
        if max(rect.width, rect.height) < 1.0:
            continue
        points = _path_points(drawing.get("items") or [], scale, (clip.x0, clip.y0))
        if len(points) < 3 and not any(item[0] == "l" for item in drawing.get("items") or []):
            continue
        color = lambda c: None if c is None else "#" + "".join(f"{round(v * 255):02x}" for v in c[:3])
        marks.append({
            "bbox_px": [round((rect.x0 - clip.x0) * scale, 2), round((rect.y0 - clip.y0) * scale, 2),
                        round((rect.x1 - clip.x0) * scale, 2), round((rect.y1 - clip.y0) * scale, 2)],
            "points_px": [[round(x, 2), round(y, 2)] for x, y in points],
            "fill_hex": color(drawing.get("fill")),
            "stroke_hex": color(drawing.get("color")),
            "closed": bool(drawing.get("closePath")) or drawing.get("fill") is not None,
        })
    return marks


def describe_vector(mark: dict) -> dict:
    """Glyph from an exact PDF path (panel pixel coordinates)."""
    points = np.array(mark["points_px"], dtype=np.float32)
    if len(points) >= 3 and mark.get("closed"):
        contour = (points * UPSCALE).astype(np.int32).reshape(-1, 1, 2)
        shape, confidence, vertices = classify_outline(contour)
        confidence = max(confidence, 0.95)
    else:  # open strokes only: plus / cross made of line segments
        diagonal = np.mean(np.abs(np.diff(points[:, 0])) * np.abs(np.diff(points[:, 1])) > 0)
        shape, confidence, vertices = ("cross" if diagonal > 0.5 else "plus"), 0.95, len(points)
    centre = points.mean(axis=0)
    x0, y0, x1, y1 = mark["bbox_px"]
    radius = max(1e-3, math.sqrt(max((x1 - x0) * (y1 - y0), 1e-3) / math.pi))
    filled = mark.get("fill_hex") is not None
    series_hex = mark.get("fill_hex") if filled and mark.get("fill_hex") not in ("#ffffff",) else (mark.get("stroke_hex") or mark.get("fill_hex"))
    return {
        "shape": shape, "fill": "filled" if filled and mark.get("fill_hex") != "#ffffff" else "open",
        "color_hex": series_hex or "", "fill_hex": mark.get("fill_hex"), "edge_hex": mark.get("stroke_hex") or mark.get("fill_hex"),
        "polygon": _normalise(points, centre, radius), "size_px": round(radius, 2), "vertices": int(vertices),
        "source": "pdf_vector", "confidence": round(float(confidence), 2),
    }


def load_panel_marks(path) -> list[dict]:
    try:
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8")) if path else []
    except (OSError, ValueError):
        return []


def describe_legend(crop_bgr, shapes, boxes, colors, vector_marks=()) -> list[dict]:
    """One glyph per legend entry, preferring an exact PDF path inside the symbol box."""
    glyphs = []
    for index, box in enumerate(boxes):
        mark = None
        if vector_marks:
            bx0, by0, bx1, by1 = box
            inside = [m for m in vector_marks
                      if bx0 <= (m["bbox_px"][0] + m["bbox_px"][2]) / 2 <= bx1
                      and by0 <= (m["bbox_px"][1] + m["bbox_px"][3]) / 2 <= by1]
            # the glyph is the most compact mark; the legend line stub is long and thin
            inside.sort(key=lambda m: abs((m["bbox_px"][2] - m["bbox_px"][0]) - (m["bbox_px"][3] - m["bbox_px"][1])))
            mark = inside[0] if inside else None
        color = colors[index] if index < len(colors) else None
        glyph = describe_vector(mark) if mark else describe_raster(
            crop_bgr, shapes[index] if index < len(shapes) else None, box, color,
        )
        glyphs.append(glyph)
    return glyphs


def compare(declared: dict, glyph: dict) -> list[str]:
    """Differences between a declared series style (agent 01/02) and the measured glyph."""
    if not glyph or glyph.get("confidence", 0) < 0.6 or glyph.get("shape") == "none":
        return []
    issues = []
    if declared.get("marker") and declared["marker"] != glyph["shape"]:
        issues.append(f"marker declared {declared['marker']!r}, legend glyph measures {glyph['shape']!r}")
    if declared.get("marker_fill") and declared["marker_fill"] != glyph["fill"]:
        issues.append(f"fill declared {declared['marker_fill']!r}, legend glyph measures {glyph['fill']!r}")
    if _hex_distance(declared.get("color_hex", ""), glyph.get("color_hex", "")) > 25:
        issues.append(f"color declared {declared.get('color_hex')}, legend glyph measures {glyph.get('color_hex')}")
    return issues


def draw_glyph(image, center, radius, glyph, color, thickness=1, halo=None) -> bool:
    """Draw the measured outline scaled to ``radius`` at ``center``. False when there is no outline."""
    polygon = (glyph or {}).get("polygon") or []
    if len(polygon) < 2:
        return False
    pts = np.array([[center[0] + x * radius, center[1] + y * radius] for x, y in polygon], np.float32)
    shape = glyph.get("shape")
    if shape in ("plus", "cross") and glyph.get("source") == "pdf_vector" and not glyph.get("fill_hex"):
        segments = [pts[i:i + 2] for i in range(0, len(pts) - 1, 2)]
    else:
        segments = [pts]
    for seg in segments:
        seg = np.round(seg * 4).astype(np.int32)  # sub-pixel precision via shift=2
        closed = len(seg) > 2
        if halo is not None:
            cv2.polylines(image, [seg], closed, halo, thickness + 2, cv2.LINE_AA, shift=2)
        cv2.polylines(image, [seg], closed, color, thickness, cv2.LINE_AA, shift=2)
    return True


def legend_sheet(crop_bgr, boxes, glyphs, labels, out_png, zoom=8):
    """One row per legend entry: the source symbol enlarged | the measured vector glyph | its label."""
    rows = []
    for box, glyph, label in zip(boxes, glyphs, labels):
        x0, y0, x1, y1 = (int(v) for v in box)
        symbol = crop_bgr[max(0, y0):y1, max(0, x0):x1]
        if symbol.size == 0:
            continue
        cell = 16 * zoom // 2
        factor = cell / max(symbol.shape[:2])
        scaled = cv2.resize(symbol, (max(1, round(symbol.shape[1] * factor)), max(1, round(symbol.shape[0] * factor))),
                            interpolation=cv2.INTER_NEAREST)
        big = np.full((cell, cell, 3), 255, np.uint8)
        big[:scaled.shape[0], :scaled.shape[1]] = scaled
        drawn = np.full_like(big, 255)
        color_hex = glyph.get("edge_hex") or glyph.get("color_hex") or "#000000"
        bgr = tuple(int(color_hex[i:i + 2], 16) for i in (5, 3, 1))
        if glyph.get("fill") == "filled" and glyph.get("polygon"):
            pts = np.array([[cell / 2 + x * cell / 4, cell / 2 + y * cell / 4] for x, y in glyph["polygon"]], np.int32)
            fill_hex = glyph.get("fill_hex") or color_hex
            cv2.fillPoly(drawn, [pts], tuple(int(fill_hex[i:i + 2], 16) for i in (5, 3, 1)))
        draw_glyph(drawn, (cell / 2, cell / 2), cell / 4, glyph, bgr, 2)
        text = np.full((cell, 520, 3), 255, np.uint8)
        cv2.putText(text, f"{label}"[:34], (8, cell // 2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(text, f"{glyph.get('shape')} / {glyph.get('fill')} / {glyph.get('color_hex')} ({glyph.get('source')})",
                    (8, cell // 2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
        rows.append(np.hstack([big, np.full((cell, 6, 3), 200, np.uint8), drawn, text]))
    if rows:
        cv2.imwrite(str(out_png), np.vstack(rows))
    return bool(rows)

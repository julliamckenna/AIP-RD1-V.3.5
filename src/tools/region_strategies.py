"""Agent 02 coaching applied inside the Python extraction.

Agent 02 can tell Python more than a structure: where one clean marker of a series is
(``sample_px``), how a region must be handled (``region_strategies``) and at which pressures the
series share marker columns (``shared_x_columns``).  This module turns that guidance into
deterministic pixel work; every action is returned for the audit, and nothing here loosens the
calibration gates.

  sample_color            measure the series color at the agent's sample marker (plot color,
                           not the legend rendering) - rescues series the legend reading got wrong
  ignore                   no points of the listed series (all series when none listed) in the box
  split_filled_open        same-color filled / open twins in the box: each marker goes to the
                           filled or the open series by the ink share inside the legend glyph's
                           interior (glyph size, not the per-series detector radius, which drifts
                           between open rings and filled discs).  A filled centre wins: where an
                           open ring sits exactly on a filled marker the ring can not be seen
  sample_band_at_columns   fused band (markers drawn so densely they form a solid line): one
                           point per shared x column where the series' ink crosses it, marked as a
                           line sample (``is_inferred``, review-only), never a visible marker
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.models import naming


def _inside(box, x, y):
    x0, y0, x1, y1 = box
    return x0 <= x <= x1 and y0 <= y <= y1


def _norm(label):
    return "".join(naming.series_label(label).lower().split())


def sample_color(lab, px, background_lab, radius=6):
    """Lab color of the marker at ``px``: the plot color, not the legend rendering.

    Colored markers often carry a dark outline, so the most saturated pixels (the face) decide;
    only a grey/black marker falls back to the pixels farthest from the background.
    """
    x, y = int(round(px[0])), int(round(px[1]))
    window = lab[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1].reshape(-1, 3)
    if not len(window):
        return None
    distance = np.linalg.norm(window - np.float32(background_lab), axis=1)
    if float(distance.max()) < 20:
        return None  # nothing but background there
    chroma = np.hypot(window[:, 1] - 128.0, window[:, 2] - 128.0)  # OpenCV 8-bit Lab: a, b centred on 128
    if float(np.quantile(chroma, 0.9)) >= 15:
        strong = window[chroma >= np.quantile(chroma, 0.8)]
    else:
        strong = window[distance >= np.quantile(distance, 0.7)]
    return np.median(strong, axis=0).astype(np.float32)


def apply_sample_colors(spec, colors, lab, plot_mask):
    """Replace legend colors by colors sampled at agent 02's marker positions. Returns the audit."""
    if not any(series.get("sample_px") for series in spec.get("series", [])):
        return []
    plot = lab[plot_mask > 0]
    background = np.median(plot[:: max(1, len(plot) // 20000)], axis=0) if len(plot) else np.float32([100, 0, 0])
    audit = []
    for index, series in enumerate(spec.get("series", [])):
        if not series.get("sample_px") or index >= len(colors):
            continue
        measured = sample_color(lab, series["sample_px"], background)
        entry = {"label": series.get("label"), "sample_px": series["sample_px"],
                 "legend_lab": [round(float(v), 1) for v in colors[index]]}
        if measured is None:
            entry["result"] = "no marker ink at the sample position; legend color kept"
        else:
            entry.update(sample_lab=[round(float(v), 1) for v in measured],
                         distance=round(float(np.linalg.norm(measured - colors[index])), 1), result="sample color used")
            colors[index] = measured
        audit.append(entry)
    return audit


def mask_ignored(spec, plot_mask):
    """Zero the plot mask inside every ``ignore`` region that applies to all series."""
    for region in spec.get("region_strategies") or []:
        if region.get("strategy") == "ignore" and not region.get("series_labels"):
            x0, y0, x1, y1 = region["bbox_px"]
            plot_mask[y0:y1, x0:x1] = 0


def _centre_fill(lab, color, x, y, radius, tolerance):
    r = max(1, int(round(0.4 * radius)))
    yy, xx = int(round(y)), int(round(x))
    window = lab[max(0, yy - r):yy + r + 1, max(0, xx - r):xx + r + 1]
    if not window.size:
        return 0.0
    return float((np.linalg.norm(window - np.float32(color), axis=2) < tolerance).mean())


def _pixel_x(value, model):
    slope, intercept, scale = float(model["slope"]), float(model["intercept"]), model.get("scale", "linear")
    if scale == "log":
        if value <= 0:
            return None
        value = math.log10(value)
    return (value - intercept) / slope if slope else None


def _expected_y(series, px_x):
    """Curve y (pixels) at column px_x from the series' own points; None when not bracketed."""
    pts = sorted((p[0], p[1]) for p in series["pts"])
    left = [p for p in pts if p[0] <= px_x]
    right = [p for p in pts if p[0] >= px_x]
    if left and right:
        (x0, y0), (x1, y1) = left[-1], right[0]
        return y0 if x1 == x0 else y0 + (y1 - y0) * (px_x - x0) / (x1 - x0)
    near = left[-1:] or right[:1]
    return near[0][1] if near else None


def glyph_radius(glyphs, series, fallback):
    """Marker radius from the measured legend glyph (all markers of a chart share one size)."""
    glyph = glyphs[series["idx"]] if glyphs and series["idx"] < len(glyphs) else None
    size = float((glyph or {}).get("size_px") or 0)
    return size / 2 if size >= 4 else fallback


def apply(all_series, spec, lab, x_model, r_med, glyphs=None):
    """Apply ignore / split_filled_open / sample_band_at_columns. Returns (band_points, audit).

    ``band_points`` = {series index: [(x_px, y_px)]} - emitted as line samples, not markers.
    """
    regions = spec.get("region_strategies") or []
    by_label = {_norm(series["spec"].get("label")): series for series in all_series}
    band_points, audit = {}, []
    for region in regions:
        box, strategy = region["bbox_px"], region.get("strategy")
        listed = [by_label[_norm(label)] for label in region.get("series_labels") or [] if _norm(label) in by_label]
        entry = {"strategy": strategy, "bbox_px": box, "series": [s["spec"].get("label") for s in listed]}
        if strategy == "ignore" and listed:
            removed = 0
            for series in listed:
                before = len(series["pts"])
                series["pts"] = [p for p in series["pts"] if not _inside(box, p[0], p[1])]
                removed += before - len(series["pts"])
            entry["removed"] = removed
        elif strategy == "split_filled_open":
            filled = [s for s in listed if str(s["spec"].get("marker_fill")) == "filled"]
            opened = [s for s in listed if str(s["spec"].get("marker_fill")) == "open"]
            if not filled or not opened:
                entry["skipped"] = "needs at least one filled and one open series"
                audit.append(entry)
                continue
            pool = []
            for series in listed:
                pool += [(series, p) for p in series["pts"] if _inside(box, p[0], p[1])]
                series["pts"] = [p for p in series["pts"] if not _inside(box, p[0], p[1])]
            moved = 0
            for origin, point in pool:
                radius = glyph_radius(glyphs, origin, r_med)
                share = _centre_fill(lab, origin["col"], point[0], point[1], radius, origin["tol"])
                targets = filled if share >= 0.5 else opened
                target = min(targets, key=lambda s: float(np.linalg.norm(s["col"] - origin["col"])))
                if any(math.hypot(q[0] - point[0], q[1] - point[1]) < 0.5 * r_med for q in target["pts"]):
                    continue  # the same marker already assigned
                target["pts"].append(point)
                moved += target is not origin
            for series in listed:
                series["pts"].sort()
            entry.update(points=len(pool), reassigned=moved)
        elif strategy == "sample_band_at_columns":
            columns = [c for c in spec.get("shared_x_columns") or []]
            added = 0
            for series in listed:
                mask = (np.linalg.norm(lab - np.float32(series["col"]), axis=2) < series["tol"]).astype(np.uint8)
                for value in columns:
                    px_x = _pixel_x(float(value), x_model)
                    if px_x is None or not box[0] <= px_x <= box[2]:
                        continue
                    if any(abs(p[0] - px_x) < 0.8 * r_med for p in series["pts"]):
                        continue  # a real marker already sits in this column
                    col = int(round(px_x))
                    strip = mask[box[1]:box[3], max(0, col - 1):col + 2].max(axis=1)
                    rows = np.flatnonzero(strip)
                    if not len(rows):
                        continue
                    runs = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
                    centres = [box[1] + float(run.mean()) for run in runs]
                    expected = _expected_y(series, px_x)
                    y = min(centres, key=lambda c: abs(c - expected)) if expected is not None else centres[0]
                    band_points.setdefault(series["idx"], []).append((float(px_x), y))
                    added += 1
            entry["added_line_samples"] = added
            if not columns:
                entry["skipped"] = "no shared_x_columns given"
        audit.append(entry)
    return band_points, audit

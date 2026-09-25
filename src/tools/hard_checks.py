"""Hard checks on an extraction (python, agent03 or agent04 stage) - deterministic, no model call.

Agents judge images; these checks judge numbers, so no agent can approve values that the axes
contradict.  Three checks, saved as ``<stage>_hard_checks.json`` with ``<stage>_redraw_diff.png``:

  axis_range  (blocking)  every value lies inside the printed axis range (plus ``axis_range_margin``
                          of the span), and the plot frame, converted with the fitted calibration,
                          overhangs the outer printed ticks by at most ``axis_frame_max_overhang_steps``
                          tick steps.  A shifted or stretched tick mapping fails here even when every
                          marker is found at the right pixel (the overlay can not show that).
  redraw      (score)     per series: recall = share of the native marker ink of the series color
                          covered by an extracted point, precision = share of extracted points that
                          sit on that ink.  The diff image shows missed ink red, unsupported points blue.
  physics     (flags)     src/tools/checks.py: isotherms rise with pressure, colder above warmer, ...

Only ``axis_range`` blocks.  The other two are evidence for agents 02-04 and the review page.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.settings import CFG
from src.tools import checks


def _setting(key, default):
    return float((CFG.get("checks") or {}).get(key, default))


def _printed_range(axis: dict) -> tuple[float, float] | None:
    ticks = [float(v) for v in axis.get("ticks") or []]
    if len(ticks) < 2:
        return None
    low, high = min(ticks), max(ticks)
    if axis.get("minimum") is not None:
        low = min(low, float(axis["minimum"]))
    if axis.get("maximum") is not None:
        high = max(high, float(axis["maximum"]))
    return low, high


def _to_scale(value: float, scale: str) -> float | None:
    if scale == "log":
        return math.log10(value) if value > 0 else None
    return value


def _tick_step(axis: dict, scale: str) -> float:
    values = sorted(v for v in (_to_scale(float(t), scale) for t in axis.get("ticks") or []) if v is not None)
    steps = [b - a for a, b in zip(values, values[1:]) if b > a]
    return float(np.median(steps)) if steps else 0.0


def _axis_value(model: dict, pixel: float) -> float:
    """Calibrated value in scale space (log10 for log axes)."""
    return float(model["slope"]) * float(pixel) + float(model["intercept"])


def axis_range(extraction: dict, spec: dict) -> dict:
    """Data and frame ranges against the printed ticks, per axis."""
    margin = _setting("axis_range_margin", 0.05)
    max_overhang = _setting("axis_frame_max_overhang_steps", 1.0)
    calibration = extraction.get("calibration") or {}
    models = calibration.get("axis_models") or {}
    frame = calibration.get("frame_px") or []
    report = {"passed": True, "problems": [], "margin": margin, "max_frame_overhang_steps": max_overhang}
    for axis_id in ("x", "y"):
        axis = spec.get(axis_id) or {}
        scale = str(axis.get("scale") or "linear")
        printed = _printed_range(axis)
        values = [float(p[axis_id]) for s in extraction.get("series", []) for p in s.get("points", [])
                  if p.get(axis_id) is not None]
        entry = {"unit": axis.get("unit", ""), "scale": scale,
                 "printed": list(printed) if printed else None,
                 "data": [min(values), max(values)] if values else None, "points_outside": 0, "passed": True}
        report[axis_id] = entry
        if printed is None:
            continue
        low, high = (_to_scale(v, scale) for v in printed)
        if low is None or high is None:
            continue
        pad = margin * (high - low)
        allowed = [low - pad, high + pad]
        model = models.get(axis_id)
        step = _tick_step(axis, scale)
        if model and len(frame) == 4 and step > 0:
            ends = (frame[0], frame[2]) if axis_id == "x" else (frame[3], frame[1])
            frame_low, frame_high = sorted(_axis_value(model, pixel) for pixel in ends)
            overhang = [(low - frame_low) / step, (frame_high - high) / step]
            entry["frame"] = [frame_low, frame_high] if scale != "log" else [10 ** frame_low, 10 ** frame_high]
            entry["frame_overhang_steps"] = [round(v, 2) for v in overhang]
            if min(overhang) < -max_overhang or max(overhang) > max_overhang:
                entry["passed"] = False
                report["problems"].append(
                    f"{axis_id}: the plot frame reads {entry['frame'][0]:.4g} to {entry['frame'][1]:.4g} "
                    f"{entry['unit']} but the printed ticks run {printed[0]:g}-{printed[1]:g}; overhang "
                    f"{overhang[0]:.2f} / {overhang[1]:.2f} tick steps - the tick-to-pixel mapping is shifted or stretched")
            else:  # a consistent frame may run past the outer tick: data drawn inside it is real
                allowed = [min(allowed[0], frame_low), max(allowed[1], frame_high)]
        entry["allowed"] = allowed if scale != "log" else [10 ** v for v in allowed]
        scaled = [_to_scale(v, scale) for v in values]
        outside = [v for v, s in zip(values, scaled) if s is None or s < allowed[0] or s > allowed[1]]
        entry["points_outside"] = len(outside)
        if outside:
            entry["passed"] = False
            report["problems"].append(
                f"{axis_id}: {len(outside)} of {len(values)} values outside the axis "
                f"({entry['allowed'][0]:.4g} to {entry['allowed'][1]:.4g} {entry['unit']}, printed "
                f"{printed[0]:g}-{printed[1]:g}; data {min(values):.4g} to {max(values):.4g})")
        report["passed"] = report["passed"] and entry["passed"]
    return report


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Open markers become solid discs, so they survive the line-removing opening."""
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    return (mask | (1 - flood[1:-1, 1:-1])).astype(np.uint8)


def _panel_crop(spec: dict) -> np.ndarray | None:
    image = cv2.imread(str(spec.get("image") or ""))
    if image is None:
        return None
    x0, y0, x1, y1 = spec["panel_bbox"]
    return image[y0:y1, x0:x1]


def _hex_lab(value):
    text = str(value or "").lstrip("#")
    if len(text) != 6:
        return None
    rgb = [int(text[k:k + 2], 16) for k in (0, 2, 4)]
    return cv2.cvtColor(np.uint8([[rgb[::-1]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def _uses_direct_labels(spec: dict) -> bool:
    position = str(spec.get("legend_position") or (spec.get("llm_spec") or {}).get("legend_position") or "").lower()
    return "direct" in position and ("label" in position or "annot" in position)


def _ink_labels(lab_img, plot, colors, background, max_distance=35.0):
    """Index of the nearest declared series color per plot pixel, -1 for background / no series."""
    stack = np.stack([np.linalg.norm(lab_img - np.float32(c), axis=2) for c in colors] +
                     [np.linalg.norm(lab_img - np.float32(background), axis=2)], axis=0)
    nearest = stack.argmin(axis=0)
    labels = np.where((nearest < len(colors)) & (stack.min(axis=0) < max_distance) & (plot > 0), nearest, -1)
    return labels


def redraw(extraction: dict, spec: dict, out_png=None) -> dict:
    """Recall / precision of the extracted points against the native marker ink, per series.

    Every plot pixel goes to its nearest declared series color (emitted or excluded), so ink of
    one series is never counted as missed for another.  Series whose colors are nearly identical
    form a group: a marker held by any group member counts as found; a group that contains an
    excluded series has no measurable recall (its ink can not be split by color).
    """
    crop = _panel_crop(spec)
    calibration = extraction.get("calibration") or {}
    frame = calibration.get("frame_px") or []
    if crop is None or len(frame) != 4:
        return {"available": False}
    left, top, right, bottom = map(int, frame)
    plot = np.zeros(crop.shape[:2], np.uint8)
    plot[top + 2:bottom - 1, left + 2:right - 1] = 1
    legend = spec.get("legend_bbox") or []
    if len(legend) == 4 and not _uses_direct_labels(spec):
        plot[legend[1]:legend[3], legend[0]:legend[2]] = 0
    for box in spec.get("mask_bboxes") or []:
        plot[box[1]:box[3], box[0]:box[2]] = 0
    lab_img = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    if not plot.any():
        return {"available": False}
    background = np.median(lab_img[plot > 0][::7], axis=0)

    emitted = {s["label"]: s for s in extraction.get("series", [])}
    members = []  # (label, emitted?, [color prototypes in Lab]): measured on the plot and declared
    for declared in spec.get("series", []):
        label = declared.get("label")
        prototypes = [np.float32(c) for c in [(emitted.get(label) or {}).get("color_lab")] if c]
        prototypes += [c for c in [_hex_lab(declared.get("color_hex"))] if c is not None]
        if prototypes:
            members.append((label, label in emitted, prototypes))
    if not members:
        return {"available": False}
    owner = [index for index, (_, _, protos) in enumerate(members) for _ in protos]
    nearest = _ink_labels(lab_img, plot, [c for _, _, protos in members for c in protos], background)
    labels = np.where(nearest >= 0, np.array(owner + [-1])[nearest], -1)

    def close(i, j):
        return min(float(np.linalg.norm(a - b)) for a in members[i][2] for b in members[j][2]) < 20

    heat = (0.35 * crop + 0.65 * 255).astype(np.uint8)
    report = {"available": True, "series": {}}
    for index, (label, is_emitted, _) in enumerate(members):
        series = emitted.get(label)
        if not is_emitted or series is None:
            continue
        group = [j for j in range(len(members)) if close(index, j)]
        radius = max(2.0, float(series.get("marker_radius_px") or 4.0))
        group_ink = np.isin(labels, group).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.6 * radius) + 1,) * 2)
        blobs = cv2.morphologyEx(_fill_holes(group_ink), cv2.MORPH_OPEN, kernel)  # connecting lines thinner than a marker go
        covered = np.zeros_like(blobs)
        for j in group:
            for p in (emitted.get(members[j][0]) or {}).get("points", []):
                if p.get("px"):
                    cv2.circle(covered, (int(round(p["px"][0])), int(round(p["px"][1]))), int(round(1.2 * radius)), 1, -1)
        points = [(float(p["px"][0]), float(p["px"][1])) for p in series.get("points", []) if p.get("px")]
        on_ink = []
        for x, y in points:
            yy, xx, r = int(round(y)), int(round(x)), int(radius)
            window = group_ink[max(0, yy - r):yy + r + 1, max(0, xx - r):xx + r + 1]
            on_ink.append(bool(window.size and window.mean() >= 0.2))
        missed = blobs & (1 - covered)
        excluded_in_group = [members[j][0] for j in group if not members[j][1]]
        ink_px = int(blobs.sum())
        entry = {
            "points": len(points),
            "precision": round(sum(on_ink) / len(points), 3) if points else None,
            "recall": round(1 - float(missed.sum()) / ink_px, 3) if ink_px else None,
            "missed_marker_equivalents": round(float(missed.sum()) / (math.pi * radius * radius), 1),
            "same_color_group": [members[j][0] for j in group],
        }
        if excluded_in_group:
            entry.update(recall=None, missed_marker_equivalents=None,
                         note="recall not measurable: same color as excluded series " + ", ".join(map(str, excluded_in_group)))
        else:
            heat[missed > 0] = (40, 40, 230)
        report["series"][label] = entry
        for (x, y), ok in zip(points, on_ink):
            cv2.circle(heat, (int(round(x)), int(round(y))), int(round(radius)) + 1,
                       (40, 170, 40) if ok else (230, 120, 20), 1)
    recalls = [v["recall"] for v in report["series"].values() if v["recall"] is not None]
    precisions = [v["precision"] for v in report["series"].values() if v["precision"] is not None]
    report["recall"] = round(float(np.mean(recalls)), 3) if recalls else None
    report["precision"] = round(float(np.mean(precisions)), 3) if precisions else None
    if out_png is not None:
        cv2.putText(heat, "red = native marker ink not extracted  green = point on ink  blue = point without ink",
                    (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_png), heat)
    return report


def run(extraction: dict, spec: dict, out_dir=None, stage: str = "python") -> dict:
    """All hard checks for one stage; ``passed`` is False only when the axis range check fails."""
    axes = axis_range(extraction, spec)
    diff_png = None if out_dir is None else f"{out_dir}/{stage}_redraw_diff.png"
    try:
        physics = checks.run(extraction)
    except Exception as error:  # physics flags are evidence only; never let them break a stage
        physics = {"error": str(error)}
    unresolved = extraction.get("unresolved_slots") or {}
    detector = (extraction.get("calibration") or {}).get("dense_circle_detector") or {}
    limitations = {
        "unresolved_slots": int(unresolved.get("total", 0)),
        "dense_regions_with_no_distinct_centres": int(detector.get("unresolved_fused_regions", 0)),
        "note": "ink coverage is not marker completeness; unresolved dense evidence requires review",
    }
    report = {
        "stage": stage,
        "passed": axes["passed"],
        "problems": list(axes["problems"]),
        "axis_range": axes,
        "redraw": redraw(extraction, spec, diff_png),
        "physics": {"verdict": physics.get("verdict"),
                    "flags": {label: item.get("flags", []) for label, item in (physics.get("series") or {}).items()},
                    "cross_series": physics.get("cross", [])},
        "limitations": limitations,
    }
    # Keep the historical axis-only ``passed`` result, but expose the
    # separate acceptance status so callers cannot mistake a good redraw for
    # complete dense-marker evidence.
    report["accepted_extraction"] = bool(report["passed"] and not limitations["unresolved_slots"])
    return report


def summary_line(report: dict) -> str:
    """One line for prompts and the run report."""
    axes = report["axis_range"]
    parts = []
    for axis_id in ("x", "y"):
        entry = axes.get(axis_id) or {}
        if entry.get("printed") and entry.get("data"):
            parts.append(f"{axis_id} printed {entry['printed'][0]:g}-{entry['printed'][1]:g}, "
                         f"data {entry['data'][0]:.4g}-{entry['data'][1]:.4g} {entry.get('unit', '')}".strip())
    redraw_ = report.get("redraw") or {}
    if redraw_.get("recall") is not None:
        parts.append(f"redraw recall {redraw_['recall']:.2f}, precision {redraw_['precision']:.2f}")
    return ("PASS" if report["passed"] else "FAIL") + ": " + "; ".join(parts)

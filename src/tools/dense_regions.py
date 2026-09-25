"""Evidence-bound handling for crowded, low-pressure marker regions.

This module deliberately separates a *slot hypothesis* from a coordinate.  A
shared pressure column can make a missing marker worth reviewing, but it can
never create a CSV point: only a local color/template core may do that.
"""
from __future__ import annotations

from collections import Counter
import json
import pathlib
import cv2
import numpy as np
from src.tools import marker_assignment


def validate_complex_regions(regions, frame_px, fx, x_limits, panel_shape):
    """Convert accepted panel-normalized boxes to plot pixels.

    A region described as low-pressure must actually be confined to the low-x
    part of the calibrated plot.  This catches broad horizontal bands (often
    the N2 traces) that happen to sit low in image coordinates.
    """
    left, top, right, bottom = map(float, frame_px)
    height, width = panel_shape[:2]
    xmin, xmax = map(float, x_limits)
    span = max(1e-9, xmax - xmin)
    accepted, rejected = [], []
    for raw in regions or []:
        box = raw.get("bbox_norm") or []
        if len(box) != 4:
            rejected.append({"region": raw, "reason": "missing_bbox_norm"}); continue
        x0, y0, x1, y1 = map(float, box)
        xa, xb = sorted((x0 * width, x1 * width)); ya, yb = sorted((y0 * height, y1 * height))
        clipped = (max(left, xa), max(top, ya), min(right, xb), min(bottom, yb))
        if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
            rejected.append({"region": raw, "reason": "outside_calibrated_plot"}); continue
        kind = str(raw.get("kind") or "").lower()
        # Read legacy saved runs, but new Agent 02 responses are schema-bound
        # to `kind`; this is not a free-text classification path.
        if not kind and raw.get("reason"):
            kind = "dense_markers" if "dense" in str(raw["reason"]).lower() else "annotation"
        if kind not in {"dense_markers", "overlapping_markers", "thick_connecting_lines", "crossing", "inset", "annotation"}:
            rejected.append({"region": raw, "reason": "invalid_region_kind"}); continue
        data_x0, data_x1 = sorted((float(fx(clipped[0])), float(fx(clipped[2]))))
        if (raw.get("low_pressure") or "low-pressure" in str(raw.get("reason") or "").lower()) and (data_x1 - xmin) / span > 0.35:
            rejected.append({"region": raw, "reason": "low_pressure_box_extends_beyond_low_x_domain",
                             "data_x": [data_x0, data_x1]}); continue
        accepted.append({"bbox_px": [round(v, 2) for v in clipped], "data_x": [data_x0, data_x1], **raw})
    return accepted, rejected


def infer_low_pressure_roi(
    frame_px,
    all_series,
    fraction=0.15,
    *,
    axis_scale="linear",
    axis_limits=None,
    slope=None,
    intercept=None,
):
    """Return a frame/axis-derived low-x strip spanning the full plot height.

    ``fraction`` is a fraction of the *data* domain when calibration details
    are supplied (including on logarithmic axes).  The pixel-fraction fallback
    preserves compatibility with direct callers that only know the frame.
    Existing marker y coordinates deliberately do not trim the strip: missing
    starts often sit above or below the already detected fan.
    """
    left, top, right, bottom = map(float, frame_px)
    fraction = min(0.5, max(0.01, float(fraction)))
    cut = left + fraction * (right - left)
    limits = list(axis_limits or [])
    strip_start, strip_end = None, None
    if len(limits) == 2 and slope not in (None, 0):
        low, high = sorted(float(value) for value in limits)
        if axis_scale == "log" and low > 0 and high > low:
            strip_start = low
            strip_end = low * (high / low) ** fraction
            start_transformed = float(np.log10(strip_start))
            end_transformed = float(np.log10(strip_end))
        else:
            # Axes whose frame extends below zero often include blank negative
            # padding. Start the search at zero so that padding does not use up
            # the crowded-strip width.
            strip_start = 0.0 if low <= 0 < high else low
            strip_end = strip_start + fraction * (high - strip_start)
            start_transformed, end_transformed = strip_start, strip_end
        start_px = (start_transformed - float(intercept or 0.0)) / float(slope)
        end_px = (end_transformed - float(intercept or 0.0)) / float(slope)
        if np.isfinite(start_px) and np.isfinite(end_px):
            strip_left, strip_right = sorted((
                min(right, max(left, start_px)),
                min(right, max(left, end_px)),
            ))
            if strip_right > strip_left:
                return [strip_left, top, strip_right, bottom]
    cut = left + fraction * (right - left)
    # Pixel-fraction fallback is conservative for callers without axes.
    return [left, top, cut, bottom]


def infer_pressure_columns(all_series, roi, radius):
    """Infer columns from clear native markers, not Agent 01's grid hypothesis."""
    xa, ya, xb, yb = roi
    hits = []
    for series in all_series:
        for x, y, confidence, overlap, sure in series.get("pts", []):
            if xa <= x <= xb and ya <= y <= yb and sure and confidence >= .65:
                hits.append((x, series["idx"]))
    hits.sort()
    columns = []
    for x, owner in hits:
        if not columns or x - columns[-1]["x"] > max(2.0, .8 * radius):
            columns.append({"x": float(x), "owners": {owner}})
        else:
            col = columns[-1]; col["owners"].add(owner); col["x"] = (col["x"] + x) / 2
    return [col for col in columns if len(col["owners"]) >= 2]


def _local_core(lab, color, tol, roi, column_x, y_guess, radius, line_width=1.0, native_mask=None):
    xa, ya, xb, yb = map(int, roi)
    search = max(4, int(round(1.5 * radius)))
    x0, x1 = max(xa, int(column_x - search)), min(xb, int(column_x + search + 1))
    y0, y1 = max(ya, int(y_guess - 2.5 * radius)), min(yb, int(y_guess + 2.5 * radius + 1))
    if x0 >= x1 or y0 >= y1: return None
    if native_mask is None:
        dist = np.linalg.norm(lab[y0:y1, x0:x1] - color[None, None, :], axis=2)
        mask = (dist < tol).astype(np.uint8)
    else:
        mask = (native_mask[y0:y1, x0:x1] > 0).astype(np.uint8)
    # Lines are thin; a marker has a distance-transform core.  This is the
    # line-suppressed residual used for resolution, not a fitted trajectory.
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, peak, _, loc = cv2.minMaxLoc(dt)
    # A thick connector can have a real distance-transform peak.  It is only
    # marker evidence when its core exceeds the declared/observed line width.
    if peak < max(1.0, .32 * radius, 0.75 * line_width): return None
    return float(x0 + loc[0]), float(y0 + loc[1]), float(peak)


def resolve_dense_region(all_series, lab, roi, radius, native_mask=None, geometry_indices=None):
    """Jointly resolve native-supported cores and record every unsupported slot.

    The one-series/one-column constraint is enforced by considering each pair
    once.  Existing measured points win; no coordinate is interpolated.
    """
    columns = infer_pressure_columns(all_series, roi, radius)
    resolved, unresolved = [], []
    for column_index, column in enumerate(columns):
        for series in all_series:
            points = series.get("pts", [])
            series_radius = max(2.0, float(series.get("r") or radius))
            native_points = [
                point for point in points
                if len(point) >= 5 and bool(point[4]) and float(point[2]) >= 0.65
            ]
            # Trajectory continuity supplies a local search window only. At a
            # boundary, one-sided native anchors may guide that search; the
            # native core check below must still confirm the marker.
            left = [p for p in native_points if p[0] < column["x"]]
            right = [p for p in native_points if p[0] > column["x"]]
            if left and right:
                anchor_left = max(left, key=lambda p: p[0])
                anchor_right = min(right, key=lambda p: p[0])
                if anchor_left[0] == anchor_right[0]:
                    continue
                trajectory_slope = (anchor_right[1] - anchor_left[1]) / (anchor_right[0] - anchor_left[0])
                y_guess = anchor_left[1] + trajectory_slope * (column["x"] - anchor_left[0])
                trajectory_mode = "interpolated_native_anchors"
            elif right:
                right_ordered = sorted(right, key=lambda p: p[0])
                anchors = right_ordered[:1] + [p for p in right_ordered[1:] if p[0] != right_ordered[0][0]][:1]
                anchor = anchors[0]
                anchor_gap = abs(float(anchor[0]) - float(column["x"]))
                if len(anchors) == 2:
                    trajectory_slope = (anchors[1][1] - anchors[0][1]) / (anchors[1][0] - anchors[0][0])
                    y_guess = anchor[1] + trajectory_slope * (column["x"] - anchor[0])
                else:
                    y_guess = anchor[1]
                trajectory_mode = "one_sided_native_search"
            elif left:
                left_ordered = sorted(left, key=lambda p: p[0], reverse=True)
                anchors = left_ordered[:1] + [p for p in left_ordered[1:] if p[0] != left_ordered[0][0]][:1]
                anchor = anchors[0]
                anchor_gap = abs(float(anchor[0]) - float(column["x"]))
                if len(anchors) == 2:
                    trajectory_slope = (anchors[0][1] - anchors[1][1]) / (anchors[0][0] - anchors[1][0])
                    y_guess = anchor[1] + trajectory_slope * (column["x"] - anchor[0])
                else:
                    y_guess = anchor[1]
                trajectory_mode = "one_sided_native_search"
            else:
                unresolved.append({"series_label": series["spec"].get("label"), "series_index": series["idx"], "column_index": column_index, "column_px": round(column["x"], 2), "state": "unresolved", "reason": "no_native_trajectory_anchors"}); continue
            if trajectory_mode == "one_sided_native_search":
                neighbor_gaps = [
                    abs(float(a[0]) - float(b[0]))
                    for a, b in zip(sorted(native_points, key=lambda p: p[0]), sorted(native_points, key=lambda p: p[0])[1:])
                ]
                typical_gap = float(np.median(neighbor_gaps)) if neighbor_gaps else 0.0
                search_span = max(4.0 * series_radius, 2.0 * typical_gap)
                if anchor_gap > search_span:
                    unresolved.append({
                        "series_label": series["spec"].get("label"), "series_index": series["idx"],
                        "column_index": column_index, "column_px": round(column["x"], 2),
                        "predicted_y_px": round(float(y_guess), 2), "state": "unresolved",
                        "reason": "one_sided_search_outside_native_anchor_span",
                    })
                    continue
            if not (roi[1] <= y_guess <= roi[3]):
                continue
            # Column membership is two-dimensional.  A real marker stacked
            # above or below this prediction does not occupy the predicted
            # marker slot merely because it shares an x coordinate.
            if any(np.hypot(p[0] - column["x"], p[1] - y_guess) < .8 * series_radius for p in points):
                continue
            core = _local_core(
                lab,
                series["col"],
                series["tol"],
                roi,
                column["x"],
                y_guess,
                series_radius,
                series.get("line_width", 1.0),
                native_mask=native_mask if series["idx"] in set(geometry_indices or ()) else None,
            )
            slot = {"series_label": series["spec"].get("label"), "series_index": series["idx"],
                    "column_index": column_index, "column_px": round(column["x"], 2),
                    "predicted_y_px": round(float(y_guess), 2), "state": "unresolved",
                    "trajectory_mode": trajectory_mode}
            if core is None:
                slot["reason"] = "no_native_marker_core_after_line_suppression"; unresolved.append(slot); continue
            x, y, core_radius = core
            # A core must be genuinely marker-sized and inside the ROI. This
            # rejects a one-pixel colored curve even when continuity predicts it.
            if core_radius < .45 * series_radius:
                slot["reason"] = "line_like_residual_not_marker"; unresolved.append(slot); continue
            resolved.append({"series_index": series["idx"], "column_index": column_index,
                             "px": [round(x, 2), round(y, 2)], "confidence": .65,
                             "core_radius": round(core_radius, 2)})
    # Same-color, multi-series cores remain hypotheses, never rows.
    by_location = {}
    for item in resolved:
        by_location.setdefault((item["column_index"], round(item["px"][0]), round(item["px"][1])), []).append(item)
    for items in by_location.values():
        if len(items) > 1:
            for item in items:
                unresolved.append({"series_label": all_series[item["series_index"]]["spec"].get("label"), "series_index": item["series_index"], "column_index": item["column_index"], "column_px": item["px"][0], "predicted_y_px": item["px"][1], "state": "hypothesis", "reason": "multi_series_overlap_hypothesis"})
            resolved = [item for item in resolved if item not in items]
    # Defensive duplicate prevention: a series owns at most one point per slot.
    unique = {}
    for item in resolved:
        key = (item["series_index"], item["column_index"])
        if key not in unique or item["core_radius"] > unique[key]["core_radius"]: unique[key] = item
    return list(unique.values()), unresolved, columns


def unresolved_summary(slots):
    counts = Counter(slot["series_label"] for slot in slots)
    return {"total": len(slots), "by_series": dict(sorted(counts.items())), "slots": slots}


def write_evidence(image, extraction, spec, out_dir):
    """Write Agent 03 inspection views for the calibrated dense ROI."""
    info = (extraction.get("calibration") or {}).get("dense_region_resolver") or {}
    roi = info.get("low_pressure_strip_px") or info.get("roi_px")
    if roi and isinstance(roi[0], (list, tuple)):
        roi = roi[0]  # a representative evidence tile; manifest contains all typed ROIs
    if not roi:
        return {}
    xa, ya, xb, yb = map(int, roi)
    panel = image.copy()
    native = panel[ya:yb, xa:xb]
    if native.size == 0:
        return {}
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    native_path = out_dir / "dense_roi_native.png"
    cv2.imwrite(str(native_path), cv2.resize(native, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
    # Morphological opening removes broad marker cores; residual is primarily
    # thin fitted lines, making the distinction inspectable without claiming a point.
    residual = cv2.absdiff(native, cv2.morphologyEx(native, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)))
    residual_path = out_dir / "dense_roi_line_suppressed.png"
    cv2.imwrite(str(residual_path), cv2.resize(residual, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
    likelihood = native.copy()
    for series in extraction.get("series", []):
        for point in series.get("points", []):
            px = point.get("px") or []
            if len(px) == 2 and xa <= px[0] <= xb and ya <= px[1] <= yb:
                cv2.circle(likelihood, (int(px[0] - xa), int(px[1] - ya)), 4, (0, 255, 255), 1)
    likelihood_path = out_dir / "dense_roi_series_likelihood.png"
    cv2.imwrite(str(likelihood_path), cv2.resize(likelihood, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
    assignments = likelihood.copy()
    for slot in (extraction.get("unresolved_slots") or {}).get("slots", []):
        column_px = slot.get("column_px")
        predicted_y_px = slot.get("predicted_y_px")
        # Some unresolved slots intentionally have no y coordinate.  In
        # particular, a trajectory that is not bracketed by native anchors is
        # evidence of a missing assignment, not evidence of a pixel location.
        # Keep it in the ledger but do not invent a marker for this image.
        if not isinstance(column_px, (int, float)) or not isinstance(
            predicted_y_px, (int, float)
        ):
            continue
        if not np.isfinite(column_px) or not np.isfinite(predicted_y_px):
            continue
        x, y = int(column_px - xa), int(predicted_y_px - ya)
        cv2.drawMarker(assignments, (x, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 7, 1)
    assignments_path = out_dir / "dense_roi_proposed_assignments.png"
    cv2.imwrite(str(assignments_path), cv2.resize(assignments, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
    evidence = {"native": native_path.name, "line_suppressed": residual_path.name,
                "series_likelihood": likelihood_path.name, "proposed_assignments": assignments_path.name}

    # Preserve untouched native tiles for every disputed color-ambiguous
    # candidate.  Agent 03 can inspect these without mistaking a Python ring
    # or score annotation for coordinate evidence.
    audit = (extraction.get("calibration") or {}).get("series_assignment_audit") or {}
    disputed = [
        candidate for candidate in audit.get("candidates", [])
        if candidate.get("action") in {"unresolved", "reassigned"}
    ]
    native_tiles = []
    score_overlay = panel.copy()
    for index, candidate in enumerate(disputed[:80], start=1):
        px = candidate.get("px") or []
        if len(px) != 2:
            continue
        cx, cy = map(int, map(round, px))
        xa, xb = max(0, cx - 32), min(panel.shape[1], cx + 33)
        ya, yb = max(0, cy - 32), min(panel.shape[0], cy + 33)
        tile = panel[ya:yb, xa:xb]
        if tile.size == 0:
            continue
        tile_path = out_dir / f"ambiguity_{index:03d}_native.png"
        cv2.imwrite(str(tile_path), cv2.resize(tile, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
        native_tiles.append(tile_path.name)
        color = (0, 0, 255) if candidate.get("action") == "unresolved" else (0, 165, 255)
        cv2.circle(score_overlay, (cx, cy), 8, color, 1)
        cv2.putText(score_overlay, str(index), (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    if native_tiles:
        score_path = out_dir / "ambiguity_assignment_scores.png"
        cv2.imwrite(str(score_path), score_overlay)
        diagnostics_path = out_dir / "assignment_diagnostics.json"
        diagnostics_path.write_text(json.dumps(audit, indent=1), encoding="utf-8")
        evidence["ambiguity_native_tiles"] = native_tiles
        evidence["ambiguity_assignment_scores"] = score_path.name
        evidence["assignment_diagnostics"] = diagnostics_path.name
    return evidence

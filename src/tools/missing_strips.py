"""Optional, pixel-confirmed recovery for unresolved crowded-strip slots.

This is deliberately a small deterministic pass.  It consumes slots already
recorded by the extractor; it never asks a model for coordinates and it never
turns a trajectory prediction into a point without both native color and
marker-shape support. It requires verified chart structure/calibration plus an
explicit undercoverage signal. Agent 02 may remain dissatisfied because of
that coverage gap.
"""

from __future__ import annotations

import json
import math

import cv2
import numpy as np

from src.tools import extract


_REQUEST_WORDS = (
    "undercoverage",
    "under-coverage",
    "missing marker",
    "missing markers",
    "marker shortfall",
    "coverage shortfall",
    "unresolved slot",
    "unresolved slots",
)


def _verified_setup(check, extraction, spec):
    """Require stable chart structure and valid calibrated axes before recovery."""
    if not isinstance(check, dict) or check.get("structure_ok") is not True:
        return False, "Agent 02 has not verified chart structure"
    axis_check = check.get("axis_check") or {}
    for axis in ("x", "y"):
        if (axis_check.get(axis) or {}).get("result") != "pass":
            return False, f"Agent 02 has not verified the {axis} axis"
    calibration = extraction.get("calibration") or {}
    frame = calibration.get("frame_px") or spec.get("panel_bbox") or []
    models = calibration.get("axis_models") or {}
    try:
        if len(frame) != 4 or float(frame[2]) <= float(frame[0]) or float(frame[3]) <= float(frame[1]):
            return False, "invalid calibrated plot frame"
        for axis in ("x", "y"):
            model = models.get(axis) or {}
            slope, intercept = float(model["slope"]), float(model["intercept"])
            if not math.isfinite(slope) or not math.isfinite(intercept) or slope == 0:
                return False, f"invalid {axis} axis calibration"
    except (KeyError, TypeError, ValueError):
        return False, "missing calibrated plot axes"
    return True, "verified structure and calibrated axes"


def _native_shape_support(spec, color_lab, radius, center, tolerance=42.0):
    """Reject line-like color hits; accept a compact core or a marker-shaped arc."""
    image = cv2.imread(str(spec.get("image") or ""))
    bbox = spec.get("panel_bbox") or []
    if image is None or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = map(int, bbox)
    panel = image[y0:y1, x0:x1]
    cx, cy = map(float, center)
    radius = max(2.0, float(radius))
    pad = max(5, int(math.ceil(2.0 * radius)))
    xa, xb = max(0, int(cx) - pad), min(panel.shape[1], int(cx) + pad + 1)
    ya, yb = max(0, int(cy) - pad), min(panel.shape[0], int(cy) + pad + 1)
    if xa >= xb or ya >= yb:
        return None
    lab = extract.to_lab(panel[ya:yb, xa:xb])
    color = np.asarray(color_lab, dtype=np.float32)
    # Try the strict tolerance first, then a modest antialias allowance. A
    # larger tolerance can merge a thin connector into the candidate shape.
    for tol in (min(30.0, tolerance), min(42.0, tolerance), min(54.0, tolerance)):
        mask = extract.color_mask(lab, color, tol)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        candidates = []
        for label in range(1, count):
            bx, by, bw, bh, area = map(int, stats[label])
            if area < 5:
                continue
            component = labels == label
            ys, xs = np.where(component)
            abs_xs, abs_ys = xs + xa, ys + ya
            nearest = float(np.min(np.hypot(abs_xs - cx, abs_ys - cy)))
            centroid_x, centroid_y = centroids[label][0] + xa, centroids[label][1] + ya
            if nearest > max(2.0, 0.65 * radius) or math.hypot(centroid_x - cx, centroid_y - cy) > 1.25 * radius:
                continue
            local = component.astype(np.uint8)
            core = float(cv2.distanceTransform(local, cv2.DIST_L2, 5).max())
            aspect = max(bw, bh) / max(1, min(bw, bh))
            coords = np.column_stack((abs_xs, abs_ys)).astype(float)
            eigenvalues = np.linalg.eigvalsh(np.cov(coords.T)) if len(coords) >= 3 else np.array([0.0, 0.0])
            compactness = float(eigenvalues[0] / max(1e-6, eigenvalues[-1]))
            # A 2px connector has a distance-transform radius near 2px. A
            # stronger core helps, but does not establish marker shape by
            # itself: a thick line can also have a large distance-transform
            # radius. Require compact geometry for both core paths so a real
            # marker drawn over a line can pass while a thick connector cannot.
            core_threshold = max(2.0, 0.42 * radius)
            compact_shape = aspect <= 2.5 and compactness >= 0.12
            strong_marker_core = core > max(core_threshold, 0.55 * radius) and compact_shape
            compact_core = core > core_threshold and compact_shape
            if strong_marker_core or compact_core:
                candidates.append((nearest, core, area, bw, bh, abs_xs, abs_ys, "native_visible"))
                continue
            # A thin open glyph can lack a filled core. Require a near-circular
            # arc around the detected centre; straight connectors have too few
            # angular bins or inconsistent radii.
            distances = np.hypot(abs_xs - cx, abs_ys - cy)
            annulus = (distances >= 0.55 * radius) & (distances <= 1.45 * radius)
            if annulus.sum() < 8:
                continue
            arc_distances = distances[annulus]
            angles = np.arctan2(abs_ys[annulus] - cy, abs_xs[annulus] - cx)
            bins = np.unique(((angles + np.pi) / (2 * np.pi) * 24).astype(int))
            radial_error = float(np.median(np.abs(arc_distances - np.median(arc_distances))))
            coverage = len(bins) / 24.0
            median_radius = float(np.median(arc_distances))
            if coverage >= 0.35 and radial_error <= 0.22 * radius and 0.65 * radius <= median_radius <= 1.35 * radius:
                kind = "native_visible" if coverage >= 0.6 else "partial_marker"
                candidates.append((nearest, core, area, bw, bh, abs_xs, abs_ys, kind))
        if candidates:
            nearest, core, area, bw, bh, _, _, kind = min(candidates, key=lambda item: item[0])
            uncertainty = max(1.0, min(radius, radius - core)) if core >= max(1.25, 0.30 * radius) else max(1.5, 0.3 * radius)
            if kind == "partial_marker":
                uncertainty = max(uncertainty, 0.35 * radius)
            return {"evidence_kind": kind, "uncertainty_px": round(float(uncertainty), 2),
                    "core_px": round(core, 2), "component_area_px": int(area),
                    "component_bbox_px": [int(bw), int(bh)], "tol_lab": float(tol)}
    return None


def undercoverage_requested(check: dict | None) -> bool:
    """Return true only when Agent 02 explicitly mentions a coverage gap."""
    if not isinstance(check, dict):
        return False
    for key in ("undercoverage_requested", "request_missing_strip_recovery"):
        if check.get(key) is True:
            return True
    text = json.dumps(
        {key: check.get(key) for key in ("issues", "notes", "reasoning")},
        ensure_ascii=False,
    ).lower()
    return any(word in text for word in _REQUEST_WORDS)


def recover_confirmed_slots(
    extraction: dict,
    spec: dict,
    check: dict | None = None,
    *,
    enabled: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Return ``(operations, audit)`` for native-confirmed unresolved slots.

    ``operations`` use the same mechanical shape as an Agent 03 ``add`` and
    can therefore be passed through :func:`stage_edits.apply_agent_edits`.
    ``audit`` records every skipped slot, making the default no-op observable.
    No operation is emitted unless all gates pass and the current
    ``confirm_marker_near`` finding native color and a separate geometry check
    confirming a compact marker core or marker-shaped arc.
    """
    slots = list((extraction.get("unresolved_slots") or {}).get("slots", []))
    audit: list[dict] = []
    if not enabled:
        return [], [{"status": "disabled", "detail": "missing-strip recovery is disabled"}]
    setup_ok, setup_detail = _verified_setup(check, extraction, spec)
    if not setup_ok:
        return [], [{"status": "gated", "detail": setup_detail}]
    if not undercoverage_requested(check):
        return [], [{"status": "gated", "detail": "no explicit undercoverage request"}]

    bbox = spec.get("panel_bbox") or [0, 0, 0, 0]
    width = float(bbox[2] - bbox[0])
    height = float(bbox[3] - bbox[1])
    if width <= 0 or height <= 0:
        return [], [{"status": "gated", "detail": "invalid panel dimensions"}]

    by_label = {
        str(series.get("label")): series
        for series in extraction.get("series", [])
        if series.get("extract") is True and str(series.get("gas", "")).upper() == "CO2"
    }
    operations: list[dict] = []
    occupied: list[tuple[float, float, float]] = []
    for series in by_label.values():
        radius = max(2.0, float(series.get("marker_radius_px") or 4.0))
        for point in series.get("points", []):
            px = point.get("px") or []
            if len(px) == 2:
                occupied.append((float(px[0]), float(px[1]), radius))

    for index, slot in enumerate(slots, start=1):
        label = str(slot.get("series_label") or "")
        series = by_label.get(label)
        px, py = slot.get("column_px"), slot.get("predicted_y_px")
        record = {"slot_index": index, "series_label": label, "status": "skipped"}
        if series is None or not isinstance(px, (int, float)) or not isinstance(py, (int, float)):
            record["detail"] = "no eligible CO2 series or predicted pixel"
            audit.append(record)
            continue
        color_lab = series.get("color_lab")
        if not isinstance(color_lab, (list, tuple)) or len(color_lab) != 3:
            record["detail"] = "series has no calibrated native color"
            audit.append(record)
            continue
        radius = max(2.0, float(series.get("marker_radius_px") or 4.0))
        if any(math.hypot(px - ox, py - oy) < 0.9 * max(radius, rr) for ox, oy, rr in occupied):
            record["detail"] = "duplicate of an existing point"
            audit.append(record)
            continue
        confirmed = extract.confirm_marker_near(
            spec,
            color_lab,
            radius,
            float(px),
            float(py),
            search=max(8.0, 1.5 * radius),
        )
        if confirmed is None:
            record["detail"] = "no native marker pixels confirmed"
            audit.append(record)
            continue
        cx, cy, confidence = confirmed
        shape_support = _native_shape_support(
            spec, color_lab, radius, (cx, cy),
            tolerance=float(series.get("color_tolerance_lab") or series.get("color_tolerance") or 42.0),
        )
        if shape_support is None:
            record["detail"] = "native color hit was line-like or lacked marker-shaped support"
            record["source_pixel"] = [cx, cy]
            audit.append(record)
            continue
        evidence_kind = shape_support["evidence_kind"]
        operations.append(
            {
                "operation_id": f"missing-strip-{index:04d}",
                "action": "add",
                "point_id": None,
                "series_label": label,
                "x_norm": float(cx) / width,
                "y_norm": float(cy) / height,
                "overlap": True,
                "reason": "native marker-shaped evidence confirmed at an unresolved crowded-strip slot",
                "source_evidence": (
                    f"Native source panel {evidence_kind} at panel pixel ({cx:.2f}, {cy:.2f}); "
                    f"color hit passed marker-core/arc geometry (core {shape_support['core_px']:.2f}px, "
                    f"{shape_support['component_area_px']} color pixels)."
                ),
                "evidence_kind": evidence_kind,
                "uncertainty_px": shape_support["uncertainty_px"],
                "evidence_ref": "panel.png",
            }
        )
        occupied.append((float(cx), float(cy), radius))
        audit.append({**record, "status": "confirmed", "source_pixel": [cx, cy], "confidence": confidence,
                      "shape_support": shape_support})
    return operations, audit


def recover_missing_strips(extraction: dict, spec: dict, check: dict | None = None, *, enabled: bool = False):
    """Compatibility alias for callers using the former v3.1 name."""
    return recover_confirmed_slots(extraction, spec, check, enabled=enabled)

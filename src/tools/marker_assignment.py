"""Geometry-first marker evidence and deterministic series assignment.

Color is useful evidence, but it is not a safe ownership gate when nearby
legend colors overlap in Lab space.  This module deliberately keeps native
marker detection independent from a particular series color and exposes the
same feature calculator to the regular and dense-region extractors.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def ambiguity_groups(colors, series, distance_lab=28.0):
    """Return connected groups of series whose nearest legend colors overlap.

    The graph includes excluded series.  A group is connected rather than just
    a list of adjacent pairs so a three-color chain is treated as one joint
    decision.  Results are sorted by declaration order for deterministic
    diagnostics.
    """
    n = len(colors)
    if n < 2:
        return []
    threshold = float(distance_lab)
    parent = list(range(n))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    distances = {}
    for left in range(n):
        for right in range(left + 1, n):
            distance = float(np.linalg.norm(np.asarray(colors[left]) - np.asarray(colors[right])))
            distances[(left, right)] = distance
            if distance <= threshold:
                union(left, right)

    groups = {}
    for index in range(n):
        groups.setdefault(find(index), []).append(index)
    output = []
    for indices in sorted((sorted(value) for value in groups.values() if len(value) > 1)):
        nearest = {}
        pair_distances = {}
        for index in indices:
            candidates = {
                other: distances[tuple(sorted((index, other)))]
                for other in indices
                if other != index
            }
            nearest[index] = min(candidates.items(), key=lambda item: (item[1], item[0]))
            for other, distance in candidates.items():
                pair_distances[f"{index}:{other}"] = round(float(distance), 3)
        output.append(
            {
                "group_id": len(output),
                "series_indices": indices,
                "series_labels": [str(series[index].get("label", index)) for index in indices],
                "nearest_legend_color": {
                    str(index): {"series_index": int(other), "distance_lab": round(float(distance), 3)}
                    for index, (other, distance) in sorted(nearest.items())
                },
                "pair_distances_lab": dict(sorted(pair_distances.items())),
                "threshold_lab": threshold,
            }
        )
    return output


def marker_fill(series):
    """Normalise the declared filled/open marker state."""
    value = series.get("marker_fill")
    if value is None:
        value = series.get("fill")
    value = str(value or "filled").strip().lower().replace("-", "_")
    return "open" if value in {"open", "hollow", "outline", "none"} else "filled"


def native_ink_mask(grey, lab, plot_mask):
    """Build a color-independent native-ink mask inside the calibrated plot."""
    grey = np.asarray(grey)
    lab = np.asarray(lab)
    # JPEG and anti-aliased light colors often have little saturation.  Local
    # contrast keeps them while the edge dilation supplies open-marker rings.
    blurred = cv2.GaussianBlur(grey, (0, 0), 1.2)
    local_contrast = cv2.absdiff(grey, blurred)
    dark = grey < 245
    chromatic = (np.linalg.norm(lab[..., 1:] - 128.0, axis=2) > 10.0) & (local_contrast > 1)
    edges = cv2.Canny(grey, 25, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    mask = (dark | chromatic | edges).astype(np.uint8) * 255
    if plot_mask is not None:
        mask = cv2.bitwise_and(mask, plot_mask)
    return mask


def _local_geometry(mask, x, y, radius):
    height, width = mask.shape[:2]
    r = max(3, int(round(radius)))
    x0, x1 = max(0, int(round(x)) - r), min(width, int(round(x)) + r + 1)
    y0, y1 = max(0, int(round(y)) - r), min(height, int(round(y)) + r + 1)
    patch = mask[y0:y1, x0:x1] > 0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    distance = np.hypot(xx - float(x), yy - float(y))
    center = patch[distance <= max(1.5, 0.35 * radius)]
    ring = patch[(distance >= max(1.0, 0.55 * radius)) & (distance <= 1.25 * radius)]
    return patch, float(center.mean()) if center.size else 0.0, float(ring.mean()) if ring.size else 0.0


def native_marker_candidates(native_mask, radius, roi=None):
    """Find marker-sized native cores/outlines without a color ownership gate.

    The detector accepts both filled cores and open rings.  Thin connector
    lines have neither sufficient core distance nor ring support and are
    discarded.  Returned coordinates are observations from native pixels,
    never trajectory predictions.
    """
    mask = (native_mask > 0).astype(np.uint8)
    if roi is not None:
        xa, ya, xb, yb = map(int, roi)
        local = np.zeros_like(mask)
        local[max(0, ya):min(mask.shape[0], yb), max(0, xa):min(mask.shape[1], xb)] = 1
        mask &= local
    if not mask.any():
        return []
    radius = max(2.5, float(radius))
    dt = cv2.distanceTransform(mask * 255, cv2.DIST_L2, 5)
    peak_threshold = max(1.35, 0.22 * radius)
    maxima = cv2.dilate(dt, np.ones((max(3, int(round(radius))),) * 2, np.uint8))
    locations = np.argwhere((dt >= peak_threshold) & (dt >= maxima - 1e-6))
    # Long chart lines can create many shallow local maxima.  Keep the
    # strongest native peaks before the small per-candidate feature loop.
    if len(locations) > 500:
        strengths = dt[locations[:, 0], locations[:, 1]]
        locations = locations[np.argsort(strengths)[-500:]]
    candidates = []
    for y, x in locations:
        patch, center_density, ring_density = _local_geometry(mask, x, y, radius)
        # Open markers rely on ring support; filled markers rely on their core.
        is_filled = center_density >= 0.22 and dt[y, x] >= 0.28 * radius
        is_open = ring_density >= 0.12 and center_density <= 0.78
        if not (is_filled or is_open):
            continue
        support = max(center_density, ring_density)
        candidates.append(
            {
                "px": [float(x), float(y)],
                "confidence": float(min(0.82, 0.42 + 0.45 * support)),
                "geometry": {
                    "core_radius": round(float(dt[y, x]), 3),
                    "center_density": round(center_density, 3),
                    "ring_density": round(ring_density, 3),
                    "fill_state": "filled" if is_filled and center_density >= ring_density else "open",
                },
                "source": "native_geometry",
            }
        )
    candidates.sort(key=lambda item: (-item["geometry"]["core_radius"], item["px"][0], item["px"][1]))
    kept = []
    for candidate in candidates:
        if any(np.hypot(candidate["px"][0] - other["px"][0], candidate["px"][1] - other["px"][1]) < 0.65 * radius for other in kept):
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda item: (item["px"][0], item["px"][1]))


def _template_score(native_mask, x, y, radius, template):
    if template is None or np.asarray(template).size == 0:
        return 0.5
    r = max(3, int(round(radius * 1.35)))
    patch = native_mask[max(0, int(round(y)) - r):int(round(y)) + r + 1,
                       max(0, int(round(x)) - r):int(round(x)) + r + 1]
    if patch.size == 0:
        return 0.0
    target = cv2.resize((patch > 0).astype(np.uint8), (32, 32), interpolation=cv2.INTER_AREA) > 0.25
    reference = cv2.resize((np.asarray(template) > 0).astype(np.uint8), (32, 32), interpolation=cv2.INTER_AREA) > 0.25
    intersection = float(np.logical_and(target, reference).sum())
    union = float(np.logical_or(target, reference).sum())
    return intersection / max(1.0, union)


def _color_score(lab, x, y, radius, color, tolerance):
    _, center_density, ring_density = _local_geometry(
        np.linalg.norm(lab - np.asarray(color)[None, None, :], axis=2) < max(3.0, float(tolerance)),
        x, y, radius,
    )
    # The likelihood uses the native Lab sample, but never rejects a candidate.
    h = max(2, int(round(0.38 * radius)))
    ix, iy = int(round(x)), int(round(y))
    patch = lab[max(0, iy - h):iy + h + 1, max(0, ix - h):ix + h + 1].reshape(-1, 3)
    if patch.size == 0:
        return 0.0, float("inf")
    distance = float(np.median(np.linalg.norm(patch - np.asarray(color), axis=1)))
    return float(math.exp(-distance / max(5.0, 1.6 * float(tolerance)))), distance


def _fill_score(grey, x, y, radius, expected):
    native = (grey < 245).astype(np.uint8)
    _, center, ring = _local_geometry(native, x, y, radius)
    if expected == "open":
        return float(0.65 * ring + 0.35 * (1.0 - center))
    return float(0.7 * center + 0.3 * ring)


def _line_score(native_mask, x, y, radius, points):
    neighbours = sorted(
        (point for point in points if abs(float(point[0]) - x) > 0.8 * radius),
        key=lambda point: np.hypot(float(point[0]) - x, float(point[1]) - y),
    )[:4]
    if not neighbours:
        return 0.5
    scores = []
    for point in neighbours:
        px, py = float(point[0]), float(point[1])
        if abs(px - x) < 0.5 * radius:
            continue
        count = max(5, int(np.hypot(px - x, py - y) / 2))
        samples = [native_mask[int(round(y + t * (py - y))), int(round(x + t * (px - x)))] > 0
                   for t in np.linspace(0.15, 0.85, count)
                   if 0 <= int(round(y + t * (py - y))) < native_mask.shape[0]
                   and 0 <= int(round(x + t * (px - x))) < native_mask.shape[1]]
        scores.append(float(np.mean(samples)) if samples else 0.0)
    return max(scores, default=0.5)


def _trajectory_score(candidate, points, fx=None, fy=None):
    if fx is None or fy is None:
        return 0.5
    x, y = candidate["px"]
    left = [p for p in points if p[0] < x]
    right = [p for p in points if p[0] > x]
    if not left or not right:
        return 0.5
    before, after = max(left, key=lambda p: p[0]), min(right, key=lambda p: p[0])
    if after[0] <= before[0]:
        return 0.5
    expected = before[1] + (after[1] - before[1]) * (x - before[0]) / (after[0] - before[0])
    # This is assignment evidence only.  It does not create a coordinate.
    residual_px = abs(y - expected)
    return float(math.exp(-residual_px / max(2.0, 1.5 * (abs(after[0] - before[0]) ** 0.35))))


def score_candidate(candidate, series, lab, grey, native_mask, radius, raw_points, fx=None, fy=None, weights=None):
    """Return component and weighted scores for one native candidate/series pair."""
    weights = dict(weights or {})
    shape_weight = float(weights.get("shape", 0.40))
    color_weight = float(weights.get("color", 0.20))
    trajectory_weight = float(weights.get("trajectory", 0.20))
    line_weight = max(0.0, 1.0 - shape_weight - color_weight - trajectory_weight)
    x, y = candidate["px"]
    shape = _template_score(native_mask, x, y, radius, series.get("shape"))
    fill = _fill_score(grey, x, y, radius, marker_fill(series.get("spec", series)))
    color, color_distance = _color_score(
        lab, x, y, radius, series["col"], series.get("tol", 28.0)
    )
    line = _line_score(native_mask, x, y, radius, raw_points)
    trajectory = _trajectory_score(candidate, raw_points, fx=fx, fy=fy)
    shape_and_fill = 0.75 * shape + 0.25 * fill
    total = (
        shape_weight * shape_and_fill
        + color_weight * color
        + trajectory_weight * trajectory
        + line_weight * line
    )
    return {
        "total": round(float(total), 6),
        "shape": round(float(shape), 6),
        "fill_open": round(float(fill), 6),
        "color": round(float(color), 6),
        "color_distance_lab": round(float(color_distance), 4),
        "native_line": round(float(line), 6),
        "trajectory": round(float(trajectory), 6),
        "weights": {
            "shape": shape_weight,
            "color": color_weight,
            "trajectory": trajectory_weight,
            "native_line": line_weight,
        },
    }

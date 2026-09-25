"""Recover overlapping open circles from native ring templates, without a grid.

An isolated source marker supplies the template when available. Peaks must have
ring ink and a distinct maximum in every direction; a uniform fused stripe is
not evidence for an arbitrary sequence of experimental markers.
"""

import cv2
import numpy as np
from scipy.ndimage import maximum_filter


def ring_template(mask, points, radius):
    half = max(5, int(np.ceil(1.6 * radius)))
    templates = []
    xy = np.array([p[:2] for p in points], dtype=float).reshape(-1, 2)
    for index, (x, y) in enumerate(xy):
        distances = np.linalg.norm(xy - (x, y), axis=1)
        distances[index] = np.inf
        if distances.min() < 3 * radius:
            continue
        cx, cy = int(round(x)), int(round(y))
        if min(cx, cy) < half or cx + half >= mask.shape[1] or cy + half >= mask.shape[0]:
            continue
        patch = mask[cy-half:cy+half+1, cx-half:cx+half+1].copy()
        # Fill enclosed holes only to measure the outer shape, then remove thin
        # connectors. The template itself retains the source's hollow interior.
        flood = cv2.copyMakeBorder(patch, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        cv2.floodFill(flood, None, (0, 0), 1)
        filled = patch | (1 - flood[1:-1, 1:-1])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        distance = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
        _, peak, _, center = cv2.minMaxLoc(distance)
        if not 0.5 * radius <= peak <= 1.4 * radius:
            continue
        dx, dy = center[0] - half, center[1] - half
        centered = cv2.warpAffine(patch.astype(np.float32), np.float32([[1, 0, -dx], [0, 1, -dy]]),
                                 (2 * half + 1, 2 * half + 1))
        yy, xx = np.mgrid[-half:half+1, -half:half+1]
        centered[np.hypot(xx, yy) > 1.1 * peak] = 0
        if centered[half-1:half+2, half-1:half+2].mean() > 0.5:
            continue
        templates.append(centered)
    if templates:
        return np.median(templates, axis=0).astype(np.float32), "isolated_native_rings"
    template = np.zeros((2 * half + 1, 2 * half + 1), np.uint8)
    cv2.circle(template, (half, half), max(2, int(round(radius))), 1, max(1, int(round(radius / 3))))
    return template.astype(np.float32), "declared_circle_geometry"


def detect(mask, points, roi, radius):
    """Return validated 2-D ring centers and diagnostics; never sample a curve."""
    mask = (mask > 0).astype(np.uint8)
    template, source = ring_template(mask, points, radius)
    half = template.shape[0] // 2
    x0, y0, x1, y1 = (int(round(v)) for v in roi)
    x0, x1 = max(0, x0), min(mask.shape[1], x1)
    y0, y1 = max(0, y0), min(mask.shape[0], y1)
    roi_ink = int(mask[y0:y1, x0:x1].sum()) if x0 < x1 and y0 < y1 else 0
    if min(mask.shape) < template.shape[0]:
        return [], {"template_source": source, "searches": 0, "validated_centers": 0,
                    "roi_ink_pixels": roi_ink, "rejected": {"too_small": 1}}
    response = cv2.matchTemplate(mask.astype(np.float32), template, cv2.TM_CCOEFF_NORMED)
    coverage = cv2.matchTemplate(mask.astype(np.float32), template, cv2.TM_CCORR) / max(1, template.sum())
    x0, y0, x1, y1 = map(float, roi)
    ys, xs = np.where((response >= 0.32) & (response == maximum_filter(response, size=3)))
    step = max(2, int(round(0.45 * radius)))
    candidates = []
    rejected = {"outside_roi": 0, "boundary": 0, "coverage": 0, "flat_response": 0, "solid_core": 0}
    for y, x in zip(ys, xs):
        cx, cy = x + half, y + half
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            rejected["outside_roi"] += 1
            continue
        if min(x, y) < step or x + step >= response.shape[1] or y + step >= response.shape[0]:
            rejected["boundary"] += 1
            continue
        if coverage[y, x] < 0.75:
            rejected["coverage"] += 1
            continue
        core_radius = max(1, int(round(0.35 * radius)))
        yy, xx = np.mgrid[-core_radius:core_radius + 1, -core_radius:core_radius + 1]
        core = xx * xx + yy * yy <= core_radius * core_radius
        centre_ink = mask[cy - core_radius:cy + core_radius + 1, cx - core_radius:cx + core_radius + 1]
        if centre_ink.shape == core.shape and float(centre_ink[core].mean()) > 0.96:
            # An open ring may be crossed by a neighbour, but a completely
            # filled core is a band/filled glyph, never evidence for this
            # open-circle detector.
            rejected["solid_core"] += 1
            continue
        peak = float(response[y, x])
        # Flat ridges (fused strips) must not turn into a string of markers.
        prominence = min(peak - (float(response[y+dy, x+dx]) + float(response[y-dy, x-dx])) / 2
                         for dx, dy in ((step, 0), (0, step), (step, step), (step, -step)))
        if prominence < 0.035:
            rejected["flat_response"] += 1
            continue
        offsets = []
        for dx, dy in ((1, 0), (0, 1)):
            lo, hi = float(response[y-dy, x-dx]), float(response[y+dy, x+dx])
            curvature = lo - 2 * peak + hi
            offsets.append(float(np.clip(0.5 * (lo - hi) / curvature, -0.5, 0.5)) if curvature < -1e-6 else 0)
        candidates.append((cx + offsets[0], cy + offsets[1], peak))
    kept = []
    for cx, cy, score in sorted(candidates, key=lambda p: -p[2]):
        if not any(np.hypot(cx - p[0], cy - p[1]) < max(2.0, 0.6 * radius) for p in kept):
            kept.append((cx, cy, score))
    return sorted(kept), {
        "template_source": source,
        "searches": 1,
        "validated_centers": int(len(kept)),
        "roi_ink_pixels": roi_ink,
        "rejected": {key: int(value) for key, value in rejected.items()},
    }


def merge_2d(existing, candidates, radius):
    """Match ring proposals to existing points in 2-D, one proposal per marker.

    A nearby x coordinate is not evidence that a vertically stacked marker has
    already been found.  Conversely, a template peak beside an existing centre
    is a duplicate, not a reason to move the measured point.  This function is
    deliberately coordinate-only: callers retain the source-supported centre
    and attach their own confidence/provenance.
    """
    threshold = max(1.5, 0.75 * float(radius))
    available = set(range(len(existing)))
    matched, additions = [], []
    for candidate in sorted(candidates, key=lambda item: -float(item[2])):
        x, y, score = map(float, candidate)
        nearby = sorted(
            ((float(np.hypot(x - existing[index][0], y - existing[index][1])), index)
             for index in available),
            key=lambda item: item[0],
        )
        if nearby and nearby[0][0] <= threshold:
            distance, index = nearby[0]
            available.remove(index)
            matched.append({"candidate": [x, y], "existing_index": int(index), "distance_px": distance})
        else:
            additions.append((x, y, score))
    return additions, matched

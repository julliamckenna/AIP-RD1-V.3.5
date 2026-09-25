"""Separate same-color series by where their curves run (y position).

Color twins - e.g. CO2 273K and N2 313K both black, or CO2 circles and CH4 triangles in the
same red - can not always be told apart by marker shape at print resolution, but their curves
usually run far apart.  Each series carries a rough curve from agent 01 (``y_anchors``: its y
at 25/50/75/100 % of the x range), optionally a dense curve from agent 02 (``curve_points``)
and hard boxes from agent 02 (``y_bands``).

``apply`` moves a detected point to the twin whose curve explains it much better, or drops it
when it violates its own band and no twin claims it.  Below 25 % of the x range the rough
anchors are not reliable (steep low-pressure rise), so points there are only checked against
explicit bands - agent 02's ``curve_points`` hold over their whole x range.  Every decision is returned for the audit.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.models import naming

QUARTERS = (0.25, 0.5, 0.75, 1.0)


def anchor_curve(series_spec, x_min, x_max):
    """(y(x), x_from, x_to) - where the curve is reliable - or None when the series has no curve.

    Agent 02's ``curve_points`` (dense readings along the isotherm) win and are reliable over their
    whole x range; agent 01's four ``y_anchors`` are too coarse below 25 % of the x range.
    """
    points = sorted((float(p["x"]), float(p["y"])) for p in series_spec.get("curve_points") or [])
    if len(points) >= 2:
        xs, ys = zip(*points)
        return (lambda x: float(np.interp(x, xs, ys))), xs[0], xs[-1]
    anchors = series_spec.get("y_anchors") or {}
    ys = [anchors.get(key) for key in ("p25", "p50", "p75", "p100")]
    if any(value is None for value in ys) or x_max <= x_min:
        return None
    xs = [x_min + q * (x_max - x_min) for q in QUARTERS]
    return (lambda x: float(np.interp(x, xs, ys))), x_min + 0.25 * (x_max - x_min), float("inf")


def in_band(series_spec, x, y, margin=0.0):
    """True / False against y-bands, allowing a documented pixel-scale margin.

    Bands are a coarse reading supplied by an agent whereas a marker centre is
    measured in pixels.  A centre within one marker-radius of a band boundary
    is retained and reported by ``apply`` instead of being discarded merely
    because the two coordinate systems round differently.
    """
    covering = [band for band in series_spec.get("y_bands") or [] if band["x_from"] <= x <= band["x_to"]]
    if not covering:
        return None
    return any(float(band["y_min"]) - margin <= y <= float(band["y_max"]) + margin for band in covering)


def _hex_to_lab(value):
    text = str(value or "").lstrip("#")
    if len(text) != 6:
        return None
    rgb = [int(text[k:k + 2], 16) for k in (0, 2, 4)]
    return cv2.cvtColor(np.uint8([[rgb[::-1]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def apply(all_series, spec, colors, fx, fy, *, twin_distance=30.0, ratio=0.5, min_gap_fraction=0.06):
    """Reassign / drop points of color twins by curve position. Returns an audit list."""
    ticks_x = spec["x"].get("ticks") or []
    ticks_y = spec["y"].get("ticks") or []
    if len(ticks_x) < 2 or len(ticks_y) < 2:
        return []
    x_min, x_max = float(min(ticks_x)), float(max(ticks_x))
    y_span = float(max(ticks_y) - min(ticks_y)) or 1.0
    specs = spec["series"]
    curves = [anchor_curve(s, x_min, x_max) for s in specs]
    names = {naming.series_label(s.get("label")).lower(): i for i, s in enumerate(specs)}
    declared_lab = [_hex_to_lab(s.get("color_hex")) for s in specs]  # measured open rings are paler

    def twins_of(i):
        declared = {names.get(naming.series_label(label).lower()) for label in specs[i].get("separate_from") or []} - {None}
        by_color = {j for j in range(len(specs)) if j != i and np.linalg.norm(colors[j] - colors[i]) < twin_distance}
        by_declared = {j for j in range(len(specs)) if j != i and declared_lab[i] is not None
                       and declared_lab[j] is not None and np.linalg.norm(declared_lab[j] - declared_lab[i]) < twin_distance}
        return declared | by_color | by_declared

    by_idx = {S["idx"]: S for S in all_series}
    moves, audit = [], []
    for S in all_series:
        i = S["idx"]
        twins = [j for j in twins_of(i) if j in by_idx]
        if not twins and not specs[i].get("y_bands"):
            continue
        keep = []
        for point in S["pts"]:
            x, y = float(fx(point[0])), float(fy(point[1]))
            radius = max(1.0, float(S.get("r") or 1.0))
            # Convert one marker-radius in pixel space to data units.  The
            # calculation is local so it remains correct for log axes.
            band_margin = 1.25 * max(abs(float(fy(point[1] + radius)) - y),
                                     abs(float(fy(point[1] - radius)) - y))
            own_band = in_band(specs[i], x, y, band_margin)

            def distance(curve):
                if curve is None or not curve[1] <= x <= curve[2]:
                    return None
                return abs(y - curve[0](x))

            own = distance(curves[i])
            best, best_d = None, None
            for j in twins:
                band = in_band(specs[j], x, y, band_margin)
                if band is False:
                    continue
                d = distance(curves[j])
                if band is True and d is None:
                    d = 0.0
                if d is not None and (best_d is None or d < best_d):
                    best, best_d = j, d
            move = False
            if own_band is False:
                move = True
            elif own is not None and best_d is not None:
                move = best_d < ratio * own and own > min_gap_fraction * y_span
            if not move:
                keep.append(point)
                raw_band = in_band(specs[i], x, y)
                if raw_band is False and own_band is True:
                    audit.append({"from": specs[i].get("label"), "to": specs[i].get("label"),
                                  "x": round(x, 4), "y": round(y, 4),
                                  "reason": "within_pixel_derived_y_band_margin",
                                  "y_margin": round(float(band_margin), 4)})
                continue
            record = {"from": specs[i].get("label"), "x": round(x, 4), "y": round(y, 4),
                      "reason": "outside own y_bands" if own_band is False else "closer to twin's curve"}
            if best is not None and (own_band is False or best_d is not None):
                moves.append((best, point))
                record["to"] = specs[best].get("label")
            else:
                record["to"] = None  # dropped: no series explains it
            audit.append(record)
        S["pts"] = keep
    for j, point in moves:
        by_idx[j]["pts"].append(point)
        by_idx[j]["pts"].sort()
    return audit

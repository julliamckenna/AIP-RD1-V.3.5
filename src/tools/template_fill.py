"""Fill unresolved slots in crowded strips with the series' own marker template.

The dense-region resolver predicts where each series should have a marker in a crowded strip
(between that series' real markers, at a shared pressure column) but only accepts a slot when a
clean marker core is visible.  When markers pile up the shapes are no longer identifiable, so
this module places the series template - the legend glyph measured by legend_markers (outline,
size, open/filled) - on those slots:

  template_fit   the template is slid over a small window around the prediction and the best
                 position is kept when enough of the series' color lies under it.  Pixel-backed:
                 a row with evidence_type=template_fit, low confidence, overlap flagged.
  template_fill  nothing fits (the marker is hidden under others): the template is placed at the
                 predicted position.  Not pixel-backed: is_inferred=true, evidence_type=template_fill,
                 always routed to human review.  Off with extract.template_fill_unresolved: false.

Only slots bracketed by real markers of the same series are filled (never extrapolation).
A marker belongs to one series: a template never lands on a marker another series already
holds, and it is only placed when its own series explains the pixels clearly better than
every other series' color and shape at that position (``margin``).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.tools.legend_markers import ink_mask

CIRCLE = [[math.cos(t), math.sin(t)] for t in np.linspace(0, 2 * math.pi, 16, endpoint=False)]


def template_mask(glyph: dict | None, radius: float) -> np.ndarray:
    """Binary mask of the series marker at ``radius`` (equal-area radius), centred in its array."""
    polygon = (glyph or {}).get("polygon") or CIRCLE
    half = int(math.ceil(2.2 * radius)) + 1
    mask = np.zeros((2 * half + 1, 2 * half + 1), np.uint8)
    pts = np.round((np.array(polygon, np.float32) * radius + half) * 16).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1, cv2.LINE_8, shift=4)
    if (glyph or {}).get("fill") == "open":
        stroke = max(1, int(round(0.3 * radius)))
        mask = mask - cv2.erode(mask, np.ones((2 * stroke + 1, 2 * stroke + 1), np.uint8))
    return mask


def fit(crop_bgr, color_lab, background_lab, glyph, radius, centre, taken, window=(1.2, 2.0)):
    """Best template position near ``centre``: (x, y, coverage) or None.

    ``coverage`` = share of template pixels carrying the series color.  Positions closer than
    0.8 radius to an already-detected marker of the same series (``taken``) are skipped.
    """
    template = template_mask(glyph, radius)
    th, tw = template.shape
    half_w, half_h = int(window[0] * radius) + tw // 2, int(window[1] * radius) + th // 2
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    x0, y0 = max(0, cx - half_w), max(0, cy - half_h)
    x1, y1 = min(crop_bgr.shape[1], cx + half_w + 1), min(crop_bgr.shape[0], cy + half_h + 1)
    area = crop_bgr[y0:y1, x0:x1]
    if area.shape[0] < th or area.shape[1] < tw:
        return None
    ink = ink_mask(area, color_lab, background_lab=background_lab).astype(np.float32)
    coverage = cv2.matchTemplate(ink, template.astype(np.float32), cv2.TM_CCORR) / max(float(template.sum()), 1.0)
    for tx, ty in taken:  # never re-use an existing marker of this series
        gx, gy = int(round(tx - x0 - tw // 2)), int(round(ty - y0 - th // 2))
        r = int(math.ceil(0.8 * radius))
        coverage[max(0, gy - r):max(0, gy + r + 1), max(0, gx - r):max(0, gx + r + 1)] = 0
    _, best, _, (bx, by) = cv2.minMaxLoc(coverage)
    return x0 + bx + tw // 2, y0 + by + th // 2, float(best)


def coverage_at(crop_bgr, color_lab, background_lab, glyph, radius, centre) -> float:
    """Share of the series template, centred at ``centre``, that carries the series color."""
    template = template_mask(glyph, radius)
    th, tw = template.shape
    x0, y0 = int(round(centre[0])) - tw // 2, int(round(centre[1])) - th // 2
    if x0 < 0 or y0 < 0 or x0 + tw > crop_bgr.shape[1] or y0 + th > crop_bgr.shape[0]:
        return 0.0
    ink = ink_mask(crop_bgr[y0:y0 + th, x0:x0 + tw], color_lab, background_lab=background_lab)
    return float((ink * template).sum()) / max(float(template.sum()), 1.0)


def fill_slots(crop_bgr, slots, all_series, glyphs, background_lab, *, min_coverage=0.55, fill_unresolved=True,
               margin=0.15):
    """Place templates on unresolved dense slots.

    Returns ({series_index: [(x, y, confidence, source)]}, audit list).  Each slot handled is
    updated in place with ``state`` template_fit / template_fill and the placed position.
    """
    by_idx = {series["idx"]: series for series in all_series}
    placed, audit = {}, []
    for slot in slots:
        series = by_idx.get(slot.get("series_index"))
        predicted = slot.get("predicted_y_px")
        if series is None or predicted is None or slot.get("column_px") is None:
            continue  # no bracketed prediction: stays unresolved
        mode = str((series.get("spec") or {}).get("template_fill") or "on")  # agent 02: on / fit_only / off
        if mode == "off":
            continue
        radius = max(2.0, float(series.get("r") or 4.0))
        glyph = glyphs[series["idx"]] if series["idx"] < len(glyphs) else None
        # every marker already held by ANY series is off limits (one marker = one series)
        taken = [(p[0], p[1]) for other in all_series for p in other.get("pts", [])]
        taken += [(p[0], p[1]) for held in placed.values() for p in held]
        result = fit(crop_bgr, series["col"], background_lab, glyph, radius, (slot["column_px"], predicted), taken)
        rival = 0.0
        if result:
            rival = max((coverage_at(crop_bgr, other["col"], background_lab,
                                     glyphs[other["idx"]] if other["idx"] < len(glyphs) else None,
                                     max(2.0, float(other.get("r") or radius)), result[:2])
                         for other in all_series if other["idx"] != series["idx"]), default=0.0)
            slot["rival_coverage"] = round(rival, 2)
        if result and result[2] >= min_coverage and result[2] >= rival + margin:
            x, y, coverage = result
            placed.setdefault(series["idx"], []).append((x, y, round(0.35 + 0.3 * coverage, 2), "template_fit"))
            slot.update(state="template_fit", template_px=[round(x, 2), round(y, 2)], template_coverage=round(coverage, 2))
        elif fill_unresolved and mode != "fit_only" and not (result and rival >= min_coverage):  # never guess onto another series' marker
            x, y = float(slot["column_px"]), float(predicted)
            placed.setdefault(series["idx"], []).append((x, y, 0.3, "template_fill"))
            slot.update(state="template_fill", template_px=[round(x, 2), round(y, 2)],
                        template_coverage=round(result[2], 2) if result else 0.0)
        else:
            continue
        audit.append(dict(slot))
    return placed, audit

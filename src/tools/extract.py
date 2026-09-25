"""
Python proposal and Agent 03 calculation support for chart data. Pure Python (OpenCV), no AI.

Driven by a chart-spec JSON (what the vision model supplies):
{
  "image": "fig2.png",
  "panel_bbox": [x0, y0, x1, y1],          # rough crop of ONE panel (whole panel incl. labels)
  "x": {"ticks": [0, 2, 4, 6, 8, 10], "scale": "linear", "unit": "bar"},
  "y": {"ticks": [0, 1, 2, 3, 4, 5, 6, 7], "scale": "linear", "unit": "mmol/g"},
  "legend_bbox": [x0, y0, x1, y1],         # in panel-crop coordinates, masked out + used to sample colors
  "mask_bboxes": [[x0, y0, x1, y1], ...],  # optional: insets / annotations to ignore (panel-crop coords)
  "series": [ {"label": "293K", "extract": true, "y_at_xmax": 4.1}, ... ]   # legend order, top to bottom;
                                            # y_at_xmax (optional, rough) = where the series ends at the right edge
}
Everything geometric (frame, tick pixels, calibration, marker centres, confidence) is detected here.
"""

import json
import pathlib

import cv2
import numpy as np
from scipy import ndimage

from src.settings import CFG
from src.tools import dense_circles, legend_markers, region_strategies, series_gate, template_fill
from src.tools import dense_regions
from src.tools import marker_assignment

DARK = 100  # grey threshold for axis ink
COLOR_TOL = float(
    CFG["extract"]["color_tolerance"]
)  # Lab distance tolerance for a series color
CONF_OK = float(CFG["extract"]["confidence_ok"])


# ---------- geometry ----------


def _runs(idx):
    """Group consecutive indices -> centre of each run."""
    if len(idx) == 0:
        return []
    out, s, p = [], idx[0], idx[0]
    for v in idx[1:]:
        if v != p + 1:
            out.append((s + p) / 2.0)
            s = v
        p = v
    out.append((s + p) / 2.0)
    return out


def find_frame(dark):
    """Axis frame from the longest dark lines. Works for a full box and for L-shaped axes:
    left = leftmost long vertical line, bottom = lowest long horizontal line, top/right = extent of those lines."""
    h, w = dark.shape
    rows = np.where(dark.sum(1) > 0.4 * w)[0]
    cols = np.where(dark.sum(0) > 0.4 * h)[0]
    if len(rows) == 0 or len(cols) == 0:
        raise RuntimeError("axis frame not found; tighten panel_bbox")
    left = int(min(_runs(cols)))
    # Adjacent panels may share the same left coordinate. A point intersection
    # alone does not prove that a lower panel's top border is this plot's x axis:
    # require a vertical segment above and a horizontal segment to the right.
    row_centres = [int(round(r)) for r in _runs(rows)]
    connected_rows = []
    support = max(10, min(20, int(0.1 * h)))
    for r in row_centres:
        if r < support:
            continue
        vertical = dark[r-support:r, max(0, left-1):min(w, left+2)].max(1)
        if vertical.mean() < 0.85:
            continue
        horizontal = dark[max(0, r-1):min(h, r+2), :].max(0)
        end = left
        while end < w - 1 and horizontal[end + 1]:
            end += 1
        if end - left >= max(20, int(0.2 * w)):
            connected_rows.append(r)
    if not connected_rows:
        raise RuntimeError(
            "axis frame not found; horizontal and vertical axes do not meet"
        )
    bottom = max(connected_rows)
    # extent of the left axis line -> top ; extent of the bottom axis line -> right
    col = dark[:, max(0, left - 1) : left + 2].max(1)
    ys = np.where(col)[0]
    ys = ys[ys <= bottom]
    top = int(ys.min()) if len(ys) else 0
    # walk down from top until the line is continuous to bottom (skip stray ink above the axis)
    run_top = bottom
    while run_top > 0 and col[run_top - 1]:
        run_top -= 1
    top = run_top
    row = dark[max(0, bottom - 1) : bottom + 2, :].max(0)
    run_right = left
    while run_right < w - 1 and row[run_right + 1]:
        run_right += 1
    right = run_right
    # a boxed frame: the first tall vertical line right of the axis is the frame's right edge
    # (stops the walk from running along a legend box that touches the axis)
    tall = np.where(dark[top:bottom, :].sum(0) > 0.85 * (bottom - top))[0]
    # Insets can contain tall internal edges. A true boxed-frame right edge
    # meets both the top and bottom borders of the main plot.
    tall = [
        c
        for c in tall
        if c > left + 20
        and dark[max(0, top - 2) : top + 3, c].any()
        and dark[max(0, bottom - 2) : bottom + 3, c].any()
    ]
    if tall:
        right = min(right, int(_runs(np.array(tall))[0]))
    # if a real top/right border exists, prefer it
    top_rows = [int(r) for r in _runs(rows) if r < bottom - 10]
    if top_rows:
        top = min(top, top_rows[0]) if abs(top_rows[0] - top) < 10 else top
    return left, bottom, top, right


def find_ticks(dark, left, bottom, top, right, max_len=14):
    """Inward or outward ticks along bottom and left axes; returns (pixel, length) lists."""

    def scan_x(rows):
        band = dark[rows, :]
        cand = np.where(band.sum(0) >= min(3, len(rows)))[0]
        cand = [c for c in cand if left + 3 < c < right - 3]
        ticks = []
        for c in _runs(np.array(cand)):
            c = int(round(c))
            col = dark[:, c]
            length = max(
                col[bottom - max_len : bottom].sum(),
                col[bottom + 1 : bottom + 1 + max_len].sum(),
            )
            ticks.append((c, int(length)))
        return ticks

    def scan_y(cols):
        band = dark[:, cols]
        cand = np.where(band.sum(1) >= min(3, len(cols)))[0]
        cand = [c for c in cand if top + 3 < c < bottom - 3]
        ticks = []
        for c in _runs(np.array(cand)):
            c = int(round(c))
            row = dark[c, :]
            length = max(
                row[left + 1 : left + 1 + max_len].sum(),
                row[left - max_len : left].sum(),
            )
            ticks.append((c, int(length)))
        return ticks

    xt = scan_x(list(range(bottom - 4, bottom))) + scan_x(
        list(range(bottom + 1, bottom + 5))
    )
    yt = scan_y(list(range(left + 1, left + 5))) + scan_y(list(range(left - 4, left)))
    # dedupe by pixel
    xt = sorted({p: l for p, l in xt}.items())
    yt = sorted({p: l for p, l in yt}.items())
    return xt, yt


def tick_anchor_pixel_residual(pixels, values, anchors, scale, extent_px):
    """Return pixel disagreement between a tick candidate and anchor evidence."""
    if not anchors or extent_px is None or len(anchors) < 2:
        return float("inf")
    anchor_values = np.asarray([float(item["value"]) for item in anchors], dtype=float)
    if scale == "log":
        if np.any(anchor_values <= 0) or np.any(np.asarray(values, dtype=float) <= 0):
            return float("inf")
        anchor_values = np.log10(anchor_values)
        fit_values = np.log10(np.asarray(values, dtype=float))
    else:
        fit_values = np.asarray(values, dtype=float)
    anchor_pixels = np.asarray(
        [float(item["pixel_norm"]) * extent_px for item in anchors], dtype=float
    )
    slope, intercept = np.polyfit(anchor_values, anchor_pixels, 1)
    expected_pixels = np.polyval((slope, intercept), fit_values)
    return float(np.max(np.abs(expected_pixels - np.asarray(pixels, dtype=float))))


def major_ticks(
    ticks,
    n_expected,
    values=None,
    scale="linear",
    anchors=None,
    extent_px=None,
    max_fit_residual_px=None,
):
    """Pick the N labelled ticks. Majors are longer than minors, and their pixel positions must be
    an affine image of their values (evenly spaced for linear axes) — that second test is what
    survives JPEG noise, where a minor tick can look as long as a major one."""
    if len(ticks) < n_expected:
        raise RuntimeError(f"found {len(ticks)} ticks, spec expects {n_expected}")
    max_len = max(l for _, l in ticks)
    cand = sorted(p for p, l in ticks if l >= 0.6 * max_len)
    if len(cand) < n_expected:
        cand = sorted(p for p, _ in sorted(ticks, key=lambda t: -t[1])[:n_expected])
    if len(cand) == n_expected:
        return cand
    vals = (
        np.log10(np.array(values, float))
        if (values is not None and scale == "log")
        else (
            np.array(values, float)
            if values is not None
            else np.arange(n_expected, dtype=float)
        )
    )
    from itertools import combinations

    if len(cand) > 16:  # too many combinations: keep the longest
        cand = sorted(
            p
            for p, _ in sorted([t for t in ticks if t[0] in cand], key=lambda t: -t[1])[
                :16
            ]
        )
    scored = []
    for combo in combinations(cand, n_expected):
        a, b = np.polyfit(vals, combo, 1)
        err = np.abs(np.polyval((a, b), vals) - combo).max()
        anchor_err = tick_anchor_pixel_residual(
            combo, values if values is not None else vals, anchors, scale, extent_px
        )
        scored.append((float(err), anchor_err, list(combo)))
    eligible = (
        [item for item in scored if item[0] <= max_fit_residual_px]
        if max_fit_residual_px is not None
        else scored
    )
    if anchors and eligible:
        # Anchors resolve otherwise-equally plausible deterministic fits.  The
        # strict validation below still rejects bad anchor evidence.
        return min(eligible, key=lambda item: (item[1], item[0]))[2]
    return min(scored, key=lambda item: item[0])[2]


def detect_axis_geometry(grey, spec):
    """Detect axes and ticks, retrying with lighter geometry thresholds."""
    last_error = None
    for threshold in dict.fromkeys((DARK, 130, 160, 190)):
        dark = (grey < threshold).astype(np.uint8)
        try:
            left, bottom, top, right = find_frame(dark)
            if right - left < 20 or bottom - top < 20:
                raise RuntimeError(
                    f"degenerate axis frame: left={left}, top={top}, "
                    f"right={right}, bottom={bottom}"
                )
            xt, yt = find_ticks(dark, left, bottom, top, right)
            xt = [tick for tick in xt if tick[0] <= right + 3]
            yt = [tick for tick in yt if tick[0] >= top - 3]
            max_residual = float(CFG["extract"].get("max_tick_fit_residual_px", 2.5))

            def select_ticks(ticks, start, end, axis, vertical, axis_name):
                """Prefer the better fit with or without inferred frame ticks.

                Axis limits can equal the first/last labels even when the
                printed frame extends beyond those labels.  Treat frame ticks
                as candidates rather than forcing them into the calibration.
                """
                values = (
                    sorted(axis["ticks"], reverse=True) if vertical else axis["ticks"]
                )
                scale = axis.get("scale", "linear")
                variants = [list(ticks)]
                augmented = include_boundary_ticks(ticks, start, end, axis, vertical)
                if augmented != variants[0]:
                    variants.append(augmented)
                fitted = []
                errors = []
                anchors = axis.get("tick_anchors")
                extent_px = grey.shape[0] if vertical else grey.shape[1]
                for variant in variants:
                    try:
                        pixels = major_ticks(
                            variant,
                            len(values),
                            values,
                            scale,
                            anchors=anchors,
                            extent_px=extent_px,
                            max_fit_residual_px=max_residual,
                        )
                    except RuntimeError as error:
                        errors.append(error)
                        continue
                    fit_values = (
                        np.log10(np.asarray(values, dtype=float))
                        if scale == "log"
                        else values
                    )
                    slope, intercept = np.polyfit(fit_values, pixels, 1)
                    residual = float(
                        np.abs(
                            np.polyval((slope, intercept), fit_values) - pixels
                        ).max()
                    )
                    anchor_residual = tick_anchor_pixel_residual(
                        pixels, values, anchors, scale, extent_px
                    )
                    fitted.append((residual, anchor_residual, pixels))
                if not fitted:
                    raise errors[-1]
                eligible = [item for item in fitted if item[0] <= max_residual]
                if anchors and eligible:
                    residual, _, pixels = min(
                        eligible, key=lambda item: (item[1], item[0])
                    )
                else:
                    residual, _, pixels = min(fitted, key=lambda item: item[0])
                if residual > max_residual:
                    raise RuntimeError(
                        f"{axis_name}-tick fit residual {residual:.2f}px exceeds "
                        f"{max_residual:.2f}px"
                    )
                return pixels

            XT = select_ticks(xt, left, right, spec["x"], False, "x")
            YT = select_ticks(yt, top, bottom, spec["y"], True, "y")
            return left, bottom, top, right, XT, YT
        except RuntimeError as error:
            last_error = error
    raise RuntimeError(
        f"axis/tick detection failed through grayscale threshold 190: {last_error}"
    ) from last_error


def include_boundary_ticks(ticks, start, end, axis, vertical=False):
    """Include labelled ticks that coincide with a boxed plot's frame.

    Tick scanning deliberately ignores the frame itself.  When an axis limit is
    also a labelled tick (commonly the top y tick), add that known frame
    position back before selecting major ticks.
    """
    ticks = list(ticks)
    values = [float(value) for value in (axis.get("ticks") or [])]
    if not values:
        return ticks
    lo, hi = axis.get("minimum"), axis.get("maximum")
    max_length = max((length for _, length in ticks), default=1)

    def add(pixel):
        if not any(abs(existing - pixel) <= 2 for existing, _ in ticks):
            ticks.append((int(pixel), int(max_length + 1)))

    if lo is not None and np.isclose(float(lo), min(values)):
        add(end if vertical else start)
    if hi is not None and np.isclose(float(hi), max(values)):
        add(start if vertical else end)
    return sorted(ticks)


def calibrate(pix, vals, scale):
    vals = np.log10(vals) if scale == "log" else np.array(vals, float)
    a, b = np.polyfit(pix, vals, 1)
    resid = np.abs(np.polyval((a, b), pix) - vals).max()
    f = (lambda p: 10 ** (a * p + b)) if scale == "log" else (lambda p: a * p + b)
    return f, abs(a), float(resid), float(a), float(b)


class TickAnchorMismatchError(RuntimeError):
    """Agent tick evidence disagrees with deterministic chart geometry."""


def validate_tick_anchors(pixels, values, anchors, scale, axis_name, extent_px):
    """Reject a numerically linear but visibly shifted tick/value mapping.

    Agent 02's normalized pixel/value pairs are independent calibration
    evidence.  They must agree with the tick detector; merely fitting a line
    to two internally shifted arrays is not sufficient.
    """
    if not anchors:
        return []  # compatibility for hand-authored/offline specs
    if len(anchors) < 2:
        raise RuntimeError(f"{axis_name}-axis needs at least two tick anchors")
    anchor_pixels = np.array([float(item["pixel_norm"]) * extent_px for item in anchors])
    anchor_values = np.array([float(item["value"]) for item in anchors])
    fit_values = np.log10(anchor_values) if scale == "log" else anchor_values
    if scale == "log" and np.any(anchor_values <= 0):
        raise RuntimeError(f"{axis_name}-axis log anchor is not positive")
    slope, intercept = np.polyfit(anchor_pixels, fit_values, 1)
    detected_values = np.log10(values) if scale == "log" else np.asarray(values, float)
    predicted = slope * np.asarray(pixels, float) + intercept
    residual_value = float(np.max(np.abs(predicted - detected_values)))
    px_per_value = max(abs(slope), 1e-12)
    residual_px = residual_value / px_per_value
    limit = float(CFG["extract"].get("max_tick_anchor_residual_px", 3.0))
    if residual_px > limit:
        raise TickAnchorMismatchError(
            f"{axis_name}-axis tick anchors disagree with detected ticks "
            f"({residual_px:.2f}px > {limit:.2f}px); calibration ambiguous"
        )
    return [{"pixel_norm": float(item["pixel_norm"]), "value": float(item["value"])} for item in anchors]


def pixel_to_axis(calibration, axis, pixel, anchor=None):
    """Convert a pixel coordinate to an axis value, including logarithmic axes.

    New extraction files carry the fitted calibration model. ``anchor`` keeps
    older linear extraction files readable.
    """
    model = (calibration.get("axis_models") or {}).get(axis)
    if model:
        transformed = model["slope"] * pixel + model["intercept"]
        return 10**transformed if model.get("scale") == "log" else transformed
    if anchor is None:
        raise ValueError(f"calibration has no {axis!r} axis model")
    value, anchor_pixel = anchor
    delta = calibration[f"unit_per_px_{axis}"]
    direction = -1.0 if axis == "y" else 1.0
    return value + direction * (pixel - anchor_pixel) * delta


# ---------- color ----------


def to_lab(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def legend_colors(crop, legend_bbox, n_series, inner_bbox=None):
    _inner = inner_bbox
    """Find each legend entry's symbol (compact blob), top-to-bottom; return median Lab color of each.
    Colored symbols come from a saturation mask; black/grey symbols are added from a darkness mask but only
    inside the symbol column (so legend text is not mistaken for a marker)."""
    x0, y0, x1, y1 = legend_bbox
    lg = crop[y0:y1, x0:x1]
    bh, bw = lg.shape[:2]
    hsv = cv2.cvtColor(lg, cv2.COLOR_BGR2HSV)
    grey = cv2.cvtColor(lg, cv2.COLOR_BGR2GRAY)
    labimg = to_lab(lg)

    def blobs(mask):
        n, cc, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 15 or h > 0.35 * bh or w > 0.7 * bw or h < 4:
                continue
            m = cc == i
            # marker shape: remove the legend's horizontal line piece (thin vertically) with a vertical opening,
            # then crop to the remaining blob
            sub = m[y : y + h, x : x + w].astype(np.uint8)
            # marker core = the columns at least half as tall as the tallest column (a thin line stub never is)
            colsum = sub.sum(0)
            keep = np.where(colsum >= 0.5 * colsum.max())[0]
            core = sub[:, keep.min() : keep.max() + 1]
            rows_ = np.where(core.sum(1) > 0)[0]
            shape = core[rows_.min() : rows_.max() + 1, :] if len(rows_) else sub
            med = np.median(labimg[m], axis=0)
            spread = float(
                np.percentile(np.linalg.norm(labimg[m] - med, axis=1), 90)
            )  # JPEG color noise probe
            out.append(
                {
                    "cx": cent[i][0],
                    "cy": cent[i][1],
                    "w": w,
                    "h": h,
                    "area": area,
                    "bbox": (x, y, x + w, y + h),
                    "color": med,
                    "shape": shape,
                    "spread": spread,
                }
            )
        return out

    sat_entries = blobs((hsv[..., 1] > 90) & (hsv[..., 2] > 60))
    if not sat_entries:
        raise RuntimeError("legend: no colored symbols found — adjust legend_bbox")
    # symbol columns: x positions where colored blobs line up vertically (legends can have 2-3 columns).
    # Only columns inside the legend box as given (before padding) count — data markers dragged in by the padding do not.
    cxs = np.array([e["cx"] for e in sat_entries])
    hist, edges = np.histogram(cxs, bins=max(4, int(bw / 25)), range=(0, bw))
    half = 40
    centres = []
    for k in np.argsort(hist)[::-1]:
        if hist[k] < 2:
            break
        c = (edges[k] + edges[k + 1]) / 2
        if any(abs(c - c2) <= half for c2 in centres):
            continue
        if _inner is not None and not (_inner[0] - 6 <= x0 + c <= _inner[2] + 6):
            continue
        centres.append(c)
    if not centres:
        centres = [(edges[hist.argmax()] + edges[hist.argmax() + 1]) / 2]
    centres.sort()

    def col_of(e):
        d = [abs(e["cx"] - c) for c in centres]
        k = int(np.argmin(d))
        return k if d[k] <= half else None

    entries = [e for e in sat_entries if col_of(e) is not None]
    # black / grey series symbols: same columns AND within the rows spanned by the colored symbols (± one row),
    # so tick labels or axis ink that a loose box drags in are not taken for legend entries
    cys = sorted(e["cy"] for e in entries)
    hs = [e["h"] for e in entries]
    row_pitch = (
        float(np.median(np.diff(cys))) if len(cys) > 1 else 0.35 * bh / max(n_series, 1)
    )
    y_lo, y_hi = (
        (min(cys) - 1.5 * row_pitch, max(cys) + 1.5 * row_pitch) if cys else (0, bh)
    )
    # Pale open markers can have too little saturation to survive the primary
    # mask even though the filled symbols of the same color are detected.  A
    # second, lower-saturation pass is safe only after the symbol columns and
    # legend-row extent have been established by the high-confidence entries.
    # This preserves distinct filled/open series that intentionally share a
    # color (for example adsorption/desorption pairs).
    if len(entries) < n_series:
        # Very small anti-aliased open glyphs can fall below S=45 after raster
        # scaling/JPEG compression.  The relaxed cutoff is used only inside
        # the already-established symbol columns and legend-row span.
        faint_entries = blobs((hsv[..., 1] > 25) & (hsv[..., 2] > 50))
        for e in faint_entries:
            if col_of(e) is None or not (y_lo <= e["cy"] <= y_hi):
                continue
            if any(
                abs(e["cx"] - q["cx"]) < 6 and abs(e["cy"] - q["cy"]) < 6
                for q in entries
            ):
                continue
            entries.append(e)
    for e in blobs(grey < 90):  # black/grey: JPEG chroma noise, so no saturation test
        if (
            col_of(e) is not None
            and e["w"] >= 5
            and e["h"] >= 5
            and y_lo <= e["cy"] <= y_hi
            and not any(
                abs(e["cx"] - q["cx"]) < 6 and abs(e["cy"] - q["cy"]) < 6
                for q in entries
            )
        ):
            entries.append(e)
    for e in entries:
        e["col"] = col_of(e)
    entries.sort(key=lambda e: (e["col"], e["cy"]))
    # merge blobs on the same legend row (marker + line piece): same column, closer than 60 % of a symbol height
    merged = []
    row_gap = max(4.0, 0.6 * float(np.median(hs))) if hs else 6.0
    for e in entries:
        if (
            merged
            and merged[-1]["col"] == e["col"]
            and abs(e["cy"] - merged[-1]["cy"]) < row_gap
        ):
            m = merged[-1]
            a = m["area"] + e["area"]
            m["cy"] = (m["cy"] * m["area"] + e["cy"] * e["area"]) / a
            if (
                e["area"] > m["area"]
            ):  # keep the color/shape of the bigger blob (the marker)
                m["color"] = e["color"]
                m["shape"] = e["shape"]
                m["spread"] = e["spread"]
            bx = m["bbox"]
            m["bbox"] = (
                min(bx[0], e["bbox"][0]),
                min(bx[1], e["bbox"][1]),
                max(bx[2], e["bbox"][2]),
                max(bx[3], e["bbox"][3]),
            )
            m["area"] = a
        else:
            merged.append(dict(e))
    if len(merged) < n_series:
        raise RuntimeError(
            f"legend: found {len(merged)} colored entries, spec has {n_series} series — adjust legend_bbox"
        )
    # too many entries: prefer the ones inside the box as given (before padding), then the bigger ones
    merged = sorted(
        merged,
        key=lambda e: (
            not (_inner is not None and _inner[1] <= y0 + e["cy"] <= _inner[3]),
            -e["area"],
        ),
    )[:n_series]
    merged.sort(
        key=lambda e: (e["col"], e["cy"])
    )  # legend reading order: column by column, top to bottom
    boxes = [
        (
            x0 + e["bbox"][0] - 3,
            y0 + e["bbox"][1] - 3,
            x0 + e["bbox"][2] + 3,
            y0 + e["bbox"][3] + 3,
        )
        for e in merged
    ]
    return (
        [e["color"].astype(np.float32) for e in merged],
        [e["shape"] for e in merged],
        boxes,
        [e["spread"] for e in merged],
    )


def _legend_color_fits(measured, declared, series_item, tolerance):
    """Declared color match for a legend symbol.

    Open (hollow) markers are thin rings that JPEG blurs towards the background, so for them the
    hue (a*, b* direction) must match rather than the full Lab color."""
    if np.linalg.norm(measured - declared) < tolerance:
        return True
    if str(series_item.get("marker_fill") or "").lower() != "open":
        return False
    m, d = measured[1:] - 128.0, declared[1:] - 128.0
    cm, cd = float(np.hypot(*m)), float(np.hypot(*d))
    if cm < 12 or cd < 15:  # grey ring or grey declaration: no hue to compare
        return False
    angle = float(np.degrees(np.arccos(np.clip(np.dot(m, d) / (cm * cd), -1, 1))))
    return angle < 25 and measured[0] >= declared[0] - 10  # same hue, paler (never darker) than declared


def legend_rows_by_declared_color(crop, inner_bbox, series, tolerance=40.0):
    """Read the legend row by row with the declared series colors (agent 01 / agent 02).

    Fallback when the blob-based legend reader goes wrong (e.g. data markers next to the
    legend box, merged line stubs). Candidates are symbol blobs inside the legend box; each
    series takes the first blob below the previous series' symbol whose color matches its
    declared color, so the legend's top-to-bottom order is respected.  Returns the same
    tuple as ``legend_colors`` or None when a series has no match.
    """
    x0, y0, x1, y1 = (int(v) for v in inner_bbox)
    area = crop[y0:y1, x0:x1]
    if area.size == 0:
        return None
    lab = to_lab(area)
    grey = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
    ink = (grey < 200).astype(np.uint8)  # everything drawn: symbols, lines, text
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    candidates = []
    for i in range(1, count):
        x, y, w, h, n = stats[i]
        if n < 15 or h < 5 or h > 60:
            continue
        mask = labels[y:y + h, x:x + w] == i
        colsum = mask.sum(0)
        keep = np.where(colsum >= 0.5 * colsum.max())[0]  # marker core, not the thin line stub
        core = mask[:, keep.min():keep.max() + 1]
        rows_ = np.where(core.sum(1) > 0)[0]
        core = core[rows_.min():rows_.max() + 1]
        # color of the marker core's strongest ink (the thin anti-aliased line stub is paler)
        core_mask = np.zeros_like(mask)
        core_mask[:, keep.min():keep.max() + 1] = mask[:, keep.min():keep.max() + 1]
        pixels = lab[y:y + h, x:x + w][core_mask]
        chroma = np.hypot(pixels[:, 1] - 128, pixels[:, 2] - 128)
        if np.percentile(chroma, 75) >= 15:  # colored symbol: its most saturated pixels (not a dark outline)
            strong = pixels[chroma >= np.percentile(chroma, 85)]
        else:  # black / grey symbol: its darkest pixels
            strong = pixels[pixels[:, 0] <= np.percentile(pixels[:, 0], 50)]
        color = np.median(strong, axis=0)
        spread = float(np.percentile(np.linalg.norm(strong - color, axis=1), 90))
        candidates.append({"cy": float(centroids[i][1]), "cx": float(centroids[i][0]), "color": color,
                           "shape": core.astype(np.uint8), "spread": spread,
                           "box": (x0 + x - 3, y0 + y - 3, x0 + x + w + 3, y0 + y + h + 3)})
    candidates.sort(key=lambda c: c["cy"])

    def assign(pool):
        """One symbol per series in legend order: down a column, then on to the next column."""
        chosen, used, previous = [], set(), -1.0
        for item in series:
            try:
                want = color_from_hex(item.get("color_hex"))
            except ValueError:
                return None
            fits = [c for c in pool if id(c) not in used and _legend_color_fits(c["color"], want, item, tolerance)]
            below = [c for c in fits if c["cy"] > previous + 3]
            options = below or fits  # nothing further down: the legend continues in the next column
            if not options:
                return None
            match = min(options, key=lambda c: c["cy"])
            # same legend row (text digits, split glyphs, neighbouring column): the closest color wins
            row = [c for c in options if abs(c["cy"] - match["cy"]) <= 12]
            match = min(row, key=lambda c: np.linalg.norm(c["color"] - want))
            chosen.append(match)
            used.add(id(match))
            previous = match["cy"]
        return chosen

    # legend symbols sit in one or a few columns: keep candidates in columns that hold >= 2 matches
    first = assign(candidates)
    if first is None:
        return None
    xs = [c["cx"] for c in first]
    columns = [x for x in xs if sum(abs(x - other) <= 30 for other in xs) >= 2]
    pool = [c for c in candidates if any(abs(c["cx"] - x) <= 30 for x in columns)] if columns else candidates
    chosen = assign(pool) or first
    return ([c["color"].astype(np.float32) for c in chosen], [c["shape"] for c in chosen],
            [c["box"] for c in chosen], [c["spread"] for c in chosen])


def legend_reading_is_suspect(colors, boxes, series, inner_bbox, max_distance=35.0):
    """True when legend symbols fall outside the legend box or disagree with the declared colors."""
    x0, y0, x1, y1 = inner_bbox
    outside = any(not (x0 - 4 <= (b[0] + b[2]) / 2 <= x1 + 4 and y0 - 4 <= (b[1] + b[3]) / 2 <= y1 + 4) for b in boxes)
    wrong = 0
    for color, item in zip(colors, series):
        try:
            wrong += np.linalg.norm(color - color_from_hex(item.get("color_hex"))) > max_distance
        except ValueError:
            return outside
    return outside or wrong > 0.2 * max(len(series), 1)


def shape_score_maps(mask, shapes):
    """Normalised cross-correlation of the color mask with each marker shape template."""
    m = mask.astype(np.float32) / 255.0
    maps = []
    for sh in shapes:
        if sh is None:
            maps.append(None)
            continue
        t = sh.astype(np.float32)
        if (
            t.shape[0] < 3
            or t.shape[1] < 3
            or t.shape[0] >= m.shape[0]
            or t.shape[1] >= m.shape[1]
        ):
            maps.append(None)
            continue
        r = cv2.matchTemplate(m, t, cv2.TM_CCOEFF_NORMED)
        full = np.full(m.shape, -1.0, np.float32)
        oy, ox = t.shape[0] // 2, t.shape[1] // 2
        full[oy : oy + r.shape[0], ox : ox + r.shape[1]] = r
        maps.append(full)
    return maps


def refine_color(lab_img, plot_mask, seed, wide=60.0):
    """The legend symbol is only a seed (it is often rendered lighter / blurrier than the data markers).
    Re-measure the color on marker interiors in the plot: pixels within `wide` of the seed, keeping only
    the cores of blobs (distance transform >= 2 px) so edges and anti-aliasing do not bias the median.
    Returns (color, tolerance)."""
    m = cv2.bitwise_and(color_mask(lab_img, seed, wide), plot_mask)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    core = dt >= 2.0
    if core.sum() < 50:
        return seed, None
    px = lab_img[core]
    med = np.median(px, axis=0).astype(np.float32)
    if (
        np.linalg.norm(med - seed) > 45
    ):  # refinement drifted to another series: keep the seed
        return seed, None
    spread = float(np.percentile(np.linalg.norm(px - med, axis=1), 90))
    return med, spread


def color_mask(lab_img, color, tol=COLOR_TOL):
    d = np.linalg.norm(lab_img - color[None, None, :], axis=2)
    return (d < tol).astype(np.uint8) * 255


def color_from_hex(value):
    """Convert an RGB hex color supplied by structural analysis to OpenCV Lab."""
    text = str(value or "").strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"invalid RGB color {value!r}")
    red, green, blue = (int(text[i : i + 2], 16) for i in (0, 2, 4))
    bgr = np.uint8([[[blue, green, red]]])
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def uses_direct_series_labels(spec):
    """True when labels identify series directly and no marker legend exists."""
    position = str(
        spec.get("legend_position")
        or (spec.get("llm_spec") or {}).get("legend_position")
        or ""
    ).lower()
    return "direct" in position and ("label" in position or "annot" in position)


def proposal_quality(result, spec):
    """Return cheap structural warnings for a supporting Python proposal."""
    warnings = []
    expected_by_label = {
        str(series.get("label")): series.get("n_markers_estimate")
        for series in spec.get("series", [])
    }
    for series in result.get("series", []):
        points = [
            point
            for point in series.get("points", [])
            if point.get("source") not in {"curve", "grid"}
        ]
        expected = expected_by_label.get(str(series.get("label")))
        if expected and len(points) < 0.75 * float(expected):
            warnings.append(
                {
                    "check": "marker_count",
                    "series": series.get("label"),
                    "detail": f"detected {len(points)} of about {expected} expected markers",
                }
            )
        if points and all(
            point.get("overlap_flag") or float(point.get("confidence", 0)) < CONF_OK
            for point in points
        ):
            warnings.append(
                {
                    "check": "all_points_flagged",
                    "series": series.get("label"),
                    "detail": "every detected marker is low-confidence or overlap-flagged",
                }
            )
        weak_points = [
            point
            for point in points
            if point.get("overlap_flag") or float(point.get("confidence", 0)) < CONF_OK
        ]
        maximum_weak_fraction = float(
            CFG["extract"].get("max_weak_point_fraction", 0.25)
        )
        if points and len(weak_points) / len(points) > maximum_weak_fraction:
            warnings.append(
                {
                    "check": "weak_marker_coverage",
                    "series": series.get("label"),
                    "detail": (
                        f"{len(weak_points)} of {len(points)} markers are low-confidence or overlapping; "
                        f"maximum is {maximum_weak_fraction:.0%}"
                    ),
                }
            )
    for axis_id in ("x", "y"):
        axis = spec.get(axis_id) or {}
        ticks = [float(value) for value in axis.get("ticks") or []]
        minimum, maximum = axis.get("minimum"), axis.get("maximum")
        if not ticks or minimum is None or maximum is None:
            continue
        if not (
            np.isclose(float(minimum), min(ticks))
            and np.isclose(float(maximum), max(ticks))
        ) and not axis.get("boundary_limits_confirmed", False):
            warnings.append(
                {
                    "check": "axis_boundary_conflict",
                    "axis": axis_id,
                    "detail": (
                        f"{axis_id}-axis limits {minimum}..{maximum} differ from the printed tick range "
                        f"{min(ticks)}..{max(ticks)} without confirmed unlabeled boundaries"
                    ),
                }
            )
    calibration = result.get("calibration") or {}
    maximum_residual = float(CFG["extract"].get("max_tick_fit_residual_px", 2.5))
    for axis in ("x", "y"):
        residual = calibration.get(f"tick_fit_resid_{axis}_px")
        if residual is not None and float(residual) > maximum_residual:
            warnings.append(
                {
                    "check": "calibration_residual",
                    "axis": axis,
                    "detail": (
                        f"{axis}-axis tick reprojection residual is {float(residual):.2f}px; "
                        f"maximum is {maximum_residual:.2f}px"
                    ),
                }
            )
    return {"review_required": bool(warnings), "warnings": warnings}


# ---------- markers ----------


def hollow_circle_centres(mask, expected_count, plot_width):
    """Detect open circular outlines in a single-series color mask.

    This path is used for directly annotated charts, where no legend glyph is
    available as a template.  Hough candidates must also have substantial ink
    on their fitted annulus, which suppresses ordinary line and text fragments.
    """
    if not expected_count or expected_count < 2:
        return [], 0.0
    min_distance = max(5.0, 1.15 * float(plot_width) / float(expected_count))
    max_radius = max(6, min(9, int(round(0.018 * plot_width))))
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(mask, (3, 3), 0),
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=min_distance,
        param1=50,
        param2=6,
        minRadius=3,
        maxRadius=max_radius,
    )
    if circles is None:
        return [], 0.0
    height, width = mask.shape
    accepted = []
    for cx, cy, radius in circles[0]:
        extent = int(np.ceil(radius + 2))
        x0, x1 = max(0, int(cx) - extent), min(width, int(cx) + extent + 1)
        y0, y1 = max(0, int(cy) - extent), min(height, int(cy) + extent + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        distance = np.hypot(xx - cx, yy - cy)
        annulus = (distance >= max(1.0, radius - 2.0)) & (distance <= radius + 2.0)
        coverage = (
            float((mask[y0:y1, x0:x1][annulus] > 0).mean()) if annulus.any() else 0.0
        )
        if coverage >= 0.18:
            accepted.append((float(cx), float(cy), min(0.95, 0.65 + coverage), False))
    radius = float(np.median(circles[0, :, 2])) if len(circles[0]) else 0.0
    accepted.sort()
    return accepted, radius


def _circle_fit(xs, ys, robust_iters=3, inlier_px=1.5):
    """Algebraic (Kasa) circle fit with iterative inlier trimming -> cx, cy, R, rms(inliers), inlier fraction.
    Edge pixels of lines crossing the marker are dropped instead of inflating the residual."""
    keep = np.ones(len(xs), bool)
    for _ in range(robust_iters + 1):
        A = np.c_[2 * xs[keep], 2 * ys[keep], np.ones(keep.sum())]
        b = xs[keep] ** 2 + ys[keep] ** 2
        (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
        R = np.sqrt(max(c + cx**2 + cy**2, 1e-6))
        res = np.abs(np.hypot(xs - cx, ys - cy) - R)
        new_keep = res < inlier_px
        if new_keep.sum() < 6 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    rms = float(np.sqrt(np.mean(res[keep] ** 2)))
    return cx, cy, R, rms, float(keep.mean())


def marker_centres(mask, sat_img, r_hint=None, template=None, force_hint=False):
    """Distance-transform peaks propose candidates; a circle fit on the mask edge accepts
    true markers (incl. partially occluded crescents) and rejects line bands.
    r_hint: known marker radius (px) — use it when the mask is a small window where merged blobs would inflate the estimate."""
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # marker radius from the plot: median of the local distance-transform maxima (a pair of overlapping markers
    # still peaks at one radius; only fused blocks and bands peak higher, and the median ignores them)
    mx0 = ndimage.maximum_filter(dt, size=7)
    pk0 = dt[
        (dt == mx0) & (dt >= max(2.0, 0.4 * float(dt.max())))
    ]  # ignore the line's own small peaks
    r_dt = (
        float(np.percentile(pk0, 93))
        if len(pk0) >= 5
        else (float(np.percentile(dt[dt > 0], 99)) if (dt > 0).any() else 0)
    )
    if force_hint and r_hint:
        r_est = float(r_hint)
    elif r_hint and len(pk0) >= 5:
        # the legend glyph selects the cluster of peaks that are markers (bands peak lower, fused blocks higher)
        near = pk0[(pk0 >= 0.7 * r_hint) & (pk0 <= 1.3 * r_hint)]
        r_est = float(np.median(near)) if len(near) >= 5 else r_dt
    else:
        r_est = r_dt
    if r_est < 2.5:
        return [], r_est
    min_r = max(2.0, 0.45 * r_est)
    mx = ndimage.maximum_filter(dt, size=int(max(5, r_est)))
    peaks = np.argwhere((dt == mx) & (dt >= min_r))
    # also peaks of the opened mask (isolated markers whose dt peak was displaced by the line)
    k0 = int(max(1, round(0.45 * r_est)))
    op0 = cv2.morphologyEx(
        (mask > 0).astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k0 + 1, 2 * k0 + 1)),
    )
    dt2 = cv2.distanceTransform(op0 * 255, cv2.DIST_L2, 5)
    mx2 = ndimage.maximum_filter(dt2, size=int(max(5, r_est)))
    peaks2 = np.argwhere((dt2 == mx2) & (dt2 >= max(1.5, 0.28 * r_est)))
    peaks = np.concatenate([peaks, peaks2]) if len(peaks2) else peaks
    # fused blocks (several touching markers): split by matching the legend glyph itself, with non-max suppression
    if template is not None and template.shape[0] >= 5 and template.shape[1] >= 5:
        tpl = template.astype(np.float32)
        th, tw = tpl.shape
        big = np.zeros(mask.shape, np.uint8)
        n_b, cc_b, st_b, _ = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8)
        )
        for i in range(1, n_b):
            if st_b[i, cv2.CC_STAT_AREA] > 2.2 * np.pi * r_est**2:
                big[cc_b == i] = 1
        if mask.shape[0] > th and mask.shape[1] > tw:
            # whole mask: the legend glyph finds markers whose distance-transform peak is small (triangles, stars)
            whole = (mask > 0).astype(np.float32)
            resp = cv2.matchTemplate(whole, tpl, cv2.TM_CCORR_NORMED)
            full = np.zeros(mask.shape, np.float32)
            full[
                th // 2 : th // 2 + resp.shape[0], tw // 2 : tw // 2 + resp.shape[1]
            ] = resp
            mxr = ndimage.maximum_filter(full, size=int(max(3, 1.2 * r_est)))
            tp = np.argwhere((full == mxr) & (full >= 0.82) & (mask > 0))
            # fused blocks: stricter threshold, split by the glyph
            if big.any():
                resp_b = cv2.matchTemplate(
                    big.astype(np.float32), tpl, cv2.TM_CCORR_NORMED
                )
                full_b = np.zeros(mask.shape, np.float32)
                full_b[
                    th // 2 : th // 2 + resp_b.shape[0],
                    tw // 2 : tw // 2 + resp_b.shape[1],
                ] = resp_b
                mxb = ndimage.maximum_filter(full_b, size=int(max(3, 1.2 * r_est)))
                tb = np.argwhere((full_b == mxb) & (full_b >= 0.9) & (big > 0))
                tp = np.concatenate([tp, tb]) if len(tb) else tp
            if len(tp):
                # annulus test: a marker (plus its thin line) leaves the ring 1.15r..1.5r mostly empty; a band fills it
                yy, xx = np.mgrid[
                    -int(1.5 * r_est) - 1 : int(1.5 * r_est) + 2,
                    -int(1.5 * r_est) - 1 : int(1.5 * r_est) + 2,
                ]
                ring = (np.hypot(yy, xx) >= 1.15 * r_est) & (
                    np.hypot(yy, xx) <= 1.5 * r_est
                )
                keep_tp = []
                m01 = mask > 0
                for y_, x_ in tp:
                    ya_, yb_ = y_ + yy.min(), y_ + yy.max() + 1
                    xa_, xb_ = x_ + xx.min(), x_ + xx.max() + 1
                    if ya_ < 0 or xa_ < 0 or yb_ > mask.shape[0] or xb_ > mask.shape[1]:
                        continue
                    frac = m01[ya_:yb_, xa_:xb_][ring].mean()
                    if frac <= 0.30:
                        keep_tp.append((y_, x_))
                if keep_tp:
                    peaks = np.concatenate([peaks, np.array(keep_tp)])
    edge = (mask > 0) & (cv2.erode(mask, np.ones((3, 3), np.uint8)) == 0)
    H, W = mask.shape
    # connected components on an OPENED mask (a disc a bit smaller than the marker erases the thin connecting
    # line, leaving markers as separate compact blobs of ANY shape)
    k = int(max(1, round(0.45 * r_est)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    opened = cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_OPEN, kernel)
    n_cc, cc, cc_stats, cc_cent = cv2.connectedComponentsWithStats(opened)
    disc = np.pi * r_est**2
    pts = []
    for y, x in peaks:
        if any(
            abs(x - px) < 0.9 * r_est and abs(y - py) < 0.9 * r_est
            for px, py, *_ in pts
        ):
            continue
        ci = cc[y, x]
        if ci > 0:
            bx, by, bw_, bh_, area = cc_stats[ci]
            aspect = bw_ / max(1, bh_)
            if 0.45 * disc <= area <= 1.25 * disc and 0.55 <= aspect <= 1.8:
                solidity = area / max(1.0, bw_ * bh_)
                if (
                    solidity >= 0.55
                ):  # squares ~1, discs ~0.78, diamonds/triangles ~0.5-0.6
                    cx, cy = float(cc_cent[ci][0]), float(cc_cent[ci][1])
                    pts.append((cx, cy, float(min(1.0, 0.6 + 0.5 * solidity)), False))
                    continue
            if 1.25 * disc < area <= 2.05 * disc and r_hint:
                # two overlapping markers of the same series (adsorption/desorption twins): the union area of two
                # equal discs fixes their separation d; the blob's principal axis gives the direction
                ys_, xs_ = np.where(cc == ci)
                cxm, cym = xs_.mean(), ys_.mean()
                cov = np.cov(np.vstack([xs_ - cxm, ys_ - cym]))
                evals, evecs = np.linalg.eigh(cov)
                ax_ = evecs[:, int(np.argmax(evals))]
                elong = np.sqrt(max(evals) / max(1e-6, min(evals)))
                if elong >= 1.15:
                    R_ = r_est

                    def union_area(d):
                        d = min(d, 2 * R_)
                        return 2 * np.pi * R_**2 - (
                            2 * R_**2 * np.arccos(d / (2 * R_))
                            - (d / 2) * np.sqrt(max(0.0, 4 * R_**2 - d**2))
                        )

                    lo_, hi_ = 0.0, 2 * R_
                    for _ in range(30):
                        mid = (lo_ + hi_) / 2
                        if union_area(mid) < area:
                            lo_ = mid
                        else:
                            hi_ = mid
                    d = (lo_ + hi_) / 2
                    if d >= 0.5 * R_:
                        for sgn in (-1, 1):
                            pts.append(
                                (
                                    float(cxm + sgn * d / 2 * ax_[0]),
                                    float(cym + sgn * d / 2 * ax_[1]),
                                    0.6,
                                    True,
                                )
                            )
                        continue
        R = int(np.ceil(1.6 * r_est))
        y0, y1, x0, x1 = (
            max(0, y - R),
            min(H, y + R + 1),
            max(0, x - R),
            min(W, x + R + 1),
        )
        ey, ex = np.where(edge[y0:y1, x0:x1])
        ey, ex = ey + y0, ex + x0
        d = np.hypot(ex - x, ey - y)
        keep = (d > 0.55 * r_est) & (d < 1.45 * r_est)
        if keep.sum() < 8:
            continue
        cx, cy, Rf, rms, inl = _circle_fit(
            ex[keep].astype(float), ey[keep].astype(float)
        )
        if not (0.75 * r_est < Rf < 1.3 * r_est) or rms > 0.2 * r_est or inl < 0.7:
            continue  # straight edges / wrong size -> not a marker
        # arc coverage: fraction of 24 angular bins with an edge pixel near the fitted circle
        on = np.abs(np.hypot(ex - cx, ey - cy) - Rf) < 1.5
        ang = np.arctan2(ey[on] - cy, ex[on] - cx)
        cover = len(np.unique(((ang + np.pi) / (2 * np.pi) * 24).astype(int))) / 24.0
        if cover < 0.35:
            continue
        # annulus test against line bands: the ring 1.15r..1.5r around a marker is mostly empty (its thin line only)
        R2 = int(1.5 * r_est) + 1
        ya_, yb_, xa_, xb_ = (
            int(cy) - R2,
            int(cy) + R2 + 1,
            int(cx) - R2,
            int(cx) + R2 + 1,
        )
        if ya_ >= 0 and xa_ >= 0 and yb_ <= mask.shape[0] and xb_ <= mask.shape[1]:
            yy_, xx_ = np.mgrid[ya_:yb_, xa_:xb_]
            rr_ = np.hypot(yy_ - cy, xx_ - cx)
            ring_ = (rr_ >= 1.15 * r_est) & (rr_ <= 1.5 * r_est)
            frac_ = (
                float((mask[ya_:yb_, xa_:xb_] > 0)[ring_].mean())
                if ring_.any()
                else 0.0
            )
            core_ = float(
                dt[
                    min(mask.shape[0] - 1, max(0, int(round(cy)))),
                    min(mask.shape[1] - 1, max(0, int(round(cx)))),
                ]
            )
            if frac_ > 0.25 and core_ < 0.8 * r_est:
                continue  # ring full AND no full disc at the centre: a band, not a marker (a crowded marker
                # has a full core; an occluded crescent has an empty ring)
        overlap = float(dt[y, x]) > 1.25 * r_est
        pts.append((float(cx), float(cy), float(cover), overlap))
    pts.sort()
    return pts, r_est


def settle_twins(
    all_series,
    fx,
    fy,
    lab=None,
    plot_mask=None,
    tol=None,
    shapes=None,
    x_ticks_vals=None,
    skip_indices=None,
    global_pool=False,
):
    """Series with near-identical colors share candidate points. Pool them, link markers that are joined by a
    line of that color into chains, and let each chain vote for a series by marker shape (a chain of 20 markers
    is a far better voter than one blurred marker). Points off any chain are decided by their own shape score."""
    by_idx = {S["idx"]: S for S in all_series}
    if len(all_series) < 2:
        return all_series
    if global_pool:
        # Crowded, filled, adsorption-only panels need the older invariant:
        # all detector passes compete for one physical marker location.  The
        # caller applies a narrow structural guard, so open rings and genuine
        # adsorption/desorption twins never enter this mode.
        groups = {"global": sorted(by_idx)}
    else:
        # Otherwise ownership resolution is only meaningful within a visually
        # confusable component. Pooling unrelated series can erase valid
        # same-pixel markers on general chart types.
        parent = {s["idx"]: s["idx"] for s in all_series}
        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i
        for a, left_series in enumerate(all_series):
            for right_series in all_series[a + 1:]:
                color_distance = float(np.linalg.norm(left_series["col"] - right_series["col"]))
                threshold = max(float(left_series.get("tol") or 0), float(right_series.get("tol") or 0)) + 6.0
                if color_distance <= threshold:
                    parent[root(left_series["idx"])] = root(right_series["idx"])
        groups = {}
        for index in parent:
            groups.setdefault(root(index), []).append(index)
    hypotheses = []
    for group in (tuple(sorted(members)) for members in groups.values() if len(members) > 1):
        if not global_pool and skip_indices and any(index in skip_indices for index in group):
            # Ambiguous groups use the geometry-first joint scorer below.  Do
            # not let the legacy color/shape pool discard candidates first.
            continue
        # ---- pool + dedupe ----
        r = max(by_idx[j]["r"] for j in group)
        pool = []
        for j in group:
            for cx, cy, conf, ov, sure in by_idx[j]["pts"]:
                k = next(
                    (
                        i
                        for i, q in enumerate(pool)
                        if abs(q["cx"] - cx) < 0.8 * r and abs(q["cy"] - cy) < 0.8 * r
                    ),
                    None,
                )
                if k is None:
                    pool.append(
                        {"cx": cx, "cy": cy, "conf": conf, "ov": ov, "from": {j}}
                    )
                else:
                    pool[k]["from"].add(j)
                    pool[k]["conf"] = max(pool[k]["conf"], conf)
        if not pool:
            return all_series
        # pixel color of each pooled marker (median Lab inside the marker)
        for q in pool:
            yy, xx = int(round(q["cy"])), int(round(q["cx"]))
            rr = max(2, int(0.5 * r))
            patch = lab[
                max(0, yy - rr) : yy + rr + 1, max(0, xx - rr) : xx + rr + 1
            ].reshape(-1, 3)
            q["lab"] = np.median(patch, axis=0)
            q["cdist"] = [
                float(np.linalg.norm(q["lab"] - by_idx[j]["col"])) for j in group
            ]
        # ---- union color mask of the group (dilated a little for JPEG gaps) ----
        m = np.zeros(plot_mask.shape, np.uint8)
        for j in group:
            m |= cv2.bitwise_and(
                color_mask(lab, by_idx[j]["col"], by_idx[j]["tol"]), plot_mask
            )
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))
        # ---- shape score maps for every series of the group ----
        maps = shape_score_maps(m, [shapes[j] for j in group])

        def score(j_pos, cx, cy):
            mp = maps[j_pos]
            if mp is None:
                return 0.0
            yy, xx = int(round(cy)), int(round(cx))
            return float(mp[max(0, yy - 2) : yy + 3, max(0, xx - 2) : xx + 3].max())

        for q in pool:
            q["scores"] = [score(k, q["cx"], q["cy"]) for k in range(len(group))]
        # ---- link markers joined by a line ----
        pool.sort(key=lambda q: q["cx"])
        parent = list(range(len(pool)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def segment_ok(a, b):
            ax, ay, bx, by = pool[a]["cx"], pool[a]["cy"], pool[b]["cx"], pool[b]["cy"]
            n = max(6, int(np.hypot(bx - ax, by - ay) / 2))
            inside = 0
            for t in np.linspace(0.15, 0.85, n):
                x, y = ax + t * (bx - ax), ay + t * (by - ay)
                if m[int(round(y)), int(round(x))]:
                    inside += 1
            if inside < 0.85 * n:
                return False
            # the segment must not run through a third marker (that would merge two series)
            for c, q in enumerate(pool):
                if c in (a, b):
                    continue
                t = ((q["cx"] - ax) * (bx - ax) + (q["cy"] - ay) * (by - ay)) / max(
                    1e-9, (bx - ax) ** 2 + (by - ay) ** 2
                )
                if (
                    0.1 < t < 0.9
                    and np.hypot(
                        q["cx"] - (ax + t * (bx - ax)), q["cy"] - (ay + t * (by - ay))
                    )
                    < 1.2 * r
                ):
                    return False
            return True

        def crowded(i):
            return (
                sum(
                    1
                    for q in pool
                    if q is not pool[i]
                    and abs(q["cx"] - pool[i]["cx"]) < 2.5 * r
                    and abs(q["cy"] - pool[i]["cy"]) < 2.5 * r
                )
                >= 1
            )

        for a in range(len(pool)):
            if crowded(a):
                continue  # dense zone (low pressure): do not chain, decide point by point
            # candidates to the right, nearest first, within a generous window
            cands = sorted(
                [
                    b
                    for b in range(a + 1, len(pool))
                    if pool[b]["cx"] - pool[a]["cx"] > 0.5 * r
                    and pool[b]["cx"] - pool[a]["cx"] < 25 * r
                ],
                key=lambda b: np.hypot(
                    pool[b]["cx"] - pool[a]["cx"], pool[b]["cy"] - pool[a]["cy"]
                ),
            )
            for b in cands[:6]:
                if not crowded(b) and segment_ok(a, b):
                    parent[find(a)] = find(b)
                    break
        chains = {}
        for i in range(len(pool)):
            chains.setdefault(find(i), []).append(i)

        # ---- shape classes: members whose templates are near-identical (e.g. CO2 373K ● and N2 273K ●) ----
        def norm_t(t):
            if t is None:
                return None
            return (
                cv2.resize(t.astype(np.float32), (16, 16), interpolation=cv2.INTER_AREA)
                > 0.5
            )

        T = [norm_t(shapes[j]) for j in group]
        same = [
            [
                True
                if T[a] is None and T[b] is None
                else (
                    False
                    if T[a] is None or T[b] is None
                    else (T[a] & T[b]).sum() / max(1, (T[a] | T[b]).sum()) > 0.75
                )
                for b in range(len(group))
            ]
            for a in range(len(group))
        ]
        anchors, curves_hint = {}, {}
        for j in group:
            sp_j = by_idx[j]["spec"]
            ya = sp_j.get("y_at_xmax")
            yan = sp_j.get("y_anchors")
            if yan and all(k in yan for k in ("p25", "p50", "p75", "p100")):
                ya = yan["p100"]
                curves_hint[j] = (
                    [0.0, 0.25, 0.5, 0.75, 1.0],
                    [0.0, yan["p25"], yan["p50"], yan["p75"], yan["p100"]],
                )
            if ya is not None:
                anchors[j] = ya
        x_ref = max(x_ticks_vals) if x_ticks_vals else None

        def compatible(j, cx, cy):
            """With 4 anchors: the point must lie within ±20 % (of ymax) of the interpolated rough curve.
            With one anchor: isotherm rule, increasing + concave -> ymax*x/xref <= y <= ymax (12 % slack)."""
            ya = anchors.get(group[j])
            if ya is None or x_ref is None:
                return True
            x, y = fx(cx), fy(cy)
            if group[j] in curves_hint:
                fr, yy = curves_hint[group[j]]
                y_hint = float(np.interp(max(0.0, x) / x_ref, fr, yy))
                band = 0.20 * max(abs(ya), 0.05) + 0.02
                return abs(y - y_hint) <= band
            lo = ya * max(0.0, x) / x_ref
            return (y <= ya * 1.12 + 0.02) and (y >= lo * 0.88 - 0.02)

        def choose(members):
            """members: pool indices of one chain (or a singleton). returns (group position, sure)"""
            right = max(members, key=lambda i: pool[i]["cx"])
            cands = [
                k
                for k in range(len(group))
                if all(compatible(k, pool[i]["cx"], pool[i]["cy"]) for i in members)
            ]
            if not cands:
                cands = list(range(len(group)))
            if len(cands) == 1:
                return cands[0], True

            # --- adaptive shape trust: are the candidate shapes distinguishable at this resolution? ---
            def template_iou(a, b):
                if T[a] is None or T[b] is None:
                    return 1.0 if T[a] is None and T[b] is None else 0.0
                return (T[a] & T[b]).sum() / max(1, (T[a] | T[b]).sum())

            iou = [[template_iou(a, b) for b in cands] for a in cands]
            max_iou = max(
                (
                    iou[a][b]
                    for a in range(len(cands))
                    for b in range(len(cands))
                    if a != b
                ),
                default=1.0,
            )
            shape_votes = np.zeros(len(group))
            for i in members:
                sc = np.array(
                    [
                        pool[i]["scores"][c] if c in cands else -9
                        for c in range(len(group))
                    ]
                )
                shape_votes[int(sc.argmax())] += 1
            top = int(shape_votes.argmax())
            share = shape_votes[top] / max(1, len(members))
            crisp = (
                r >= 10 and max_iou < 0.6
            )  # big markers AND clearly different shapes
            if crisp and share >= 0.7 and len(members) >= 2:
                return top, True  # shape is decisive here: use it before color
            # color: keep candidates whose legend color is close to the markers' own color
            cd = np.array(
                [
                    np.mean([pool[i]["cdist"][c] for i in members])
                    for c in range(len(group))
                ]
            )
            best_cd = min(cd[c] for c in cands)
            near = [c for c in cands if cd[c] <= best_cd + 14]
            if len(near) == 1:
                return near[0], True
            cands = near
            votes = np.zeros(len(group))
            for i in members:
                sc = np.array(pool[i]["scores"])
                sc = np.array([sc[c] if c in cands else -9 for c in range(len(group))])
                votes[int(sc.argmax())] += 1
            class_votes = np.array(
                [
                    sum(votes[c] for c in cands if same[k][c]) if k in cands else -1
                    for k in range(len(group))
                ]
            )
            k = int(class_votes.argmax())
            cls = [c for c in cands if same[k][c]]
            if len(cls) > 1 and all(group[c] in anchors for c in cls):
                y_r = fy(pool[right]["cy"])
                # nearest anchor at or above the chain's right end
                k = min(
                    cls,
                    key=lambda c: (
                        anchors[group[c]] < y_r * 0.95,
                        abs(anchors[group[c]] - y_r),
                    ),
                )
            sure = len(members) >= 3 and class_votes[k] >= 0.6 * len(members)
            return k, sure

        # ---- vote per chain ----
        assigned = {j: [] for j in group}
        for members in chains.values():
            k, sure = choose(members)
            for i in members:
                q = pool[i]
                if not sure and len(q["from"]) > 1:
                    hypotheses.append({
                        "px": [round(q["cx"], 2), round(q["cy"], 2)],
                        "candidate_series_indices": sorted(q["from"]),
                        "reason": "multi_series_overlap_hypothesis",
                    })
                    continue
                assigned[group[k]].append(
                    (
                        q["cx"],
                        q["cy"],
                        q["conf"] if sure else min(q["conf"], 0.7),
                        q["ov"],
                        bool(sure),
                    )
                )
        for j in group:
            by_idx[j]["pts"] = sorted(assigned[j])
        settle_twins.pool = pool
    settle_twins.hypotheses = hypotheses
    return all_series


def pixel_color_check(all_series, lab, skip_indices=None):
    """Last word on ownership: the 5x5 pixels at a marker centre. If they clearly are another series' color
    (that color within 15 Lab, the owner's more than 30 away) the marker moves to that series; if it is merely
    doubtful it stays but loses its confidence."""
    cols = [(S["idx"], S["col"]) for S in all_series]
    by_idx = {S["idx"]: S for S in all_series}
    moved = []
    skip_indices = set(skip_indices or ())
    for S in all_series:
        kept = []
        for p in S["pts"]:
            if S["idx"] in skip_indices:
                kept.append(p)
                continue
            cx, cy = int(round(p[0])), int(round(p[1]))
            marker_fill = str(S.get("spec", {}).get("marker_fill", "filled")).lower()
            if marker_fill == "open":
                radius = max(3, int(round(S.get("r") or 3)))
                y0, y1 = max(0, cy - radius), min(lab.shape[0], cy + radius + 1)
                x0, x1 = max(0, cx - radius), min(lab.shape[1], cx + radius + 1)
                yy, xx = np.mgrid[y0:y1, x0:x1]
                distance = np.hypot(xx - cx, yy - cy)
                annulus = (distance >= 0.55 * radius) & (distance <= 1.25 * radius)
                win = lab[y0:y1, x0:x1][annulus].reshape(-1, 3).astype(np.float32)
            else:
                win = (
                    lab[max(0, cy - 2) : cy + 3, max(0, cx - 2) : cx + 3]
                    .reshape(-1, 3)
                    .astype(np.float32)
                )
            if len(win) == 0:
                kept.append(p)
                continue
            d = {i: float(np.median(np.linalg.norm(win - c, axis=1))) for i, c in cols}
            mine = d[S["idx"]]
            other, d_other = min(
                ((i, v) for i, v in d.items() if i != S["idx"]),
                key=lambda t: t[1],
                default=(None, 1e9),
            )
            wrong_color = other is not None and d_other + 10 < mine
            # A trusted Hough outline must contain its declared color around
            # the fitted annulus.  At high raster resolutions Hough can fit a
            # circle to two neighbouring marker arcs plus their connector;
            # those gap reads fail this test and are not useful proposals.
            # Keep the older low-confidence behavior for fallback detections,
            # where partial/merged rings may be the only available evidence.
            if S.get("hough_outline_only") and marker_fill == "open" and wrong_color:
                continue
            if (
                other is not None
                and d_other < 15
                and mine > 30
                and np.linalg.norm(by_idx[other]["col"] - S["col"]) > 25
            ):
                moved.append((other, p))
            elif wrong_color:
                kept.append((p[0], p[1], min(p[2], 0.5), True, False))
            else:
                kept.append(p)
        S["pts"] = kept
    for i, p in moved:
        by_idx[i]["pts"].append((p[0], p[1], min(p[2], 0.7), p[3], False))
    for S in all_series:
        S["pts"].sort()
    return all_series


def curve_owner(candidate, series_list, fx, fy, x_range):
    """The one series whose agent 02 curve / y band explains a disputed candidate, else None.

    Runs only when agent 02 coached every competing series (``curve_points`` or ``y_bands``):
    same-color twins such as a CO2 and an N2 isotherm in the same black run far apart, so the
    curve position decides where marker shape and color can not.
    """
    if fx is None or fy is None or len(series_list) < 2 or x_range is None:
        return None
    if not all(s["spec"].get("curve_points") or s["spec"].get("y_bands") for s in series_list):
        return None
    x, y = float(fx(candidate["px"][0])), float(fy(candidate["px"][1]))
    plausible = []
    for series in series_list:
        band = series_gate.in_band(series["spec"], x, y)
        if band is False:
            continue
        curve = series_gate.anchor_curve(series["spec"], *x_range)
        distance = abs(y - curve[0](x)) if curve and curve[1] <= x <= curve[2] else None
        plausible.append((0.0 if band else distance, series["idx"]))
    if len(plausible) == 1:
        return plausible[0][1]
    known = sorted((d, idx) for d, idx in plausible if d is not None)
    if len(known) == len(plausible) and len(known) >= 2 and known[0][0] < 0.5 * known[1][0]:
        return known[0][1]
    return None


def spatial_ownership_guard(spec, ambiguity_groups=None, complex_regions=None):
    """Return the narrow trigger for cross-series native-marker ownership.

    A global spatial pass is intentionally not a general chart deduplicator:
    open-marker/labelled panels and adsorption/desorption pairs can legitimately
    put different branches at nearly the same pixel.  The problematic family
    is the crowded, multi-series, *adsorption-only* family where all declared
    series use filled symbols.  The returned fields are kept verbatim in the
    calibration audit so a run can explain why the pass was (or was not)
    enabled.
    """
    chart = spec or {}
    llm = chart.get("llm_spec") or {}
    branches = str(llm.get("branches") or chart.get("branches") or "").strip().lower()
    branches = branches.replace("-", "_").replace(" ", "_")
    series = list(chart.get("series") or [])
    multi_series = len(series) >= 2
    filled = all(
        str(item.get("marker_fill") or "filled").strip().lower()
        not in {"open", "hollow", "outline", "ring"}
        for item in series
    )
    dense_kinds = {"dense_markers", "overlapping_markers", "thick_connecting_lines", "crossing"}
    regions = list(complex_regions if complex_regions is not None else chart.get("complex_regions") or [])
    crowded_region = any(str(item.get("kind") or "") in dense_kinds for item in regions)
    ambiguous = bool(ambiguity_groups)
    enabled = bool(branches == "adsorption_only" and multi_series and filled and (ambiguous or crowded_region))
    return {
        "enabled": enabled,
        "branches": branches or None,
        "multi_series": multi_series,
        "all_filled": filled,
        "ambiguity_group_count": len(ambiguity_groups or []),
        "crowded_region_count": sum(
            1 for item in regions if str(item.get("kind") or "") in dense_kinds
        ),
        "reason": (
            "adsorption_only_multi_series_filled_crowded"
            if enabled
            else "guard_not_satisfied"
        ),
    }


def global_spatial_ownership(
    all_series,
    *,
    spec=None,
    enabled=None,
    radius=None,
    distance_fraction=0.8,
    confidence_margin=0.08,
):
    """Give each overlapping native marker to at most one series.

    The input is the extractor's pixel-space ``all_series`` structure.  A
    component is formed only across *different* series and only inside a
    marker-radius-aware distance.  A clear confidence/provenance winner keeps
    its point; a close tie removes all members from the component and records
    it as unresolved.  Thus this pass cannot invent a coordinate or silently
    duplicate one physical marker.  It mutates the series in the same style as
    the other pixel-space ownership helpers and returns ``(all_series, audit)``.
    """
    structural = spatial_ownership_guard(spec or {}) if spec is not None else {
        "enabled": True,
        "reason": "explicit_or_direct_call",
        "branches": None,
        "multi_series": len(all_series or []) >= 2,
        "all_filled": True,
        "ambiguity_group_count": 0,
        "crowded_region_count": 0,
    }
    use_pass = structural["enabled"] if enabled is None else bool(enabled)
    audit = {
        **structural,
        "enabled": bool(use_pass),
        "distance_fraction": float(distance_fraction),
        "confidence_margin": float(confidence_margin),
        "candidate_count": 0,
        "overlap_components": 0,
        "deduplicated": 0,
        "unresolved": 0,
        "components": [],
    }
    if not use_pass or not all_series or len(all_series) < 2:
        return all_series, audit

    by_index = {series.get("idx", i): series for i, series in enumerate(all_series)}
    nodes = []
    for series_position, series in enumerate(all_series):
        series_index = series.get("idx", series_position)
        marker_radius = float(series.get("r") or radius or 0.0)
        for point_position, point in enumerate(series.get("pts") or []):
            if len(point) < 2 or point[0] is None or point[1] is None:
                continue
            nodes.append(
                {
                    "series_index": series_index,
                    "series_position": series_position,
                    "point_position": point_position,
                    "point": point,
                    "radius": max(1.0, marker_radius),
                }
            )
    audit["candidate_count"] = len(nodes)
    if not nodes:
        return all_series, audit
    parent = list(range(len(nodes)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            if nodes[left]["series_index"] == nodes[right]["series_index"]:
                continue
            distance = float(np.hypot(
                nodes[left]["point"][0] - nodes[right]["point"][0],
                nodes[left]["point"][1] - nodes[right]["point"][1],
            ))
            threshold = float(distance_fraction) * max(nodes[left]["radius"], nodes[right]["radius"])
            if distance <= threshold:
                union(left, right)
    components = {}
    for index in range(len(nodes)):
        components.setdefault(find(index), []).append(index)
    components = [members for members in components.values() if len({nodes[i]["series_index"] for i in members}) > 1]
    audit["overlap_components"] = len(components)
    if not components:
        return all_series, audit

    remove = {series.get("idx", i): set() for i, series in enumerate(all_series)}
    for component_index, members in enumerate(components):
        def rank(node):
            point = node["point"]
            confidence = float(point[2]) if len(point) > 2 else 0.5
            overlap = bool(point[3]) if len(point) > 3 else False
            sure = bool(point[4]) if len(point) > 4 else confidence >= CONF_OK
            # Native/sure evidence dominates confidence; overlap is a
            # deterministic tie-breaker, never a reason to discard a clear
            # native candidate.
            return (int(sure), confidence, int(not overlap))

        ranked = sorted(members, key=lambda i: (-rank(nodes[i])[0], -rank(nodes[i])[1], -rank(nodes[i])[2], nodes[i]["series_index"], nodes[i]["point_position"]))
        winner = nodes[ranked[0]]
        runner = nodes[ranked[1]]
        winner_rank, runner_rank = rank(winner), rank(runner)
        confidence_gap = float(winner_rank[1] - runner_rank[1])
        clear = winner_rank[0] > runner_rank[0] or confidence_gap >= float(confidence_margin)
        labels = []
        for index in sorted({nodes[i]["series_index"] for i in members}):
            labels.append(str(by_index[index].get("spec", {}).get("label", index)))
        component_audit = {
            "component_index": component_index,
            "series_indices": sorted({nodes[i]["series_index"] for i in members}),
            "series_labels": labels,
            "candidate_pixels": [
                [round(float(nodes[i]["point"][0]), 2), round(float(nodes[i]["point"][1]), 2)]
                for i in members
            ],
            "winner": None,
            "runner_up": None,
            "confidence_gap": round(confidence_gap, 6),
            "action": "unresolved",
        }
        if clear:
            component_audit["winner"] = {
                "series_index": winner["series_index"],
                "point_position": winner["point_position"],
                "rank": list(winner_rank),
            }
            component_audit["runner_up"] = {
                "series_index": runner["series_index"],
                "point_position": runner["point_position"],
                "rank": list(runner_rank),
            }
            component_audit["action"] = "deduplicated_to_owner"
            for node_index in members:
                node = nodes[node_index]
                if node is not winner:
                    remove[node["series_index"]].add(node["point_position"])
                    audit["deduplicated"] += 1
        else:
            for node_index in members:
                node = nodes[node_index]
                remove[node["series_index"]].add(node["point_position"])
            audit["unresolved"] += 1
        audit["components"].append(component_audit)
    for series_position, series in enumerate(all_series):
        series_index = series.get("idx", series_position)
        drop = remove.get(series_index, set())
        if drop:
            series["pts"] = [point for point_position, point in enumerate(series.get("pts") or []) if point_position not in drop]
    return all_series, audit


# Backwards-friendly descriptive alias for callers that use the phrase from
# the calibration/audit terminology.
spatial_ownership_dedupe = global_spatial_ownership


def prune_expected_count_excess(
    all_series,
    spec,
    fx,
    fy,
    *,
    enabled=False,
    tolerance_fraction=0.15,
    minimum_tolerance=2,
):
    """Soft-prune native candidates above Agent02's expected x-range counts.

    Expected counts are visual estimates, not hard quotas.  A cap is therefore
    ``max(expected + minimum_tolerance, ceil(expected * (1 + fraction)))``.
    Candidates are removed from the weakest end of a deterministic quality
    ranking: distance from the coached curve/band, native/sure evidence,
    confidence, overlap flag, and finally local x-spacing.  The pass is only
    called for the guarded crowded adsorption-only filled-marker family; it
    deliberately does nothing for open-marker or single-series panels.
    """
    audit = {
        "enabled": bool(enabled),
        "tolerance_fraction": float(tolerance_fraction),
        "minimum_tolerance": int(minimum_tolerance),
        "series": [],
        "removed": 0,
    }
    if not enabled or not all_series:
        return all_series, audit
    chart_specs = {
        str(item.get("label")): item for item in (spec or {}).get("series", [])
    }
    ticks_x = (spec or {}).get("x", {}).get("ticks") or []
    ticks_y = (spec or {}).get("y", {}).get("ticks") or []
    chart_x_min = float(min(ticks_x)) if ticks_x else None
    chart_x_max = float(max(ticks_x)) if ticks_x else None
    y_span = abs(float(max(ticks_y) - min(ticks_y))) if ticks_y else 1.0
    y_span = max(y_span, 1e-9)

    def point_xy(point):
        try:
            return float(fx(point[0])), float(fy(point[1]))
        except (TypeError, ValueError, OverflowError):
            return None, None

    def curve_badness(series_spec, x, y):
        if x is None or y is None:
            return 1.0
        bands = series_spec.get("y_bands") or []
        covering = [band for band in bands if float(band.get("x_from", -np.inf)) <= x <= float(band.get("x_to", np.inf))]
        if covering:
            if any(float(band["y_min"]) <= y <= float(band["y_max"]) for band in covering):
                return 0.0
            gap = min(
                min(abs(y - float(band["y_min"])), abs(y - float(band["y_max"])))
                for band in covering
            )
            return min(1.0, gap / y_span)
        if chart_x_min is None or chart_x_max is None:
            return 0.5
        curve = series_gate.anchor_curve(series_spec, chart_x_min, chart_x_max)
        if curve is None or not curve[1] <= x <= curve[2]:
            return 0.5
        return min(1.0, abs(y - float(curve[0](x))) / y_span)

    for series_position, series in enumerate(all_series):
        series_index = series.get("idx", series_position)
        series_spec = dict(series.get("spec") or {})
        series_spec.update(chart_specs.get(str(series_spec.get("label")), {}))
        expected_ranges = []
        for item in series_spec.get("expected_counts") or []:
            try:
                x_from = float(item["x_from"])
                x_to = float(item["x_to"])
                expected = max(0, int(round(float(item["count"]))))
            except (KeyError, TypeError, ValueError):
                continue
            if x_from < x_to and expected > 0:
                cap = max(expected + int(minimum_tolerance), int(np.ceil(expected * (1.0 + float(tolerance_fraction)))))
                expected_ranges.append({"x_from": x_from, "x_to": x_to, "expected": expected, "cap": cap})
        if not expected_ranges:
            continue
        active = list(series.get("pts") or [])
        before = len(active)
        removed = []
        # Process ranges in declaration order.  This is deterministic and
        # naturally handles overlapping ranges because each later range sees
        # the survivors of the earlier one.
        for expected_range in expected_ranges:
            def in_range(point):
                x, _ = point_xy(point)
                return x is not None and expected_range["x_from"] <= x <= expected_range["x_to"]

            while sum(1 for point in active if in_range(point)) > expected_range["cap"]:
                ranged = [point for point in active if in_range(point)]
                ranked = []
                for point in ranged:
                    x, y = point_xy(point)
                    confidence = float(point[2]) if len(point) > 2 else 0.5
                    overlap = bool(point[3]) if len(point) > 3 else False
                    sure = bool(point[4]) if len(point) > 4 else confidence >= CONF_OK
                    others = [point_xy(other)[0] for other in ranged if other is not point]
                    nearest_dx = min((abs(x - other_x) for other_x in others if other_x is not None), default=float("inf"))
                    expected_spacing = (expected_range["x_to"] - expected_range["x_from"]) / max(1, expected_range["expected"] - 1)
                    spacing_bad = 1.0 if nearest_dx == float("inf") else min(1.0, nearest_dx / max(expected_spacing * 0.75, 1e-9))
                    curve_distance = curve_badness(series_spec, x, y)
                    # Larger is weaker.  Keep tuple components in the audit so
                    # a reviewer can see why an excess candidate was removed.
                    quality = (
                        2.5 * curve_distance
                        + 1.5 * (1.0 if not sure else 0.0)
                        + 1.0 * (1.0 - min(1.0, max(0.0, confidence)))
                        + 0.75 * (1.0 if overlap else 0.0)
                        + 0.5 * (1.0 - spacing_bad)
                    )
                    ranked.append((quality, curve_distance, sure, confidence, overlap, spacing_bad, x, y, point))
                # Stable pixel tie-breaks ensure reruns make identical choices.
                _, curve_distance, sure, confidence, overlap, spacing_bad, x, y, victim = max(
                    ranked,
                    key=lambda item: (item[0], -int(item[2]), -item[3], int(item[4]), -item[5], -(item[6] if item[6] is not None else float("inf")), -(item[7] if item[7] is not None else float("inf"))),
                )
                active.remove(victim)
                removed.append({
                    "x": round(float(x), 4) if x is not None else None,
                    "y": round(float(y), 4) if y is not None else None,
                    "px": [round(float(victim[0]), 2), round(float(victim[1]), 2)],
                    "reason": "expected_count_soft_cap_excess",
                    "range": {"x_from": expected_range["x_from"], "x_to": expected_range["x_to"]},
                    "expected": expected_range["expected"],
                    "cap": expected_range["cap"],
                    "quality": round(float(_), 6),
                    "curve_badness": round(float(curve_distance), 6),
                    "sure": bool(sure),
                    "confidence": round(float(confidence), 6),
                    "overlap": bool(overlap),
                    "spacing_quality": round(float(spacing_bad), 6),
                })
        series["pts"] = sorted(active)
        if before != len(active) or expected_ranges:
            audit["series"].append({
                "series_index": series_index,
                "series_label": series_spec.get("label"),
                "before": before,
                "after": len(active),
                "expected_ranges": expected_ranges,
                "removed": removed,
            })
            audit["removed"] += len(removed)
    return all_series, audit


def assign_ambiguous_series(
    all_series,
    raw_candidates,
    ambiguity_groups,
    lab,
    grey,
    plot_mask,
    radius,
    fx=None,
    fy=None,
    x_range=None,
    rescore_single_source=False,
):
    """Jointly assign native candidates for color-overlapping series.

    Candidate coordinates come from either an existing native marker detector
    or the color-independent grayscale/edge detector.  The latter is not a
    coordinate prediction: it is retained only when native marker-sized
    evidence is present.  When agent 02 coached the competing series with curves
    or y bands, the curve position decides first (``curve_owner``).  A close
    best/runner-up pair is deliberately left unresolved for Agent 03.
    """
    if not ambiguity_groups:
        return all_series, {"groups": [], "candidates": []}, []
    native_mask = marker_assignment.native_ink_mask(grey, lab, plot_mask)
    config = CFG.get("extract", {})
    weights = {
        "shape": config.get("shape_assignment_weight", 0.40),
        "color": config.get("color_assignment_weight", 0.20),
        "trajectory": config.get("trajectory_assignment_weight", 0.20),
    }
    margin_required = float(config.get("series_assignment_margin", 0.12))
    by_idx = {series["idx"]: series for series in all_series}
    audit = {"groups": [], "candidates": [], "margin_required": margin_required}
    unresolved = []

    for group in ambiguity_groups:
        indices = list(group["series_indices"])
        present = [by_idx[index] for index in indices if index in by_idx]
        if not present:
            continue
        group_radius = max([float(item.get("r") or radius) for item in present] or [radius])
        pool = []
        for index in indices:
            for point in raw_candidates.get(index, []):
                item = {
                    "px": [float(point[0]), float(point[1])],
                    "confidence": float(point[2]),
                    "geometry": {"source": "existing_native_candidate"},
                    "source": "existing_native_candidate",
                    "source_indices": [index],
                }
                near = next(
                    (
                        candidate
                        for candidate in pool
                        if np.hypot(
                            candidate["px"][0] - item["px"][0],
                            candidate["px"][1] - item["px"][1],
                        )
                        < 0.55 * group_radius
                    ),
                    None,
                )
                if near is None:
                    pool.append(item)
                else:
                    near["confidence"] = max(near["confidence"], item["confidence"])
                    near["source_indices"] = sorted(
                        set(near["source_indices"]) | {index}
                    )
        # The color passes supply a bounded native search pool for large
        # groups.  Scan for additional geometry-only candidates when that
        # pool is sparse; this avoids turning every long chart line into a
        # whole-panel candidate list while retaining the recovery path.
        native = (
            marker_assignment.native_marker_candidates(native_mask, group_radius)
            if len(pool) < 100
            else []
        )
        # Existing color passes already establish a conservative search
        # neighbourhood.  Native geometry is allowed to recover ownership
        # inside that neighbourhood without becoming a whole-plot line/text
        # detector; direct callers can still use the unrestricted helper.
        if pool:
            native = [
                item
                for item in native
                if any(
                    np.hypot(
                        item["px"][0] - candidate["px"][0],
                        item["px"][1] - candidate["px"][1],
                    )
                    <= 2.5 * group_radius
                    for candidate in pool
                )
            ]
        for item in native:
            near = next(
                (
                    candidate
                    for candidate in pool
                    if np.hypot(
                        candidate["px"][0] - item["px"][0],
                        candidate["px"][1] - item["px"][1],
                    )
                    < 0.55 * group_radius
                ),
                None,
            )
            if near is None:
                pool.append(item)
            else:
                near["confidence"] = max(near["confidence"], item["confidence"])
                near["geometry"] = item.get("geometry", near.get("geometry", {}))
                near["source"] = "native_geometry_and_color_candidate"

        assigned = {index: [] for index in indices}
        group_audit = dict(group)
        group_audit["candidate_count"] = len(pool)
        group_audit["assignment_margin"] = margin_required
        ordered_pool = sorted(pool, key=lambda item: (item["px"][0], item["px"][1]))
        disputed_ids = {
            id(candidate)
            for candidate in ordered_pool
            if len(candidate.get("source_indices", [])) != 1
        }
        if len(disputed_ids) > 160:
            # Deterministic bounded work on noisy charts: the weakest native
            # candidates are the ones most likely to have been lost by the
            # color gate.  The remainder stay unresolved rather than being
            # guessed by a smoothness or spacing rule.
            ranked_disputed = sorted(
                (candidate for candidate in ordered_pool if id(candidate) in disputed_ids),
                key=lambda candidate: (float(candidate.get("confidence", 0.5)), candidate["px"][0], candidate["px"][1]),
            )
            score_disputed_ids = {id(candidate) for candidate in ranked_disputed[:160]}
        else:
            score_disputed_ids = disputed_ids
        for candidate_index, candidate in enumerate(ordered_pool):
            # A candidate seen at the same native location by multiple
            # color passes is the disputed case that needs joint scoring.
            # Single-source candidates normally remain native observations
            # owned by their detector.  Crowded adsorption-only panels are an
            # exception: color passes can each produce one *different*
            # single-source owner for the same physical marker.  In that
            # guarded mode they must enter the same scorer as shared-source
            # candidates; preserving them here inflated ownership downstream.
            source_indices = candidate.get("source_indices", [])
            if (
                len(source_indices) == 1
                and source_indices[0] in assigned
                and not rescore_single_source
            ):
                owner = source_indices[0]
                assigned[owner].append(
                    (
                        candidate["px"][0], candidate["px"][1],
                        float(candidate.get("confidence", 0.5)), False, True,
                    )
                )
                audit["candidates"].append(
                    {
                        "group_id": group["group_id"],
                        "candidate_index": candidate_index,
                        "px": [round(float(value), 2) for value in candidate["px"]],
                        "source": candidate.get("source"),
                        "source_series_indices": source_indices,
                        "winner": {"series_index": owner, "series_label": by_idx[owner]["spec"].get("label"), "score": None, "components": None},
                        "runner_up": {"series_index": None, "series_label": None, "score": None, "components": None},
                        "score_margin": None,
                        "action": "preserved_native_candidate",
                    }
                )
                continue
            owner = curve_owner(candidate, present, fx, fy, x_range)
            if owner is not None:
                assigned[owner].append(
                    (candidate["px"][0], candidate["px"][1], float(candidate.get("confidence", 0.5)), False, True)
                )
                audit["candidates"].append(
                    {
                        "group_id": group["group_id"], "candidate_index": candidate_index,
                        "px": [round(float(value), 2) for value in candidate["px"]],
                        "source": candidate.get("source"), "source_series_indices": source_indices,
                        "winner": {"series_index": owner, "series_label": by_idx[owner]["spec"].get("label"),
                                   "score": None, "components": None},
                        "runner_up": {"series_index": None, "series_label": None, "score": None, "components": None},
                        "score_margin": None, "action": "assigned_by_agent02_curve",
                    }
                )
                continue
            if id(candidate) not in score_disputed_ids:
                unresolved.append(
                    {
                        "series_label": "ambiguous: " + ", ".join(str(by_idx[index]["spec"].get("label")) for index in indices),
                        "candidate_series_indices": indices,
                        "candidate_series_labels": [str(by_idx[index]["spec"].get("label")) for index in indices],
                        "series_index": None,
                        "candidate_index": candidate_index,
                        "column_px": round(float(candidate["px"][0]), 2),
                        "predicted_y_px": round(float(candidate["px"][1]), 2),
                        "px": [round(float(value), 2) for value in candidate["px"]],
                        "state": "unresolved",
                        "reason": "assignment_work_budget_exceeded",
                    }
                )
                audit["candidates"].append(
                    {
                        "group_id": group["group_id"], "candidate_index": candidate_index,
                        "px": [round(float(value), 2) for value in candidate["px"]],
                        "source": candidate.get("source"), "source_series_indices": source_indices,
                        "winner": {"series_index": None, "series_label": None, "score": None, "components": None},
                        "runner_up": {"series_index": None, "series_label": None, "score": None, "components": None},
                        "score_margin": None, "action": "unresolved",
                    }
                )
                continue
            scored = []
            # Keep the joint comparison bounded on very crowded charts.  The
            # nearest legend colors are supporting evidence only: native
            # shape/line/trajectory features still decide among this set, and
            # every source series remains included in the fallback path.
            iy, ix = map(int, map(round, candidate["px"]))
            sample = lab[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3].reshape(-1, 3)
            if sample.size:
                nearest_indices = sorted(
                    indices,
                    key=lambda index: float(np.linalg.norm(np.median(sample, axis=0) - by_idx[index]["col"])),
                )[: min(8, len(indices))]
            else:
                nearest_indices = indices
            nearest_indices = sorted(set(nearest_indices) | set(candidate.get("source_indices", [])))
            score_series = [by_idx[index] for index in nearest_indices]
            for series in score_series:
                points = raw_candidates.get(series["idx"], [])
                scores = marker_assignment.score_candidate(
                    candidate,
                    series,
                    lab,
                    grey,
                    native_mask,
                    group_radius,
                    points,
                    fx=fx,
                    fy=fy,
                    weights=weights,
                )
                scored.append((scores["total"], series["idx"], scores))
            scored.sort(key=lambda item: (-item[0], item[1]))
            winner_score, winner_index, winner_components = scored[0]
            runner_score, runner_index, runner_components = scored[1] if len(scored) > 1 else (0.0, None, None)
            score_margin = float(winner_score - runner_score)
            action = "assigned"
            if score_margin < margin_required:
                action = "unresolved"
                unresolved.append(
                    {
                        "series_label": "ambiguous: " + ", ".join(
                            str(by_idx[index]["spec"].get("label")) for index in indices
                        ),
                        "candidate_series_indices": indices,
                        "candidate_series_labels": [
                            str(by_idx[index]["spec"].get("label")) for index in indices
                        ],
                        "series_index": winner_index,
                        "candidate_index": candidate_index,
                        "column_px": round(float(candidate["px"][0]), 2),
                        "predicted_y_px": round(float(candidate["px"][1]), 2),
                        "px": [round(float(value), 2) for value in candidate["px"]],
                        "state": "unresolved",
                        "reason": "series_assignment_margin_below_threshold",
                    }
                )
            else:
                assigned[winner_index].append(
                    (
                        candidate["px"][0],
                        candidate["px"][1],
                        min(0.98, max(float(candidate.get("confidence", 0.5)), 0.55 + 0.35 * score_margin)),
                        False,
                        True,
                    )
                )
                if winner_index not in candidate.get("source_indices", []):
                    action = "reassigned"
            audit["candidates"].append(
                {
                    "group_id": group["group_id"],
                    "candidate_index": candidate_index,
                    "px": [round(float(value), 2) for value in candidate["px"]],
                    "source": candidate.get("source"),
                    "source_series_indices": candidate.get("source_indices", []),
                    "winner": {
                        "series_index": winner_index,
                        "series_label": by_idx[winner_index]["spec"].get("label"),
                        "score": round(float(winner_score), 6),
                        "components": winner_components,
                    },
                    "runner_up": {
                        "series_index": runner_index,
                        "series_label": by_idx[runner_index]["spec"].get("label") if runner_index is not None else None,
                        "score": round(float(runner_score), 6),
                        "components": runner_components,
                    },
                    "score_margin": round(score_margin, 6),
                    "action": action,
                }
            )
        for index in indices:
            by_idx[index]["pts"] = sorted(assigned[index])
            by_idx[index]["joint_assigned"] = True
        group_audit["assigned_counts"] = {
            str(index): len(assigned[index]) for index in indices
        }
        group_audit["unresolved_count"] = sum(
            1 for item in unresolved if item.get("candidate_series_indices") == indices
        )
        audit["groups"].append(group_audit)
    return all_series, audit, unresolved


def _runs_ext(idx):
    """Group consecutive indices -> (centre, length) of each run."""
    if len(idx) == 0:
        return []
    out, s, p = [], idx[0], idx[0]
    for v in idx[1:]:
        if v != p + 1:
            out.append(((s + p) / 2.0, p - s + 1))
            s = v
        p = v
    out.append(((s + p) / 2.0, p - s + 1))
    return out


def split_twins_by_lines(all_series, lab, plot_mask):
    """Adsorption/desorption twins painted on top of each other look like ONE marker, but each branch keeps its
    own thin line. Left and right of a marker, the series mask is read in a column: two thin runs on both sides
    (the two lines, clearly apart) whose interpolation crosses the marker's column at two heights = two markers.
    The marker's blob is then replaced by the two line crossings (confidence 0.6, overlap flag)."""
    n_split = 0
    for S in all_series:
        r = max(3.0, S["r"])
        if len(S["pts"]) < 4:
            continue
        m = cv2.bitwise_and(
            color_mask(lab, S["col"], S["tol"]), plot_mask
        )  # raw: no closing, a 1 px gap matters here
        H, W = m.shape
        pts = sorted(S["pts"])
        # the series' line width: thin runs in columns halfway between consecutive markers
        lws = []
        for a, b in zip(pts, pts[1:]):
            x = int(round((a[0] + b[0]) / 2))
            yc = (a[1] + b[1]) / 2
            if abs(b[0] - a[0]) < 3 * r or x < 1 or x >= W - 1:
                continue
            lo, hi = int(max(0, yc - 3 * r)), int(min(H - 1, yc + 3 * r))
            for c, n in _runs_ext(np.where(m[lo : hi + 1, x] > 0)[0]):
                if n <= 0.9 * r:
                    lws.append(n)
        lw = float(np.median(lws)) if len(lws) >= 4 else 2.0
        out = []
        for p in pts:
            cx, cy = p[0], p[1]
            # already has a same-series partner in the same column -> nothing to split
            if any(
                q is not p and abs(q[0] - cx) < 0.6 * r and 0 < abs(q[1] - cy) < 3.0 * r
                for q in pts
            ):
                out.append(p)
                continue
            sides = []
            for sgn in (-1, 1):
                found = None
                for k in (1.6, 1.9, 2.3):
                    x = int(round(cx + sgn * k * r))
                    if x < 1 or x >= W - 1:
                        break
                    lo, hi = int(max(0, cy - 3 * r)), int(min(H - 1, cy + 3 * r))
                    col = m[lo : hi + 1, x]
                    runs = [(c + lo, n) for c, n in _runs_ext(np.where(col > 0)[0])]
                    if any(n > 1.4 * r for _, n in runs):
                        break  # a marker sits in this column, not just lines
                    lines = []
                    for c, n in runs:
                        if n >= 1.8 * lw:  # two lines touching: split the run
                            lines += [c - (n - lw) / 2, c + (n - lw) / 2]
                        else:
                            lines.append(c)
                    if len(lines) == 2 and abs(lines[0] - lines[1]) >= 2.5:
                        found = (x, sorted(lines))
                        break
                    if len(lines) > 2:
                        break
                sides.append(found)
            if not all(sides):
                out.append(p)
                continue
            (xl, (l1, l2)), (xr, (r1, r2)) = sides
            sep_l, sep_r = l2 - l1, r2 - r1
            if abs(sep_l - sep_r) > 0.5 * max(sep_l, sep_r) + 1.5:
                out.append(p)
                continue  # the two sides do not describe the same pair of lines
            t = (cx - xl) / max(1e-6, (xr - xl))
            y1, y2 = l1 + t * (r1 - l1), l2 + t * (r2 - l2)
            if not (cy - 1.2 * r <= (y1 + y2) / 2 <= cy + 1.2 * r):
                out.append(p)
                continue  # lines cross somewhere else: not this marker's pair
            out.append((cx, float(y1), min(p[2], 0.6), True, False))
            out.append((cx, float(y2), min(p[2], 0.6), True, False))
            n_split += 1
        S["pts"] = sorted(out)
    return all_series, n_split


def reclaim_along_curves(all_series, fx, fy, r_px):
    """Mid-pressure recovery: fit a smooth curve through each series' confident points and hand it any
    low-confidence point of ANOTHER series (or unassigned candidate) that sits within one marker radius of
    the curve and farther from every other curve. Fixes markers lost between two similar colors."""
    curves = {}
    for S in all_series:
        sure = [(fx(p[0]), fy(p[1])) for p in S["pts"] if p[4]]
        if len(sure) >= 6:
            X = np.array([q[0] for q in sure])
            Y = np.array([q[1] for q in sure])
            try:
                curves[S["idx"]] = (np.polyfit(X, Y, 3), X.min(), X.max())
            except Exception:
                pass
    if len(curves) < 2:
        return all_series, 0
    r_units = r_px * abs(fy(1) - fy(0))

    def dist(idx, x, y):
        c, xmin, xmax = curves[idx]
        span = xmax - xmin
        if x < xmin - 0.1 * span or x > xmax + 0.1 * span:
            return np.inf
        return abs(y - np.polyval(c, x))

    moved = 0
    by_idx = {S["idx"]: S for S in all_series}
    for S in all_series:
        keep = []
        for cx, cy, conf, ov, sure in S["pts"]:
            if sure:
                keep.append((cx, cy, conf, ov, sure))
                continue
            x, y = fx(cx), fy(cy)
            d = {j: dist(j, x, y) for j in curves}
            best = min(d, key=d.get)
            if (
                best != S["idx"]
                and d[best] <= 1.0 * r_units
                and (S["idx"] not in d or d[S["idx"]] > 2.0 * r_units)
            ):
                by_idx[best]["pts"].append((cx, cy, max(conf, 0.85), ov, True))
                moved += 1
            else:
                keep.append((cx, cy, conf, ov, sure))
        S["pts"] = keep
    for S in all_series:
        S["pts"].sort()
    return all_series, moved


def trace_knot(all_series, lab, plot_mask, fx, fy, left_px):
    """Low-pressure knot: individual markers are painted on top of each other, but the series' line is still a
    band. Walk that band leftwards from the first confident marker and emit samples of its centreline every
    ~1 marker pitch. These are CURVE samples (source='curve', confidence 0.5), not measured points."""
    out = {}
    for S in all_series:
        sure = sorted([p for p in S["pts"] if p[4]])
        if len(sure) < 4:
            continue
        # start from the leftmost MEASURED marker (any confidence): the trace only covers what has no marker at all
        allp = sorted(S["pts"])
        sure = allp[:2] if allp[0][0] < sure[0][0] else sure
        r = max(3.0, S["r"])
        m = cv2.bitwise_and(color_mask(lab, S["col"], S["tol"]), plot_mask)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        x0, y0 = sure[0][0], sure[0][1]
        x1, y1 = sure[1][0], sure[1][1]
        slope = (y1 - y0) / max(1e-6, (x1 - x0))
        samples, x, y = [], x0, y0
        step = max(2.0, 1.2 * r)
        while x - step > left_px + 2:
            x -= step
            y_pred = y + slope * (-step)
            lo, hi = (
                int(max(0, y_pred - 3 * r)),
                int(min(m.shape[0] - 1, y_pred + 3 * r)),
            )
            col = m[lo : hi + 1, int(round(x))]
            ys = np.where(col > 0)[0]
            if len(ys) == 0:
                break
            # keep the run closest to the prediction
            runs = _runs(ys + lo)
            yc = min(runs, key=lambda v: abs(v - y_pred))
            if abs(yc - y_pred) > 3 * r:
                break
            slope = (yc - y) / (-step) if abs(yc - y) < 4 * r else slope
            y = yc
            samples.append((float(x), float(y)))
        if samples:
            out[S["idx"]] = samples
    return out


def draw_marker_outline(image, center, radius, marker, color, thickness=1):
    """Draw one series-shaped validation glyph and return one rendered glyph."""
    cx, cy = (int(round(center[0])), int(round(center[1])))
    radius = max(3, int(round(radius)))
    marker = str(marker or "circle").lower().replace("-", "_")
    if marker == "square":
        cv2.rectangle(
            image,
            (cx - radius, cy - radius),
            (cx + radius, cy + radius),
            color,
            thickness,
        )
    elif marker in {
        "triangle",
        "triangle_up",
        "triangle_down",
        "triangle_left",
        "triangle_right",
    }:
        points = {
            "triangle": [
                (cx, cy - radius),
                (cx - radius, cy + radius),
                (cx + radius, cy + radius),
            ],
            "triangle_up": [
                (cx, cy - radius),
                (cx - radius, cy + radius),
                (cx + radius, cy + radius),
            ],
            "triangle_down": [
                (cx - radius, cy - radius),
                (cx + radius, cy - radius),
                (cx, cy + radius),
            ],
            "triangle_left": [
                (cx - radius, cy),
                (cx + radius, cy - radius),
                (cx + radius, cy + radius),
            ],
            "triangle_right": [
                (cx + radius, cy),
                (cx - radius, cy - radius),
                (cx - radius, cy + radius),
            ],
        }[marker]
        cv2.polylines(image, [np.array(points, np.int32)], True, color, thickness)
    elif marker == "diamond":
        points = np.array(
            [
                (cx, cy - radius),
                (cx - radius, cy),
                (cx, cy + radius),
                (cx + radius, cy),
            ],
            np.int32,
        )
        cv2.polylines(image, [points], True, color, thickness)
    elif marker in {"cross", "plus"}:
        if marker == "plus":
            cv2.line(image, (cx - radius, cy), (cx + radius, cy), color, thickness)
            cv2.line(image, (cx, cy - radius), (cx, cy + radius), color, thickness)
        else:
            cv2.line(
                image,
                (cx - radius, cy - radius),
                (cx + radius, cy + radius),
                color,
                thickness,
            )
            cv2.line(
                image,
                (cx - radius, cy + radius),
                (cx + radius, cy - radius),
                color,
                thickness,
            )
    else:
        cv2.circle(image, (cx, cy), radius, color, thickness)
    return 1


def redraw_overlay(extraction, spec, out_png):
    """Write a clean source overlay and a separate audit overlay.

    The audit marks evidence only; all text/status stays in review-page chrome
    so dense plot pixels remain inspectable.
    """
    img = cv2.imread(spec["image"])
    x0, y0, x1, y1 = spec["panel_bbox"]
    crop = img[y0:y1, x0:x1].copy()
    styles = {
        str(item.get("label")): item
        for item in (spec.get("llm_spec") or {}).get("series", [])
    }
    clean = crop.copy()
    overlap_levels = {}
    glyph_count = 0
    for s in extraction["series"]:
        r = int(s.get("marker_radius_px", 6)) + 3
        marker = (styles.get(str(s.get("label"))) or {}).get("marker", "circle")
        glyph = s.get("marker_glyph") or {}
        use_glyph = glyph.get("confidence", 0) >= 0.6 and len(glyph.get("polygon") or []) >= 2
        glyph_hex = glyph.get("edge_hex") or glyph.get("color_hex") or "#28b428"
        glyph_bgr = tuple(int(glyph_hex[i:i + 2], 16) for i in (5, 3, 1)) if len(glyph_hex) == 7 else (40, 180, 40)
        for p in s["points"]:
            if not p.get("px") or p["px"][0] is None:
                continue
            c = (
                (255, 0, 255)
                if p.get("source") == "curve"
                else (0, 140, 255)  # orange: template placed where no marker was identifiable
                if p.get("source") == "template_fill"
                else (
                    (0, 255, 255)
                    if p["confidence"] >= CONF_OK and not p.get("overlap_flag")
                    else (0, 0, 255)
                )
            )
            key = (int(round(p["px"][0])), int(round(p["px"][1])))
            level = overlap_levels.get(key, 0)
            overlap_levels[key] = level + 1
            audit_radius = (3 if p.get("source") == "curve" else r) + 3 * level
            if use_glyph and p.get("source") != "curve":
                # the measured legend outline, scaled to the marker: true shape, direction and color
                legend_markers.draw_glyph(crop, p["px"], audit_radius, glyph, c, 1)
                legend_markers.draw_glyph(clean, p["px"], max(2.0, float(s.get("marker_radius_px", 6))) + 1.5,
                                          glyph, glyph_bgr, 1, halo=(255, 255, 255))
                glyph_count += 1
                continue
            glyph_count += draw_marker_outline(
                crop,
                p["px"],
                audit_radius,
                "circle" if p.get("source") == "curve" else marker,
                c,
                1,
            )
            draw_marker_outline(clean, p["px"], max(2, int(s.get("marker_radius_px", 6))), marker, (40, 180, 40), 1)
    cv2.imwrite(out_png, crop)
    target = pathlib.Path(out_png)
    stem = target.stem.replace("_overlay", "")
    cv2.imwrite(str(target.with_name(stem + "_clean_overlay.png")), clean)
    cv2.imwrite(str(target.with_name(stem + "_audit_overlay.png")), crop)
    return glyph_count


def shared_x_grid(all_series, fx, r_px, spec):
    """Isotherm instruments measure every series at the same pressure set-points, so markers line up in
    columns. Returns grid x positions (px) that are supported by >= 3 series' confident markers, merged with
    an optional hypothesis from the chart spec ("x_grid": values in axis units) validated the same way."""
    xs = []
    for S in all_series:
        for cx, cy, conf, ov, sure in S["pts"]:
            if sure:
                xs.append((cx, S["idx"]))
    if not xs:
        return [], {}
    xs.sort()
    tol = max(2.0, 0.5 * r_px)
    cols, cur = [], [xs[0]]
    for v in xs[1:]:
        if v[0] - cur[-1][0] <= tol:
            cur.append(v)
        else:
            cols.append(cur)
            cur = [v]
    cols.append(cur)
    supported = {}
    for c in cols:
        series = {j for _, j in c}
        if len(series) >= 3:
            supported[float(np.mean([x for x, _ in c]))] = len(series)
    # hypothesis from the spec (axis units) -> px, keep if the visible part is >= 60 % supported
    hyp = spec["x"].get("grid") or spec.get("x_grid")
    grid_px = sorted(supported)
    if hyp:
        inv = lambda v: (v - fx(0)) / (fx(1) - fx(0))
        hyp_px = [inv(v) for v in hyp]
        visible = [h for h in hyp_px if any(abs(h - g) <= 1.5 * tol for g in grid_px)]
        checkable = [h for h in hyp_px if h > min(grid_px, default=0)]
        if checkable and len(visible) / len(checkable) >= 0.6:
            merged = list(grid_px)
            for h in hyp_px:
                if not any(abs(h - g) <= 1.5 * tol for g in merged):
                    merged.append(h)
            grid_px = sorted(merged)
    return grid_px, supported


def fill_on_grid(all_series, grid_px, lab, plot_mask, fx, fy, r_px):
    """For every series and every grid column that has no measured marker, read the series' line band at that
    column, close to the curve interpolated from the measured markers. Output: (cx, cy) samples, source='grid'."""
    out = {}
    if not grid_px:
        return out
    for S in all_series:
        sure = sorted([(p[0], p[1]) for p in S["pts"] if p[4]])
        if len(sure) < 4:
            continue
        X = np.array([q[0] for q in sure])
        Y = np.array([q[1] for q in sure])
        m = cv2.bitwise_and(color_mask(lab, S["col"], S["tol"]), plot_mask)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        have = [p[0] for p in S["pts"]]
        samples = []
        for gx in grid_px:
            if any(abs(gx - hx) <= 1.6 * r_px for hx in have):
                continue  # a measured marker already sits on this column
            # local expectation: linear interpolation/extrapolation from the two nearest sure markers
            k = np.argsort(np.abs(X - gx))[:2]
            if len(k) < 2 or X[k[0]] == X[k[1]]:
                continue
            slope = (Y[k[1]] - Y[k[0]]) / (X[k[1]] - X[k[0]])
            y_pred = Y[k[0]] + slope * (gx - X[k[0]])
            dist_to_data = min(abs(X - gx))
            if dist_to_data > 40 * r_px:
                continue
            xi = int(round(gx))
            if xi < 0 or xi >= m.shape[1]:
                continue
            lo, hi = (
                int(max(0, y_pred - 3 * r_px)),
                int(min(m.shape[0] - 1, y_pred + 3 * r_px)),
            )
            col = m[lo : hi + 1, max(0, xi - 1) : xi + 2].max(1)
            ys = np.where(col > 0)[0]
            if len(ys) == 0:
                continue
            runs = _runs(ys + lo)
            yc = min(runs, key=lambda v: abs(v - y_pred))
            if abs(yc - y_pred) > 2.5 * r_px:
                continue
            ambiguous = len(runs) > 1
            samples.append((float(gx), float(yc), 0.6 if not ambiguous else 0.45))
        if samples:
            out[S["idx"]] = samples
    return out


def render_tick_check(spec, out_png):
    """Draw detected ticks and return their exact coordinates for Agent 02 to confirm.

    Agent 02 remains responsible for reading the printed value attached to each
    candidate.  Returning the detector coordinates avoids asking a vision model
    to reproduce sub-three-pixel geometry from a resized image.
    """
    img = cv2.imread(spec["image"])
    x0, y0, x1, y1 = spec["panel_bbox"]
    crop = img[y0:y1, x0:x1].copy()
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    left, bottom, top, right, XT, YT = detect_axis_geometry(grey, spec)
    for px, v in zip(XT, spec["x"]["ticks"]):
        cv2.line(crop, (int(px), bottom - 18), (int(px), bottom + 18), (0, 0, 255), 2)
        cv2.putText(
            crop,
            f"x={v:g}",
            (int(px) - 18, bottom + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )
    for py, v in zip(YT, sorted(spec["y"]["ticks"], reverse=True)):
        cv2.line(crop, (left - 18, int(py)), (left + 18, int(py)), (255, 0, 0), 2)
        cv2.putText(
            crop,
            f"y={v:g}",
            (max(0, left - 70), int(py) + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2,
        )
    cv2.imwrite(out_png, crop)
    return {
        "image_width_px": int(crop.shape[1]),
        "image_height_px": int(crop.shape[0]),
        "x": [
            {
                "candidate_pixel": int(px),
                "pixel_norm": float(px / crop.shape[1]),
                "proposed_value": float(value),
            }
            for px, value in zip(XT, spec["x"]["ticks"])
        ],
        "y": [
            {
                "candidate_pixel": int(py),
                "pixel_norm": float(py / crop.shape[0]),
                "proposed_value": float(value),
            }
            for py, value in zip(YT, sorted(spec["y"]["ticks"], reverse=True))
        ],
    }


def confirm_marker_near(spec, color_lab, r_px, px, py, search=None):
    """Pixel confirmation of an AI-proposed marker near (px, py): a blob of the series color must be there.
    Isolated blob of about marker size -> its centroid (sub-pixel). Merged blob -> the distance-transform peak
    nearest to the proposal. Nothing of that color -> None. Returns (cx, cy, confidence)."""
    img = cv2.imread(spec["image"])
    x0, y0, x1, y1 = spec["panel_bbox"]
    crop = img[y0:y1, x0:x1]
    lab = to_lab(crop)
    search = search or max(8.0, 1.5 * r_px)
    ya, yb = (
        int(max(0, py - search - r_px)),
        int(min(crop.shape[0], py + search + r_px + 1)),
    )
    xa, xb = (
        int(max(0, px - search - r_px)),
        int(min(crop.shape[1], px + search + r_px + 1)),
    )
    sub = lab[ya:yb, xa:xb]
    area_ref = np.pi * r_px**2
    for tol in (30.0, 42.0, 54.0):
        m = color_mask(sub, np.array(color_lab, np.float32), tol)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, cc, stats, cent = cv2.connectedComponentsWithStats(m)
        best = None
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 0.25 * area_ref:
                continue
            cx, cy = cent[i][0] + xa, cent[i][1] + ya
            if area <= 2.5 * area_ref:  # isolated marker: centroid
                d = np.hypot(cx - px, cy - py)
                conf = 0.85
            else:  # merged blob: nearest dt peak
                dt = cv2.distanceTransform(
                    (cc == i).astype(np.uint8) * 255, cv2.DIST_L2, 5
                )
                mx = ndimage.maximum_filter(dt, size=int(max(5, r_px)))
                peaks = np.argwhere((dt == mx) & (dt >= 0.5 * r_px))
                if len(peaks) == 0:
                    continue
                k = np.argmin(
                    [np.hypot(x_ + xa - px, y_ + ya - py) for y_, x_ in peaks]
                )
                cy, cx = peaks[k][0] + ya, peaks[k][1] + xa
                d = np.hypot(cx - px, cy - py)
                conf = 0.65
            if d <= search and (best is None or d < best[3]):
                best = (float(cx), float(cy), conf, d)
        if best:
            return best[0], best[1], best[2]
    return None


# ---------- main ----------


def drop_exact_duplicates(rows, tolerance_px=0.5):
    """One marker = one row: several passes can report the same centre for one series.

    Keeps the most confident row of each group of rows whose centres lie within ``tolerance_px``.
    Returns (rows, number dropped).
    """
    kept = []
    for row in sorted(rows, key=lambda item: -float(item.get("confidence") or 0)):
        if any(abs(row["px"][0] - other["px"][0]) <= tolerance_px and abs(row["px"][1] - other["px"][1]) <= tolerance_px
               for other in kept):
            continue
        kept.append(row)
    kept.sort(key=lambda item: (item["px"][0], item["px"][1]))
    return kept, len(rows) - len(kept)


def extract(spec, out_prefix=None):
    img = cv2.imread(spec["image"])
    x0, y0, x1, y1 = spec["panel_bbox"]
    crop = img[y0:y1, x0:x1]
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    left, bottom, top, right, XT, YT = detect_axis_geometry(grey, spec)
    x_scale = spec["x"].get("scale", "linear")
    y_scale = spec["y"].get("scale", "linear")
    x_anchor_evidence = validate_tick_anchors(
        XT, spec["x"]["ticks"], spec["x"].get("tick_anchors"), x_scale, "x", grey.shape[1]
    )
    y_anchor_evidence = validate_tick_anchors(
        YT, sorted(spec["y"]["ticks"], reverse=True), spec["y"].get("tick_anchors"), y_scale, "y", grey.shape[0]
    )
    fx, sx, rx, ax, bx = calibrate(XT, spec["x"]["ticks"], x_scale)
    fy, sy, ry, ay, by = calibrate(
        YT, sorted(spec["y"]["ticks"], reverse=True), y_scale
    )
    x_residual_px = rx / max(abs(ax), 1e-12)
    y_residual_px = ry / max(abs(ay), 1e-12)

    lab = to_lab(crop)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lx0, ly0, lx1, ly1 = spec["legend_bbox"]
    ph, pw = grey.shape
    complex_regions = list(spec.get("complex_regions") or [])
    # Region coordinates are accepted only after calibration.  In particular a
    # low-pressure label means low calibrated x, not a low image row.
    accepted_regions, rejected_regions = dense_regions.validate_complex_regions(
        complex_regions,
        (left, top, right, bottom),
        fx,
        (
            spec["x"].get("minimum", min(spec["x"]["ticks"])),
            spec["x"].get("maximum", max(spec["x"]["ticks"])),
        ),
        crop.shape,
    )
    complex_boxes = [tuple(item["bbox_px"]) for item in accepted_regions]
    dense_ring_rois = [
        tuple(item["bbox_px"]) for item in accepted_regions
        if item.get("kind") in {"dense_markers", "overlapping_markers"}
    ]
    complex_passes = max(
        [int(region.get("python_passes", 2)) for region in complex_regions] or [1]
    )
    base_tolerance_factors = (0.6, 0.8, 1.0)
    extra_tolerance_factors = (
        tuple(float(value) for value in np.linspace(0.45, 1.15, min(6, complex_passes)))
        if complex_boxes
        else ()
    )
    direct_labels = uses_direct_series_labels(spec)
    if direct_labels:
        # Direct annotations contain text but no representative marker glyph.
        # Sampling that area teaches the detector letter shapes and black text
        # as series colors.  Structural analysis already supplies the colors.
        colors = [
            color_from_hex(series.get("color_hex")) for series in spec["series"]
        ]
        shapes = [None] * len(colors)
        symbol_boxes = []
        spreads = [0.0] * len(colors)
        legend_reader = "declared_colors_direct_labels"
    else:
        pad_x, pad_y = (
            int(0.06 * pw),
            int(0.06 * ph),
        )  # tolerate a loose legend box from the LLM
        sample_box = [
            max(0, lx0 - pad_x),
            max(0, ly0 - pad_y),
            min(pw, lx1 + pad_x),
            min(ph, ly1 + pad_y),
        ]
        # Declared colors (agent 01 / agent 02) make the legend readable row by row, in legend
        # order; the blob reader is the fallback when colors are missing or do not match.
        by_rows = legend_rows_by_declared_color(crop, spec["legend_bbox"], spec["series"])
        if by_rows is not None:
            colors, shapes, symbol_boxes, spreads = by_rows
            legend_reader = "rows_by_declared_color"
        else:
            colors, shapes, symbol_boxes, spreads = legend_colors(
                crop,
                sample_box,
                len(spec["series"]),
                inner_bbox=spec["legend_bbox"],
            )
            legend_reader = "blobs"
    # Each legend symbol as a vector glyph (shape, direction, fill, colors); exact PDF paths when
    # the figure is a vector drawing. The overlay draws these outlines; agent 02 checks declarations.
    glyphs = (
        legend_markers.describe_legend(
            crop, shapes, symbol_boxes, colors,
            vector_marks=legend_markers.load_panel_marks(spec.get("vector_marks")),
        )
        if symbol_boxes
        else [None] * len(spec["series"])
    )
    # adaptive color tolerance: generous, but never more than 45% of the closest pair of series colors
    if len(colors) > 1:
        dmin = min(
            np.linalg.norm(a - b)
            for i, a in enumerate(colors)
            for b in colors[i + 1 :]
        )
        tol = float(min(45.0, max(COLOR_TOL, 0.45 * dmin)))
    else:
        tol = 45.0
    result_tol = tol

    plot_mask = np.zeros(grey.shape, np.uint8)
    plot_mask[top + 2 : bottom - 1, left + 2 : right - 1] = 255
    lx0, ly0, lx1, ly1 = spec["legend_bbox"]
    # A direct-label chart has no legend glyphs.  Its ``legend_bbox`` is only
    # a structural hint about the annotation area; masking it would erase real
    # plateau markers and teach the dense-template learner a false absence.
    if not direct_labels:
        plot_mask[ly0:ly1, lx0:lx1] = 0
    for bx0, by0, bx1, by1 in spec.get("mask_bboxes", []):  # insets, annotations, etc.
        plot_mask[by0:by1, bx0:bx1] = 0
    for (
        bx0,
        by0,
        bx1,
        by1,
    ) in symbol_boxes:  # every legend symbol, wherever the box was drawn
        plot_mask[max(0, by0) : by1, max(0, bx0) : bx1] = 0
    region_strategies.mask_ignored(spec, plot_mask)  # agent 02: regions no series may use
    # refine each series color on the plot's own markers, and set a per-series tolerance from the measured noise
    tols, refined = [], []
    use_refine = bool(CFG["extract"].get("refine_colors_on_plot", False))
    for c_seed, sp_ in zip(colors, spreads):
        c_ref, sp_ref = (
            refine_color(lab, plot_mask, c_seed) if use_refine else (c_seed, None)
        )
        refined.append(c_ref)
        noise = sp_ref if sp_ref is not None else sp_
        tols.append(
            float(
                min(
                    CFG["extract"].get("max_color_tolerance", 45),
                    max(tol, 1.4 * noise),
                )
            )
        )
    colors = refined
    for i, series in enumerate(spec["series"]):  # agent 02 may set a per-series tolerance
        if series.get("color_tolerance") is not None:
            tols[i] = float(min(45.0, max(15.0, float(series["color_tolerance"]))))
    # agent 02 pointed at one clean marker of a series: its plot color beats the legend reading
    sample_color_audit = region_strategies.apply_sample_colors(spec, colors, lab, plot_mask)
    ambiguity_groups = marker_assignment.ambiguity_groups(
        colors,
        spec["series"],
        CFG["extract"].get("color_ambiguity_distance_lab", 28),
    )
    ownership_guard = spatial_ownership_guard(
        spec,
        ambiguity_groups=ambiguity_groups,
        complex_regions=accepted_regions,
    )
    ambiguous_indices = {
        index
        for group in ambiguity_groups
        for index in group["series_indices"]
    }

    result = {
        "calibration": {
            "px_per_unit_x": 1 / sx,
            "px_per_unit_y": 1 / sy,
            "unit_per_px_x": sx,
            "unit_per_px_y": sy,
            "axis_models": {
                "x": {"scale": x_scale, "slope": ax, "intercept": bx},
                "y": {"scale": y_scale, "slope": ay, "intercept": by},
            },
            "tick_fit_resid_x": rx,
            "tick_fit_resid_y": ry,
            "tick_fit_resid_x_px": x_residual_px,
            "tick_fit_resid_y_px": y_residual_px,
            "frame_px": [left, top, right, bottom],
            "x_ticks_px": XT,
            "y_ticks_px": YT,
            "tick_anchors": {"x": x_anchor_evidence, "y": y_anchor_evidence},
            "color_tolerance_lab": result_tol,
            "color_tolerance_per_series": tols,
            "base_color_passes": len(base_tolerance_factors),
            "complex_region_passes": complex_passes if complex_boxes else 0,
            "complex_region_count": len(complex_boxes),
            "complex_regions_accepted": accepted_regions,
            "complex_regions_rejected": rejected_regions,
            "legend_reader": legend_reader,
            "sample_colors": sample_color_audit,
            "legend_markers": [
                {"label": series.get("label"), "legend_box_px": [int(v) for v in box], **glyph}
                for series, box, glyph in zip(spec["series"], symbol_boxes, glyphs)
                if glyph
            ],
        },
        "series": [],
    }
    overlay = crop.copy()
    # Keep every declared series while working in pixel space: excluded curves
    # can be necessary to separate color/shape twins.  Only policy-eligible
    # CO2 series are emitted below as coordinate-bearing extraction results.
    all_series = []
    # pass 1: per-series radius estimates; markers of one chart are all the same size, so the chart-level
    # median is forced on every series in pass 2 (a series whose own estimate drifted on a band or a fused block
    # is thereby corrected)
    r_force = {}  # idx -> radius forced in pass 2 (series whose own estimate drifted)
    pass1 = []
    for _pass in (1, 2):
        r_seen = []
        all_series = []
        for idx, (s, col) in enumerate(zip(spec["series"], colors)):
            if _pass == 2 and idx not in r_force:
                all_series.append(pass1[idx])
                r_seen.append(pass1[idx]["r"])
                continue
            r_chart = r_force.get(idx)
            tol_i = tols[idx]
            # multi-tolerance union: occluded/antialiased markers appear at different color tolerances
            pts, r = [], 0.0
            # marker radius from the legend symbol (same glyph, same size) — robust where merged blobs inflate estimates
            sh = shapes[idx]
            # area-equivalent radius of the glyph (a triangle's bounding box is much bigger than its area)
            if sh is not None and max(sh.shape) >= 5:
                core_glyph = cv2.erode(
                    (sh > 0).astype(np.uint8), np.ones((3, 3), np.uint8)
                )  # strip the JPEG halo
                r_legend = (
                    float(np.sqrt(max(1.0, float(core_glyph.sum())) / np.pi)) + 1.0
                )
            else:
                r_legend = None
            r_fixed = None
            pass_factors = [(factor, False) for factor in base_tolerance_factors]
            base_keys = {round(factor, 4) for factor in base_tolerance_factors}
            pass_factors += [
                (factor, True)
                for factor in extra_tolerance_factors
                if round(factor, 4) not in base_keys
            ]
            pass_factors.sort(key=lambda item: item[0], reverse=True)
            for factor, targeted in pass_factors:
                t = tol_i * factor
                m = cv2.bitwise_and(color_mask(lab, col, t), plot_mask)
                m = cv2.morphologyEx(
                    m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
                )  # heal JPEG speckle
                p_t, r_t = marker_centres(
                    m,
                    hsv[..., 1],
                    r_hint=r_fixed or r_chart or r_legend,
                    template=sh,
                    force_hint=(r_fixed is not None) or (r_chart is not None),
                )
                if (
                    r_fixed is None
                    and r_chart is None
                    and r_legend
                    and abs(r_t - r_legend) > 0.08 * r_legend
                ):
                    # the plot's own estimate (distance-transform peak = inradius, small for triangles/stars) and the
                    # legend glyph's (area-equivalent) disagree: keep whichever the acceptance tests like better
                    p_l, r_l = marker_centres(
                        m, hsv[..., 1], r_hint=r_legend, template=sh, force_hint=True
                    )
                    if len(p_l) > 1.15 * len(p_t):
                        p_t, r_t = p_l, r_l
                if r_fixed is None:
                    r_fixed = r_t
                r = max(r, r_t)
                for cx, cy, conf, ov in p_t:
                    if targeted and not any(
                        xa <= cx <= xb and ya <= cy <= yb
                        for xa, ya, xb, yb in complex_boxes
                    ):
                        continue
                    j = next(
                        (
                            k
                            for k, q in enumerate(pts)
                            if abs(q[0] - cx) < 0.8 * r_t and abs(q[1] - cy) < 0.8 * r_t
                        ),
                        None,
                    )
                    if j is None:
                        pts.append((cx, cy, conf, ov))
                    elif conf > pts[j][2]:
                        pts[j] = (cx, cy, conf, ov)
            marker_name = str(s.get("marker", "")).lower().replace("-", "_")
            hough_outline_only = False
            dense_circle_audit = []
            if (
                direct_labels
                and s.get("marker_fill") == "open"
                and marker_name in {"circle", "o"}
            ):
                # A narrow declared-color mask preserves the hollow ring and
                # avoids cross-series antialiasing at shared baselines.
                circle_mask = cv2.bitwise_and(
                    color_mask(lab, col, min(20.0, 0.45 * tol_i)),
                    plot_mask,
                )
                # Dense templates run before Hough proposals.  A Hough centre
                # can sit between two vertically stacked rings; it must never
                # replace a stronger 2-D native ring match merely because the
                # chart's expected count made the Hough pass look "complete".
                ring_radius = max(3.0, min(12.0, float(r or r_legend or 6.0)))
                dense_mask = cv2.bitwise_and(color_mask(lab, col, tol_i), plot_mask)
                for roi in dense_ring_rois:
                    candidates, audit = dense_circles.detect(dense_mask, pts, roi, ring_radius)
                    additions, matched = dense_circles.merge_2d(pts, candidates, ring_radius)
                    for cx, cy, score in additions:
                        # A template peak is native-supported but may still be
                        # partly fused.  Keep a measured confidence below 0.9
                        # and let overlap evidence route ambiguous cases to review.
                        pts.append((cx, cy, min(0.9, 0.55 + 0.35 * score), False))
                    dense_circle_audit.append({
                        "roi_px": [round(float(v), 2) for v in roi],
                        **audit,
                        "matched_existing": int(len(matched)),
                        "native_supported_additions": int(len(additions)),
                    })
                circle_pts, circle_radius = hollow_circle_centres(
                    circle_mask,
                    s.get("n_markers_estimate"),
                    right - left,
                )
                r = max(r, circle_radius)
                hough_outline_only = bool(circle_pts)
                for cx, cy, conf, _ in circle_pts:
                    j = next(
                        (
                            k
                            for k, q in enumerate(pts)
                            if np.hypot(q[0] - cx, q[1] - cy) < 1.5 * max(r, 3.0)
                        ),
                        None,
                    )
                    if j is None:
                        # Annulus coverage alone is a proposal, not enough to
                        # certify a dense marker.  Preserve it as review-only
                        # fallback evidence rather than a high-confidence row.
                        pts.append((cx, cy, min(float(conf), 0.55), True))
            pts.sort()
            # color twins: other series whose color is within 45 Lab units -> decide by marker shape
            twins = [
                j
                for j, c2 in enumerate(colors)
                if j != idx and np.linalg.norm(c2 - col) < tol_i + 6
            ]
            if twins and pts:
                m_full = cv2.bitwise_and(color_mask(lab, col, tol_i), plot_mask)
                maps = shape_score_maps(
                    m_full, [shapes[idx]] + [shapes[j] for j in twins]
                )
                if maps[0] is not None:
                    kept = []
                    for cx, cy, conf, ov in pts:
                        yy, xx = int(round(cy)), int(round(cx))
                        win = lambda mp: (
                            mp[max(0, yy - 2) : yy + 3, max(0, xx - 2) : xx + 3].max()
                            if mp is not None
                            else -1
                        )
                        mine = win(maps[0])
                        others = max(win(mp) for mp in maps[1:])
                        kept.append(
                            (cx, cy, conf, ov, mine >= others + 0.05)
                        )  # confident by shape?
                    pts = kept
            pts = [
                (pp[0], pp[1], pp[2], pp[3], pp[4] if len(pp) > 4 else True)
                for pp in pts
            ]
            all_series.append(
                {
                    "idx": idx,
                    "spec": s,
                    "col": col,
                    "shape": shapes[idx],
                    "pts": pts,
                    "r": r,
                    "twins": twins,
                    "tol": tol_i,
                    "line_width": float(s.get("line_width_px") or 1.0),
                    "hough_outline_only": hough_outline_only,
                    "dense_circle_audit": dense_circle_audit,
                }
            )
            r_seen.append((r, r_legend))
        if _pass == 1:
            pass1 = list(all_series)
            # markers of one chart share one size: each series' measured radius should be the same fraction of its
            # legend glyph radius. A series far from the chart-wide ratio (fused blocks, a band) gets the ratio instead.
            ratios = [rr / rl for rr, rl in r_seen if rr > 0 and rl]
            if len(ratios) >= 3:
                med = float(np.median(ratios))
                for i, (rr, rl) in enumerate(r_seen):
                    if rl and (rr <= 0 or abs(rr / rl - med) > 0.35):
                        r_force[i] = rl * med
            if not r_force:
                break

    # Preserve the full native candidate pool before any legacy twin settling
    # can discard a point.  The ambiguity scorer must be able to reconsider a
    # candidate for an excluded N2 series as well as for CO2.
    raw_candidates = {
        series["idx"]: [tuple(point) for point in series["pts"]]
        for series in all_series
    }

    # Without real legend glyphs there is no shape evidence for the twin
    # resolver.  Direct-label series already have independently declared
    # colors, so pooling them would let one series steal valid points from the
    # other based on blank marker interiors.
    settled_hypotheses = []
    if not direct_labels:
        all_series = settle_twins(
            all_series,
            fx,
            fy,
            lab=lab,
            plot_mask=plot_mask,
            tol=tol,
            shapes=shapes,
            x_ticks_vals=spec["x"]["ticks"],
            skip_indices=None if ownership_guard["enabled"] else ambiguous_indices,
            global_pool=bool(ownership_guard["enabled"]),
        )
        if ownership_guard["enabled"]:
            labels_by_index = {
                series["idx"]: str(series.get("spec", {}).get("label"))
                for series in all_series
            }
            for hypothesis in getattr(settle_twins, "hypotheses", []):
                indices = list(hypothesis.get("candidate_series_indices") or [])
                px = list(hypothesis.get("px") or [None, None])
                settled_hypotheses.append({
                    "series_label": "ambiguous: " + ", ".join(labels_by_index.get(index, str(index)) for index in indices),
                    "series_index": None,
                    "candidate_series_indices": indices,
                    "candidate_series_labels": [labels_by_index.get(index, str(index)) for index in indices],
                    "column_px": px[0],
                    "predicted_y_px": px[1],
                    "px": px,
                    "state": "unresolved",
                    "reason": hypothesis.get("reason", "multi_series_overlap_hypothesis"),
                })
    r_med = float(np.median([S["r"] for S in all_series if S["r"] > 0] or [6.0]))
    all_series, n_reclaimed = reclaim_along_curves(all_series, fx, fy, r_med)
    branches = str((spec.get("llm_spec") or {}).get("branches") or "unknown")
    split_branches = (
        CFG["extract"].get("split_twins_by_lines", True)
        and not direct_labels
        and branches == "adsorption_and_desorption"
    )
    all_series, n_split = (
        split_twins_by_lines(all_series, lab, plot_mask)
        if split_branches
        else (all_series, 0)
    )
    result["calibration"]["twins_split_by_lines"] = n_split
    result["calibration"]["line_split_enabled"] = split_branches
    assignment_candidates = (
        {series["idx"]: [tuple(point) for point in series["pts"]] for series in all_series}
        if ownership_guard["enabled"]
        else raw_candidates
    )
    all_series, assignment_audit, assignment_unresolved = assign_ambiguous_series(
        all_series,
        assignment_candidates,
        ambiguity_groups,
        lab,
        grey,
        plot_mask,
        r_med,
        fx=fx,
        fy=fy,
        x_range=(float(min(spec["x"]["ticks"])), float(max(spec["x"]["ticks"]))),
        rescore_single_source=False,
    )
    all_series = pixel_color_check(all_series, lab, skip_indices=ambiguous_indices)
    # same-color series: points go to the series whose curve (agent anchors / agent 02 band) explains them
    series_gate_audit = (
        series_gate.apply(all_series, spec, colors, fx, fy)
        if CFG["extract"].get("series_gate", True)
        else []
    )
    result["calibration"]["series_gate"] = series_gate_audit
    result["calibration"]["color_ambiguity_groups"] = ambiguity_groups
    result["calibration"]["series_assignment_audit"] = assignment_audit
    result["calibration"]["series_assignment_margin"] = float(
        CFG["extract"].get("series_assignment_margin", 0.12)
    )
    result["calibration"]["reclaimed_points"] = n_reclaimed
    # Dense resolution is intentionally independent of the legacy curve/grid
    # fallbacks.  It only promotes a native distance-transform marker core.
    dense_slots = list(assignment_unresolved) + settled_hypotheses
    dense_circle_unresolved = []
    dense_circle_audit = []
    for series in all_series:
        for entry in series.get("dense_circle_audit") or []:
            audit = {"series_label": series["spec"].get("label"), **entry}
            dense_circle_audit.append(audit)
            # Ink with no distinct ring maximum is a fused/unsupported
            # region, not proof that there were no markers.  Keep a regional
            # ledger entry without inventing a y coordinate or a count.
            fused = (entry.get("rejected") or {}).get("solid_core", 0) > 0
            no_distinct = not entry.get("validated_centers", 0)
            if (entry.get("searches", 0) and entry.get("roi_ink_pixels", 0) >= max(8, int(np.pi * max(series.get("r") or 3, 3)))
                    and (no_distinct or fused)):
                dense_circle_unresolved.append({
                    "series_label": series["spec"].get("label"),
                    "series_index": series["idx"],
                    "state": "unresolved",
                    "reason": ("dense_ring_search_no_distinct_native_centres" if no_distinct
                               else "dense_ring_search_fused_native_ink"),
                    "roi_px": entry.get("roi_px"),
                })
    dense_slots.extend(dense_circle_unresolved)
    result["calibration"]["dense_circle_detector"] = {
        "regions": dense_circle_audit,
        "native_supported_additions": int(sum(item.get("native_supported_additions", 0) for item in dense_circle_audit)),
        "unresolved_fused_regions": int(len(dense_circle_unresolved)),
    }
    dense_resolved = []
    dense_columns = []
    resolvable_regions = [
        r for r in accepted_regions
        if r.get("kind") in {"dense_markers", "overlapping_markers", "thick_connecting_lines", "crossing"}
    ]
    if CFG["extract"].get("dense_region_resolver", True):
        x_limits = (
            spec["x"].get("minimum", min(spec["x"]["ticks"])),
            spec["x"].get("maximum", max(spec["x"]["ticks"])),
        )
        low_pressure_strip = dense_regions.infer_low_pressure_roi(
            (left, top, right, bottom),
            all_series,
            fraction=float(CFG["extract"].get("dense_low_pressure_fraction", 0.20)),
            axis_scale=x_scale,
            axis_limits=x_limits,
            slope=ax,
            intercept=bx,
        )
        # Always inspect the calibrated low-pressure strip at every plot y.
        # Subtract its x span from overlapping hand-marked ROIs so the same
        # columns do not create duplicate slots; keep any higher-x remainder.
        strip_x0, _, strip_x1, _ = low_pressure_strip
        rois = []
        for region in resolvable_regions:
            roi_x0, roi_y0, roi_x1, roi_y1 = map(float, region["bbox_px"])
            if roi_x1 <= strip_x0 or roi_x0 >= strip_x1:
                rois.append([roi_x0, roi_y0, roi_x1, roi_y1])
                continue
            if roi_x0 < strip_x0:
                rois.append([roi_x0, roi_y0, strip_x0, roi_y1])
            if roi_x1 > strip_x1:
                rois.append([strip_x1, roi_y0, roi_x1, roi_y1])
        rois = [roi for roi in rois if roi[0] < roi[2] and roi[1] < roi[3]]
        if low_pressure_strip[0] < low_pressure_strip[2]:
            rois.append(low_pressure_strip)
        if not rois:
            rois = [dense_regions.infer_low_pressure_roi((left, top, right, bottom), all_series)]
        for region_index, roi in enumerate(rois):
            resolved, slots, columns = dense_regions.resolve_dense_region(
                all_series,
                lab,
                roi,
                r_med,
                native_mask=marker_assignment.native_ink_mask(grey, lab, plot_mask),
                geometry_indices=ambiguous_indices,
            )
            for slot in slots:
                slot["region_index"] = region_index
            dense_resolved.extend(resolved)
            dense_slots.extend(slots)
            dense_columns.extend(columns)
        by_idx = {series["idx"]: series for series in all_series}
        for item in dense_resolved:
            series = by_idx[item["series_index"]]
            if not any(
                np.hypot(p[0] - item["px"][0], p[1] - item["px"][1]) < 0.8 * r_med
                for p in series["pts"]
            ):
                series["pts"].append(
                    (item["px"][0], item["px"][1], item["confidence"], True, True)
                )
                series["pts"].sort()
        result["calibration"]["dense_region_resolver"] = {
            "roi_px": [[round(float(v), 2) for v in roi] for roi in rois],
            "low_pressure_strip_px": [round(float(v), 2) for v in low_pressure_strip],
            "columns": len(dense_columns),
            "native_supported_additions": len(dense_resolved),
        }
    # Crowded adsorption-only charts can report one physical filled marker
    # through two color passes.  Resolve ownership after native/dense passes,
    # but before template/grid fallbacks, so only measured coordinates enter
    # this global deduplicator and unresolved ties remain explicit.
    all_series, ownership_audit = global_spatial_ownership(
        all_series,
        spec=spec,
        enabled=bool(ownership_guard["enabled"]),
        radius=r_med,
    )
    # Preserve the exact preflight trigger (including accepted dense regions)
    # rather than the helper's compact standalone-spec reconstruction.
    ownership_audit.update({
        "branches": ownership_guard.get("branches"),
        "multi_series": ownership_guard.get("multi_series"),
        "all_filled": ownership_guard.get("all_filled"),
        "ambiguity_group_count": ownership_guard.get("ambiguity_group_count"),
        "crowded_region_count": ownership_guard.get("crowded_region_count"),
        "reason": ownership_guard.get("reason"),
    })
    ownership_audit["global_twin_pool"] = bool(ownership_guard["enabled"])
    ownership_audit["joint_assignment_candidate_source"] = (
        "globally_settled_candidates" if ownership_guard["enabled"] else "raw_candidates"
    )
    ownership_audit["rescore_single_source"] = False
    all_series, count_pruning_audit = prune_expected_count_excess(
        all_series,
        spec,
        fx,
        fy,
        enabled=bool(ownership_guard["enabled"]),
    )
    ownership_audit["expected_count_pruning"] = count_pruning_audit
    result["calibration"]["global_spatial_ownership"] = ownership_audit
    # Crowded strips: put each series' own marker template on the slots the resolver could not
    # confirm (template_fit when its color is under the template, else template_fill).
    template_points = {}
    if CFG["extract"].get("template_fill", True) and dense_slots:
        plot_lab = lab[plot_mask > 0]
        background_lab = np.median(plot_lab[:: max(1, len(plot_lab) // 20000)], axis=0) if len(plot_lab) else None
        template_points, template_audit = template_fill.fill_slots(
            crop, dense_slots, all_series, glyphs, background_lab,
            min_coverage=float(CFG["extract"].get("template_fit_min_coverage", 0.55)),
            fill_unresolved=bool(CFG["extract"].get("template_fill_unresolved", True)),
            margin=float(CFG["extract"].get("template_fit_margin", 0.15)),
        )
        result["calibration"]["template_fill"] = {
            "template_fit": sum(slot["state"] == "template_fit" for slot in template_audit),
            "template_fill": sum(slot["state"] == "template_fill" for slot in template_audit),
            "slots": template_audit,
        }
        dense_slots = [slot for slot in dense_slots if not str(slot.get("state", "")).startswith("template_")]
    # agent 02 region strategies: ignore / split_filled_open / sample_band_at_columns
    band_points, strategy_audit = region_strategies.apply(
        all_series, spec, lab, {"slope": ax, "intercept": bx, "scale": x_scale}, r_med, glyphs
    )
    result["calibration"]["region_strategies"] = strategy_audit
    eligible_labels = {str(series.get("label")) for series in spec.get("series", []) if series.get("extract", True)}
    actionable_slots = [slot for slot in dense_slots if str(slot.get("series_label")) in eligible_labels]
    excluded_slots = [slot for slot in dense_slots if str(slot.get("series_label")) not in eligible_labels]
    result["unresolved_slots"] = dense_regions.unresolved_summary(actionable_slots)
    result["excluded_series_context"] = dense_regions.unresolved_summary(excluded_slots)
    result["unresolved_total"] = result["unresolved_slots"]["total"]
    knot = (
        trace_knot(all_series, lab, plot_mask, fx, fy, left)
        if CFG["extract"].get("trace_curve_in_knot", True)
        else {}
    )
    grid_px, grid_support = (
        shared_x_grid(all_series, fx, r_med, spec)
        if CFG["extract"].get("fill_shared_x_grid", True)
        else ([], {})
    )
    grid_fill = (
        fill_on_grid(all_series, grid_px, lab, plot_mask, fx, fy, r_med)
        if grid_px
        else {}
    )
    result["calibration"]["x_grid_columns"] = len(grid_px)
    result["calibration"]["x_grid_supported_columns"] = len(grid_support)

    for S in all_series:
        s, col, pts, r, twins = S["spec"], S["col"], S["pts"], S["r"], S["twins"]
        if not bool(s.get("extract", True)):
            continue
        rows = []
        for cx, cy, conf, ov, sure in pts:
            rows.append(
                {
                    "x": round(float(fx(cx)), 4),
                    "y": round(float(fy(cy)), 4),
                    "px": [round(float(cx), 2), round(float(cy), 2)],
                    "confidence": round(conf if sure else min(conf, 0.75), 2),
                    "overlap_flag": bool(ov),
                    "assigned_by": (
                        "joint_assignment"
                        if S.get("joint_assigned")
                        else ("color" if not twins else ("shape" if sure else "continuity"))
                    ),
                }
            )
            c = (
                (0, 255, 255)
                if not ov and rows[-1]["confidence"] >= CONF_OK
                else (0, 0, 255)
            )
            cv2.circle(overlay, (int(round(cx)), int(round(cy))), int(r) + 3, c, 1)
        for tx, ty, tconf, tsource in template_points.get(S["idx"], []):
            rows.append(
                {
                    "x": round(float(fx(tx)), 4),
                    "y": round(float(fy(ty)), 4),
                    "px": [round(float(tx), 2), round(float(ty), 2)],
                    "confidence": tconf,
                    "overlap_flag": True,
                    "assigned_by": "template",
                    "source": tsource,
                }
            )
        for gx, gy, gconf in grid_fill.get(S["idx"], []):
            rows.append(
                {
                    "x": round(float(fx(gx)), 4),
                    "y": round(float(fy(gy)), 4),
                    "px": [round(gx, 2), round(gy, 2)],
                    "confidence": gconf,
                    "overlap_flag": True,
                    "assigned_by": "grid",
                    "source": "grid",
                }
            )
            cv2.rectangle(
                overlay,
                (int(gx) - 3, int(gy) - 3),
                (int(gx) + 3, int(gy) + 3),
                (255, 160, 0),
                1,
            )
        for bx_, by_ in band_points.get(S["idx"], []):
            rows.append(
                {
                    "x": round(float(fx(bx_)), 4),
                    "y": round(float(fy(by_)), 4),
                    "px": [round(bx_, 2), round(by_, 2)],
                    "confidence": 0.4,
                    "overlap_flag": True,
                    "assigned_by": "band_sample",
                    "source": "curve",
                }
            )
        for kx, ky in knot.get(S["idx"], []):
            if any(abs(kx - g[0]) < 0.8 * r_med for g in grid_fill.get(S["idx"], [])):
                continue  # grid sample already covers this column
            rows.append(
                {
                    "x": round(float(fx(kx)), 4),
                    "y": round(float(fy(ky)), 4),
                    "px": [round(kx, 2), round(ky, 2)],
                    "confidence": 0.5,
                    "overlap_flag": True,
                    "assigned_by": "curve",
                    "source": "curve",
                }
            )
            cv2.circle(overlay, (int(round(kx)), int(round(ky))), 3, (255, 0, 255), 1)
        rows, n_duplicates = drop_exact_duplicates(rows)
        result["calibration"].setdefault("exact_duplicates_dropped", {})[s["label"]] = n_duplicates
        result["series"].append(
            {
                "label": s["label"],
                "n_points": len(rows),
                "marker_radius_px": round(r, 1),
                "extract": bool(s.get("extract", True)),
                "gas": s.get("gas", ""),
                "color_lab": [round(float(v), 1) for v in col],
                "marker_glyph": glyphs[S["idx"]] if S["idx"] < len(glyphs) else None,
                "color_twins": [spec["series"][j]["label"] for j in twins],
                "x_min": min((p["x"] for p in rows), default=None),
                "x_max": max((p["x"] for p in rows), default=None),
                "y_min": min((p["y"] for p in rows), default=None),
                "y_max": max((p["y"] for p in rows), default=None),
                "points": rows,
            }
        )
    result["proposal_quality"] = proposal_quality(result, spec)
    if out_prefix:
        cv2.imwrite(out_prefix + "_overlay.png", overlay)
        cv2.imwrite(out_prefix + "_panel.png", crop)
        json.dump(result, open(out_prefix + ".json", "w"), indent=1)
    return result, crop


if __name__ == "__main__":
    import sys

    spec = json.load(open(sys.argv[1]))
    res, _ = extract(spec, sys.argv[2] if len(sys.argv) > 2 else None)
    c = res["calibration"]
    print(
        f"1 px = {c['unit_per_px_x']:.4f} x-units, {c['unit_per_px_y']:.4f} y-units; tick resid {c['tick_fit_resid_x']:.3g}/{c['tick_fit_resid_y']:.3g}"
    )
    for s in res["series"]:
        flagged = sum(
            1 for p in s["points"] if p["overlap_flag"] or p["confidence"] < CONF_OK
        )
        print(
            f"{s['label']}: {s['n_points']} pts, r={s['marker_radius_px']}px, flagged={flagged}, x {s['x_min']}..{s['x_max']}, y {s['y_min']}..{s['y_max']}"
        )

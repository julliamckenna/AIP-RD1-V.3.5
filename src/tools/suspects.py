"""What agent 04 needs instead of every row: a per-series summary and the suspect points.

A 17-series chart has hundreds of rows; sending them all makes the final check slow and
expensive, and a reviewer can not judge 500 rows anyway.  Python lists the points worth a look:

  inferred      template_fill / template_fit / line_sample rows (no clean marker behind them)
  off_curve     the point sits far from its own series - measured against agent 02's
                ``curve_points`` when given, else against the median of its x neighbours - with
                the series whose curve explains it better as a hint

Agent 04 decides keep / delete / reassign for those; everything else is judged from the
summary, the re-plot and the redraw diff.
"""

from __future__ import annotations

import csv
import io

import numpy as np

INFERRED = {"template_fill", "template_fit", "line_sample"}


def _curve(series_spec):
    points = sorted((float(p["x"]), float(p["y"])) for p in series_spec.get("curve_points") or [])
    if len(points) < 2:
        return None
    xs, ys = zip(*points)
    return lambda x: float(np.interp(x, xs, ys))


def _neighbour_curve(rows):
    """y(x) from the series' own points: median of the nearest points in x (robust to one outlier)."""
    pts = sorted((float(r["x"]), float(r["y"])) for r in rows)

    def at(x, exclude=None):
        near = sorted((p for p in pts if p != exclude), key=lambda p: abs(p[0] - x))[:4]
        return float(np.median([p[1] for p in near])) if near else None
    return at


def find(rows, spec, max_rows=None, off_fraction=0.06):
    """Suspect rows as dicts with ``why`` and ``hint``, most suspicious first."""
    ticks = [float(v) for v in (spec.get("y") or {}).get("ticks") or [0, 1]]
    y_span = (max(ticks) - min(ticks)) or 1.0
    x_ticks = [float(v) for v in (spec.get("x") or {}).get("ticks") or [0, 1]]
    x_slack = 0.02 * ((max(x_ticks) - min(x_ticks)) or 1.0)  # a steep isotherm: a small x error is a big y gap
    by_series = {}
    for row in rows:
        by_series.setdefault(row["series_name"], []).append(row)
    declared = {s.get("label"): s for s in spec.get("series", [])}
    curves = {label: _curve(declared.get(label, {})) for label in by_series}
    neighbours = {label: _neighbour_curve(items) for label, items in by_series.items()}

    def expected(label, x, row=None):
        if curves.get(label):
            return curves[label](x)
        exclude = (float(row["x"]), float(row["y"])) if row is not None else None
        return neighbours[label](x, exclude) if label in neighbours else None

    def gap(label, x, y, row=None):
        """Distance in y to the series curve, allowing the point to sit up to x_slack left or right."""
        values = [expected(label, x + dx, row) for dx in np.linspace(-x_slack, x_slack, 9)]
        values = [v for v in values if v is not None]
        if not values:
            return None
        if min(values) <= y <= max(values):
            return 0.0
        return min(abs(y - v) for v in values)

    suspects = []
    for row in rows:
        label, x, y = row["series_name"], float(row["x"]), float(row["y"])
        reasons, score = [], 0.0
        if row.get("evidence_type") in INFERRED:
            reasons.append(f"inferred ({row['evidence_type']})")
            score = max(score, 0.5)
        off = gap(label, x, y, row)
        if off is not None and off > off_fraction * y_span:
            others = {other: gap(other, x, y) for other in by_series if other != label}
            others = {k: v for k, v in others.items() if v is not None}
            best = min(others, key=others.get) if others else None
            hint = f"; fits {best} better" if best and others[best] < 0.5 * off else ""
            reasons.append(f"{off:.3g} off its own curve{hint}")
            score = max(score, 1.0 + off / y_span)
        if reasons:
            suspects.append((score, row, "; ".join(reasons)))
    suspects.sort(key=lambda item: -item[0])
    selected = suspects if max_rows is None else suspects[:max_rows]
    return [{**row, "why": why} for _, row, why in selected]


def suspects_table(rows, spec, max_rows=None) -> str:
    """CSV index of every candidate row, annotating suspect rows as fallible leads."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["point_id", "series_name", "x", "y", "source_pixel_x", "source_pixel_y", "evidence_type", "why"])
    why_by_id = {str(row.get("point_id")): row["why"] for row in find(rows, spec, max_rows)}
    for row in rows:
        why = why_by_id.get(str(row.get("point_id")), "")
        writer.writerow([row.get(k) for k in ("point_id", "series_name", "x", "y", "source_pixel_x", "source_pixel_y",
                                              "evidence_type")] + [why])
    return buffer.getvalue().rstrip("\n")


def series_summary(rows, hard: dict) -> str:
    """Markdown table: one line per series with count, ranges, end value and the redraw scores."""
    redraw = (hard.get("redraw") or {}).get("series") or {}
    lines = ["| series | points | x range | y range | y at last point | redraw recall | redraw precision |",
             "|---|---|---|---|---|---|---|"]
    by_series = {}
    for row in rows:
        by_series.setdefault(row["series_name"], []).append(row)
    for label, items in by_series.items():
        xs, ys = [float(r["x"]) for r in items], [float(r["y"]) for r in items]
        last = max(items, key=lambda r: float(r["x"]))
        scores = redraw.get(label) or {}
        recall = scores.get("recall")
        lines.append(f"| {label} | {len(items)} | {min(xs):.4g} to {max(xs):.4g} | {min(ys):.4g} to {max(ys):.4g} | "
                     f"{float(last['y']):.4g} | {'n/a' if recall is None else recall} | {scores.get('precision')} |")
    return "\n".join(lines)


def compact_hard_checks(hard: dict) -> dict:
    """Hard-check facts for a prompt, with the limits of Python's supplied-calibration check stated explicitly."""
    axes = hard.get("axis_range") or {}
    return {
        "scope_note": (
            "These checks compare values to the calibration and tick values supplied to Python. They do not read "
            "printed labels or units from the source, confirm the correct crop, or prove that a point exists."
        ),
        "passed": hard.get("passed"),
        "problems": hard.get("problems", []),
        **{axis: {key: (axes.get(axis) or {}).get(key) for key in ("unit", "printed", "allowed", "data", "passed")}
           for axis in ("x", "y")},
        "redraw_recall": (hard.get("redraw") or {}).get("recall"),
        "redraw_precision": (hard.get("redraw") or {}).get("precision"),
        "physics_flags": {label: flags for label, flags in ((hard.get("physics") or {}).get("flags") or {}).items() if flags},
    }

"""
Agent 03 candidate reconstruction from extracted numbers only (no pixels from the original are used),
and put it side by side with the original panel. If the two look the same, the data is right.

  python recreate.py extraction.json spec.json panel.png out_prefix
  -> out_prefix_recreated.png  and  out_prefix_compare.png
"""

import json
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.tools import extract as extractor

MARKERS = {
    "circle": "o",
    "square": "s",
    "triangle": "^",
    "triangle_up": "^",
    "triangle_down": "v",
    "triangle-down": "v",
    "triangle-right": ">",
    "triangle-left": "<",
    "triangle_right": ">",
    "triangle_left": "<",
    "diamond": "D",
    "star": "*",
    "pentagon": "p",
    "hexagon": "h",
    "cross": "x",
    "plus": "+",
    "none": "o",
}


def lab_to_hex(lab):
    b, g, r = cv2.cvtColor(np.uint8([[[round(v) for v in lab]]]), cv2.COLOR_LAB2BGR)[
        0, 0
    ]
    return f"#{r:02x}{g:02x}{b:02x}"


def axis_label(axis, fallback):
    """Append units only when the title does not already include them."""
    title = str(axis.get("title") or fallback).strip()
    unit = str(axis.get("unit") or "").strip()
    if not unit or title == unit or title.endswith(f"({unit})"):
        return title
    return f"{title} ({unit})"


def recreate(extraction, spec, out_png, size=(8, 5.5), curve_traces=None):
    cal = extraction["calibration"]
    left, top, right, bottom = cal["frame_px"]
    # axis range = the plot frame, converted with the same calibration as the points
    p0 = next(
        (
            p
            for s in extraction["series"]
            for p in s["points"]
            if p.get("px") and p["px"][0] is not None
        ),
        None,
    )
    x_anchor = (p0["x"], p0["px"][0]) if p0 else None
    y_anchor = (p0["y"], p0["px"][1]) if p0 else None
    x_of = lambda px: extractor.pixel_to_axis(cal, "x", px, x_anchor)
    y_of = lambda py: extractor.pixel_to_axis(cal, "y", py, y_anchor)
    shapes = {
        s["label"]: MARKERS.get(str(s.get("marker", "circle")).lower(), "o")
        for s in spec.get("llm_spec", {}).get("series", [])
    }
    fills = {
        s["label"]: str(s.get("marker_fill", "filled")).lower()
        for s in spec.get("llm_spec", {}).get("series", [])
    }
    marker_radius = (
        float(
            np.median(
                [
                    float(series.get("marker_radius_px") or 6.0)
                    for series in extraction["series"]
                ]
            )
        )
        if extraction["series"]
        else 6.0
    )
    curve_traces = extraction.get("curve_traces", []) if curve_traces is None else curve_traces

    fig, ax = plt.subplots(figsize=size, dpi=150)
    for s in extraction["series"]:
        color = lab_to_hex(s["color_lab"]) if s.get("color_lab") else None
        mk = shapes.get(s["label"], "o")
        open_marker = fills.get(s["label"]) == "open"
        split = any(p.get("branch") in ("ads", "des") for p in s["points"])
        for branch in ("ads", "des") if split else ("single",):
            pts = sorted(
                [
                    p
                    for p in s["points"]
                    if p.get("branch", "single") in (branch, "single")
                    and p.get("source") not in ("grid", "curve")
                ],
                key=lambda p: p["x"],
            )
            if not pts:
                continue
            ax.plot(
                [p["x"] for p in pts],
                [p["y"] for p in pts],
                "-",
                color=color,
                lw=1,
                alpha=0.7,
            )
        for trace in curve_traces:
            if trace.get("series_name") != s["label"]:
                continue
            trace_points = trace.get("points", [])
            ax.plot(
                [point.get("x", x_of(point["px"][0])) for point in trace_points if point.get("px")],
                [point.get("y", y_of(point["px"][1])) for point in trace_points if point.get("px")],
                linestyle=(0, (4, 2)),
                color=trace.get("color_hex") or color,
                lw=1.5,
                alpha=0.95,
                label=f"{s['label']} source trace (line samples, not marker centers)",
            )
        # marker drawn at the same size relative to the plot frame as in the original, so the two pictures compare
        ms = float(
            np.clip(
                2 * marker_radius / max(1.0, right - left) * size[0] * 72 * 0.7,
                3.0,
                11.0,
            )
        )
        measured = [p for p in s["points"] if p.get("source") not in ("grid", "curve")]
        filled = [p for p in s["points"] if p.get("source") == "template_fill"]
        good = [p for p in measured if p["confidence"] >= 0.8]
        weak = [p for p in measured if p["confidence"] < 0.8]
        ax.plot(
            [p["x"] for p in good],
            [p["y"] for p in good],
            mk,
            color=color,
            ms=ms,
            mfc="none" if open_marker else color,
            mec=color if open_marker else "black",
            mew=1.0 if open_marker else 0.4,
            ls="none",
            label=s["label"],
        )
        if weak:
            ax.plot(
                [p["x"] for p in weak],
                [p["y"] for p in weak],
                mk,
                color=color,
                ms=ms,
                mfc="none" if open_marker else color,
                mec="red",
                mew=1.0,
                ls="none",
                alpha=0.85,
            )
        if filled:  # read on the line (grid column / curve sample), not a measured marker: hollow
            ax.plot(
                [p["x"] for p in filled],
                [p["y"] for p in filled],
                mk,
                mfc="none",
                mec=color,
                ms=ms,
                mew=1.0,
                ls="none",
                alpha=0.9,
            )
    ax.set_xlim(x_of(left), x_of(right))
    ax.set_ylim(y_of(bottom), y_of(top))
    ax.set_xticks(spec["x"]["ticks"])
    ax.set_yticks(spec["y"]["ticks"])
    if spec["x"].get("scale") == "log":
        ax.set_xscale("log")
    if spec["y"].get("scale") == "log":
        ax.set_yscale("log")
    ax.set_xlabel(axis_label(spec["x"], "x"))
    ax.set_ylabel(axis_label(spec["y"], "y"))
    ax.set_title(
        "recreated from extracted data  ·  dashed = separate raster line trace  ·  red edge = low confidence",
        fontsize=9,
        color="#555",
    )
    if extraction["series"]:
        ax.legend(
            fontsize=7,
            ncol=1 if len(extraction["series"]) <= 8 else 2,
            loc="best",
            framealpha=0.9,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No series passed final QA",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="#8a3b35",
        )
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def side_by_side(panel_png, recreated_png, out_png):
    a = cv2.imread(panel_png)
    b = cv2.imread(recreated_png)
    h = max(a.shape[0], b.shape[0])
    a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h))
    b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))
    gap = np.full((h, 24, 3), 255, np.uint8)
    both = np.hstack([a, gap, b])
    cv2.putText(both, "ORIGINAL", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(
        both,
        "RECREATED",
        (a.shape[1] + 36, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )
    cv2.imwrite(out_png, both)
    return out_png


def contact_sheet(panel_png, overlay_png, extracted_png, out_png):
    """Write source, Agent 03 overlay, and CSV reconstruction in one image."""
    inputs = [
        ("SOURCE", panel_png),
        ("CANDIDATE OVERLAY", overlay_png),
        ("CANDIDATE EXTRACTED", extracted_png),
    ]
    loaded = []
    for label, path in inputs:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"contact sheet: could not open {path}")
        loaded.append((label, image))

    target_height = max(image.shape[0] for _, image in loaded)
    cards = []
    for label, image in loaded:
        width = max(1, round(image.shape[1] * target_height / image.shape[0]))
        resized = cv2.resize(
            image, (width, target_height), interpolation=cv2.INTER_AREA
        )
        card = cv2.copyMakeBorder(
            resized,
            42,
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        cv2.putText(
            card, label, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (25, 25, 25), 2
        )
        cards.append(card)
    gap = np.full((target_height + 42, 20, 3), 245, np.uint8)
    sheet = np.hstack([cards[0], gap, cards[1], gap, cards[2]])
    cv2.imwrite(str(out_png), sheet)
    return out_png


if __name__ == "__main__":
    ext = json.load(open(sys.argv[1]))
    spec = json.load(open(sys.argv[2]))
    prefix = sys.argv[4]
    recreate(ext, spec, prefix + "_recreated.png")
    side_by_side(sys.argv[3], prefix + "_recreated.png", prefix + "_compare.png")
    print(prefix + "_compare.png")

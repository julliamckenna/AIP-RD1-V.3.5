"""
Benchmark generator: synthetic isotherm charts with EXACT ground truth, degraded the way real figures are.

Every chart = real-looking Toth isotherms (parameters in the range published for CO2 on MOFs / zeolites),
rendered with matplotlib in journal-like styles, then pushed down a degradation ladder:
   render at 300 dpi  ->  downscale to the target width  ->  JPEG at the target quality  (-> second JPEG pass)
Ground truth = the exact pixel centre of every marker in the FINAL image, plus its series and data value.

Factors (see CASES): number of series, marker size, image width, JPEG quality, palette (distinct vs Origin-like
near-duplicates), legend inside/outside, adsorption-only vs adsorption+desorption loops, set-point density.

  python -m evaluations.generate evaluations/cases        -> one folder per case: chart.jpg, truth.csv, spec.json
"""

import json
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

R_GAS = 8.314

MARKERS = [
    "s",
    "o",
    "^",
    "v",
    "<",
    ">",
    "D",
    "p",
    "*",
    "h",
    "d",
    "P",
    "X",
    "8",
    "s",
    "o",
    "^",
]
MARKER_NAMES = {
    "s": "square",
    "o": "circle",
    "^": "triangle",
    "v": "triangle-down",
    "<": "triangle-left",
    ">": "triangle-right",
    "D": "diamond",
    "p": "pentagon",
    "*": "star",
    "h": "hexagon",
    "d": "diamond",
    "P": "plus",
    "X": "cross",
    "8": "circle",
}
PALETTES = {
    "distinct": [
        "#000000",
        "#d62728",
        "#1f77b4",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#17becf",
        "#bcbd22",
        "#7f7f7f",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
    ],
    # Origin-like: several near-identical blues / reds, as in real multi-temperature figures
    "origin": [
        "#000000",
        "#e8302a",
        "#3a4ea5",
        "#2a8a8a",
        "#b0559b",
        "#a9a532",
        "#2b2b7a",
        "#7a1f1c",
        "#e34b8f",
        "#3d9a3f",
        "#3f3f8f",
        "#7c2f7d",
        "#6a4e9c",
        "#111111",
        "#e05050",
        "#4a4f9f",
        "#2f8f8f",
    ],
}


def toth(P, qm, K, t):
    return qm * K * P / (1 + (K * P) ** t) ** (1 / t)


def make_series(n, x_max, rng):
    """n temperatures, physically consistent: qm shared, K falls with T via a Qst of ~30 kJ/mol."""
    temps = np.linspace(273, 373, n) if n > 1 else np.array([298.0])
    qm = rng.uniform(4, 9)
    t = rng.uniform(0.35, 0.8)
    qst = rng.uniform(22e3, 34e3)
    K0 = rng.uniform(0.004, 0.02)  # at 273 K, per unit of x (mbar or bar)
    out = []
    for T in temps:
        K = K0 * np.exp(qst / R_GAS * (1 / T - 1 / 273.0))
        out.append(
            {"label": f"{int(round(T))}K", "T": float(T), "qm": qm, "K": K, "t": t}
        )
    return out


def setpoints(x_max, density, rng):
    if density == "sparse":
        base = np.linspace(0, x_max, 18)[1:]
    elif density == "normal":
        base = np.concatenate(
            [
                np.linspace(0, 0.1 * x_max, 6)[1:],
                np.linspace(0.1 * x_max, x_max, 24)[1:],
            ]
        )
    else:  # dense low-pressure knot like real instruments
        base = np.concatenate(
            [
                np.linspace(0, 0.08 * x_max, 16)[1:],
                np.linspace(0.08 * x_max, x_max, 40)[1:],
            ]
        )
    return base * rng.uniform(0.98, 1.02, size=len(base))


def render_case(case, out_dir, seed):
    rng = np.random.default_rng(seed)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_max = 1200.0 if case["x_unit"] == "mbar" else 10.0
    series = make_series(case["n_series"], x_max, rng)
    xs = setpoints(x_max, case["density"], rng)
    pal = PALETTES[case["palette"]]
    fig_w, fig_h = 8.0, 5.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    truth = []
    for i, s in enumerate(series):
        y = toth(xs, s["qm"], s["K"], s["t"]) * (1 + rng.normal(0, 0.004, len(xs)))
        col = pal[i % len(pal)]
        mk = MARKERS[i % len(MARKERS)]
        ax.plot(xs, y, "-", color=col, lw=case["line_w"], zorder=2)
        ax.plot(
            xs,
            y,
            mk,
            color=col,
            ms=case["marker_pt"],
            mec=col,
            zorder=3,
            label=s["label"],
        )
        for xv, yv in zip(xs, y):
            truth.append(
                {"series": s["label"], "x": float(xv), "y": float(yv), "branch": "ads"}
            )
        if case["loops"]:
            y_des = y * (1 + rng.uniform(0.01, 0.04))  # a little hysteresis
            ax.plot(xs, y_des, "-", color=col, lw=case["line_w"], zorder=2)
            ax.plot(xs, y_des, mk, color=col, ms=case["marker_pt"], mec=col, zorder=3)
            for xv, yv in zip(xs, y_des):
                truth.append(
                    {
                        "series": s["label"],
                        "x": float(xv),
                        "y": float(yv),
                        "branch": "des",
                    }
                )
    ax.set_xlabel(f"P / {case['x_unit']}")
    ax.set_ylabel("CO2 Loading [mmol/g]")
    xt = np.linspace(0, x_max, 7) if case["x_unit"] == "mbar" else np.arange(0, 11, 2)
    ax.set_xticks(xt)
    ymax = max(t["y"] for t in truth) * 1.08
    yt = np.arange(0, np.floor(ymax) + 0.01, 1.0)
    ax.set_yticks(yt)
    ax.set_ylim(-0.05 * ymax, ymax)
    ax.set_xlim(-0.03 * x_max, 1.05 * x_max)
    if case["legend"] == "inside":
        ax.legend(loc="lower right", fontsize=7, ncol=1 if case["n_series"] <= 8 else 2)
    else:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=7,
            ncol=1 if case["n_series"] <= 11 else 2,
        )
    fig.tight_layout()
    hi = out_dir / "chart_hi.png"
    fig.savefig(hi, dpi=300)
    # exact pixel positions at 300 dpi
    fig.canvas.draw()
    W_hi, H_hi = fig.canvas.get_width_height()
    for t in truth:
        px, py = ax.transData.transform((t["x"], t["y"]))
        t["px_hi"], t["py_hi"] = float(px), float(H_hi - py)
    # frame / ticks / legend in hi-res pixels (for the spec)
    bb = ax.get_window_extent()
    frame = [bb.x0, H_hi - bb.y1, bb.x1, H_hi - bb.y0]
    leg = ax.get_legend().get_window_extent()
    legend_hi = [leg.x0, H_hi - leg.y1, leg.x1, H_hi - leg.y0]
    plt.close(fig)
    # ---- degradation ladder ----
    img = Image.open(hi).convert("RGB")
    scale = case["width"] / img.width
    img = img.resize((case["width"], int(round(img.height * scale))), Image.LANCZOS)
    final = out_dir / "chart.jpg"
    img.save(final, quality=case["jpeg_q"])
    if case["double_jpeg"]:
        Image.open(final).save(final, quality=max(50, case["jpeg_q"] - 10))
    for t in truth:
        t["px"], t["py"] = t["px_hi"] * scale, t["py_hi"] * scale
    W, H = img.size
    # marker radius in final px (matplotlib ms is a diameter in points; 300 dpi -> px)
    r_px = case["marker_pt"] / 72 * 300 * scale / 2
    spec = {
        "image": str(final),
        "panel_bbox": [0, 0, W, H],
        "x": {
            "unit": case["x_unit"],
            "scale": "linear",
            "ticks": [float(v) for v in xt],
        },
        "y": {"unit": "mmol/g", "scale": "linear", "ticks": [float(v) for v in yt]},
        "legend_bbox": [
            int(legend_hi[0] * scale) - 4,
            int(legend_hi[1] * scale) - 4,
            int(legend_hi[2] * scale) + 4,
            int(legend_hi[3] * scale) + 4,
        ],
        "series": [
            {
                "label": s["label"],
                "extract": True,
                "y_anchors": {
                    "p25": float(toth(0.25 * x_max, s["qm"], s["K"], s["t"])),
                    "p50": float(toth(0.5 * x_max, s["qm"], s["K"], s["t"])),
                    "p75": float(toth(0.75 * x_max, s["qm"], s["K"], s["t"])),
                    "p100": float(toth(x_max, s["qm"], s["K"], s["t"])),
                },
                "y_at_xmax": float(toth(x_max, s["qm"], s["K"], s["t"])),
            }
            for s in series
        ],
        "llm_spec": {
            "series": [
                {
                    "label": s["label"],
                    "marker": MARKER_NAMES.get(MARKERS[i % len(MARKERS)], "circle"),
                }
                for i, s in enumerate(series)
            ]
        },
    }
    json.dump(spec, open(out_dir / "spec.json", "w"), indent=1)
    import csv

    with open(out_dir / "truth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["series", "branch", "x", "y", "px", "py"])
        w.writeheader()
        for t in truth:
            w.writerow({k: t[k] for k in ["series", "branch", "x", "y", "px", "py"]})
    meta = dict(case)
    meta.update(
        {
            "seed": seed,
            "width": W,
            "height": H,
            "marker_r_px": r_px,
            "n_truth": len(truth),
            "frame_px": [v * scale for v in frame],
        }
    )
    json.dump(meta, open(out_dir / "meta.json", "w"), indent=1)
    hi.unlink()
    return meta


# ---- the case grid: a ladder from easy to the figure you sent ----
CASES = []
for n_series, palette in [
    (4, "distinct"),
    (8, "distinct"),
    (11, "origin"),
    (17, "origin"),
]:
    for width, jpeg_q, double in [
        (1600, 92, False),
        (1200, 78, False),
        (1000, 70, True),
    ]:
        for marker_pt, density, loops in [(7, "normal", False), (5, "dense", True)]:
            CASES.append(
                {
                    "n_series": n_series,
                    "palette": palette,
                    "width": width,
                    "jpeg_q": jpeg_q,
                    "double_jpeg": double,
                    "marker_pt": marker_pt,
                    "density": density,
                    "loops": loops,
                    "line_w": 0.9,
                    "legend": "outside" if n_series > 8 else "inside",
                    "x_unit": "mbar",
                }
            )

if __name__ == "__main__":
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "evaluations/cases")
    rows = []
    for i, case in enumerate(CASES):
        name = f"c{i:02d}_s{case['n_series']}_{case['palette']}_w{case['width']}_q{case['jpeg_q']}{'x2' if case['double_jpeg'] else ''}_m{case['marker_pt']}_{case['density']}{'_loops' if case['loops'] else ''}"
        meta = render_case(case, root / name, seed=100 + i)
        rows.append((name, meta["n_truth"], round(meta["marker_r_px"], 1)))
        print(f"{name}: {meta['n_truth']} markers, r={meta['marker_r_px']:.1f}px")
    print(f"{len(rows)} cases in {root}")

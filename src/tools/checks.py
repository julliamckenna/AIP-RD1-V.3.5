"""
Agent 03 physics checks on the extracted isotherms. Pure Python; Agent 04 reads the flags during reconciliation.

Checks per series:
  monotonic   : uptake non-decreasing with pressure (allowing a tolerance = 2 px in y)
  branches    : residual clustering around a smooth fit -> adsorption (below) / desorption (above)
  smoothness  : (warn) interior point vs quadratic through its neighbours in log P, > 6 px
  toth        : informational Toth fit of the adsorption branch (used for cross-series checks)
Across series (if labels carry a temperature in K):
  t_ordering  : at matched pressures, uptake must decrease with temperature
  qst         : isosteric heat from Clausius-Clapeyron at fixed loading, flag if outside 10-60 kJ/mol
"""

import json
import re
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from scipy.optimize import curve_fit

from src.settings import CFG

R_GAS = 8.314
QST_MIN, QST_MAX = CFG["checks"]["qst_min_kj_mol"], CFG["checks"]["qst_max_kj_mol"]
CONF_OK = CFG["extract"]["confidence_ok"]


def toth(P, qm, K, t):
    return qm * K * P / (1 + (K * P) ** t) ** (1 / t)


def series_temperature(label):
    m = re.search(r"(\d{3})\s*K", label)
    return float(m.group(1)) if m else None


def split_branches(s, y_px):
    """Fit a smooth curve (poly deg 4 in log10 P) through all points; 2-means on the residuals.
    Desorption sits above (positive residual), adsorption below. If the two clusters are closer
    than 3 px there is no visible hysteresis and every point is labelled 'single'."""
    pts = sorted(s["points"], key=lambda p: p["x"])
    P = np.array([max(p["x"], 1e-3) for p in pts])
    Q = np.array([p["y"] for p in pts])
    if len(pts) < 8:
        for p in pts:
            p["branch"] = "single"
        return 0.0
    lp = np.log10(P)
    c = np.polyfit(lp, Q, min(4, len(pts) - 3))
    r = Q - np.polyval(c, lp)
    lo, hi = r.min(), r.max()
    for _ in range(20):
        lab = r > (lo + hi) / 2
        if lab.all() or (~lab).all():
            break
        lo, hi = r[~lab].mean(), r[lab].mean()
    sep = hi - lo
    if lab.all() or (~lab).all() or sep < 3 * y_px:
        for p in pts:
            p["branch"] = "single"
        return 0.0
    for p, is_des in zip(pts, lab):
        p["branch"] = "des" if is_des else "ads"
    return float(sep)


def local_outliers(P, Q, tol):
    """Model-free: each interior point vs a quadratic (in log10 P) through its 2 neighbours on each side."""
    lp = np.log10(np.maximum(P, 1e-4))
    bad = []
    for i in range(2, len(P) - 2):
        idx = [i - 2, i - 1, i + 1, i + 2]
        c = np.polyfit(lp[idx], Q[idx], 2)
        r = Q[i] - np.polyval(c, lp[i])
        if abs(r) > tol:
            bad.append((round(P[i], 3), round(Q[i], 3), round(float(r), 3)))
    return bad


def confident(s):
    """points the physics checks may rely on (low-confidence ones are already flagged for a human)"""
    good = [
        p
        for p in s["points"]
        if p["confidence"] >= CONF_OK and not p.get("overlap_flag")
    ]
    return good if len(good) >= 5 else s["points"]


def check_series(s, y_px):
    # Branches inferred by the physics model are diagnostic hypotheses. Keep them
    # on private point records so QA cannot change rendering or exported evidence.
    s = {**s, "points": [dict(p) for p in s["points"]]}
    flags = []
    split_branches(s, y_px)
    s = {**s, "points": confident(s)}
    tol = 2 * y_px
    for br in ("ads", "des"):
        pts = sorted(
            [p for p in s["points"] if p["branch"] in (br, "single")],
            key=lambda p: p["x"],
        )
        if not pts:
            continue
        P = np.array([p["x"] for p in pts], float)
        Q = np.array([p["y"] for p in pts], float)
        drops = [
            (round(P[i], 3), round(Q[i], 3))
            for i in range(1, len(Q))
            if Q[i] < Q[i - 1] - tol
        ]
        if drops:
            flags.append(
                {
                    "check": "monotonic",
                    "detail": f"{br}: uptake decreases at {drops[:5]}",
                }
            )
        bad = local_outliers(P, Q, 6 * y_px)
        if bad:
            flags.append(
                {
                    "check": "smoothness",
                    "level": "warn",
                    "detail": f"{br}: local outliers (P,q,resid) {bad[:5]}",
                }
            )
    fit = None
    pts = sorted(
        [p for p in s["points"] if p["branch"] in ("ads", "single")],
        key=lambda p: p["x"],
    )
    P = np.array([p["x"] for p in pts], float)
    Q = np.array([p["y"] for p in pts], float)
    P = np.maximum(P, 0.0)
    if len(P) >= 5:
        try:
            popt, _ = curve_fit(
                toth,
                P,
                Q,
                p0=[Q.max() * 1.5, 1.0, 0.7],
                bounds=([0, 1e-4, 0.1], [50, 1e3, 3]),
                maxfev=20000,
            )
            res = Q - toth(P, *popt)
            fit = {
                "qm": float(popt[0]),
                "K": float(popt[1]),
                "t": float(popt[2]),
                "rmse": float(np.sqrt(np.mean(res**2))),
            }
        except Exception:
            pass
    return flags, fit


def cross_series(series, fits):
    flags = []
    temps = {s["label"]: series_temperature(s["label"]) for s in series}
    ok = [
        s
        for s in series
        if temps[s["label"]]
        and fits.get(s["label"])
        and s.get("x_max") is not None
        and s["x_max"] > 0
    ]
    if len(ok) < 2:
        return flags
    ok.sort(key=lambda s: temps[s["label"]])
    xm = min(s["x_max"] for s in ok)
    Pgrid = np.linspace(0.1 * xm, xm, 8)
    q = np.array(
        [
            toth(
                Pgrid,
                fits[s["label"]]["qm"],
                fits[s["label"]]["K"],
                fits[s["label"]]["t"],
            )
            for s in ok
        ]
    )
    for i in range(1, len(ok)):
        if np.any(q[i] > q[i - 1] * 1.02):
            flags.append(
                {
                    "check": "t_ordering",
                    "detail": f"{ok[i]['label']} exceeds {ok[i - 1]['label']} at some pressures — series colors may be swapped",
                }
            )
    # isosteric heat: ln P vs 1/T at fixed loading, using fits inverted numerically
    T = np.array([temps[s["label"]] for s in ok])
    q_common = (
        np.linspace(max(q[:, 0]), min(q[:, -1]), 5)
        if max(q[:, 0]) < min(q[:, -1])
        else []
    )
    for qc in q_common:
        lnP = []
        for s in ok:
            f = fits[s["label"]]
            Ps = np.linspace(1e-3, s["x_max"], 4000)
            lnP.append(
                np.log(Ps[np.argmin(np.abs(toth(Ps, f["qm"], f["K"], f["t"]) - qc))])
            )
        slope = np.polyfit(1 / T, lnP, 1)[0]
        qst = -slope * R_GAS / 1000
        if not (QST_MIN <= qst <= QST_MAX):
            flags.append(
                {
                    "check": "qst",
                    "detail": f"Qst={qst:.1f} kJ/mol at q={qc:.2f} (expected {QST_MIN}-{QST_MAX})",
                }
            )
            break
    return flags


def _axis_units_per_pixel(calibration, axis):
    """Representative data-unit step for pixel-based QA tolerances."""
    model = (calibration.get("axis_models") or {}).get(axis)
    if not model or model.get("scale") != "log":
        return calibration[f"unit_per_px_{axis}"]
    left, top, right, bottom = calibration["frame_px"]
    pixel = (left + right) / 2 if axis == "x" else (top + bottom) / 2
    value = lambda p: 10 ** (model["slope"] * p + model["intercept"])
    return abs(value(pixel + 1) - value(pixel))


def run(result):
    """Return supporting diagnostics without modifying the extracted evidence."""
    cal = result["calibration"]
    y_px = _axis_units_per_pixel(cal, "y")
    fits, report = {}, {"series": {}, "cross": []}
    for s in result["series"]:
        flags, fit = check_series(s, y_px)
        fits[s["label"]] = fit
        report["series"][s["label"]] = {
            "flags": flags,
            "toth": fit,
            "low_confidence_points": sum(
                1 for p in s["points"] if p["confidence"] < CONF_OK
            ),
        }
    report["cross"] = cross_series(
        [{**s, "points": confident(s)} for s in result["series"]], fits
    )
    n_flags = sum(
        1
        for v in report["series"].values()
        for f in v["flags"]
        if f.get("level") != "warn"
    ) + len(report["cross"])
    report["verdict"] = "auto_accept" if n_flags == 0 else "review"
    return report


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as handle:
        res = json.load(handle)
    rep = run(res)
    print(json.dumps(rep, indent=1, default=float))

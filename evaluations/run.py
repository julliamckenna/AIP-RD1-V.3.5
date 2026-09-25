"""
Benchmark runner: extract every case with the deterministic pipeline (step 4 only — the chart spec comes from the
generator, so the AI steps are out of the loop and what is measured is the extractor) and score it against truth.

Metrics per case, plus breakdowns by factor:
  recall            truth markers matched by a prediction within one marker radius (any series)
  precision         predictions that match a truth marker
  ownership         matched markers whose series label is right
  centre P50 / P95  distance between prediction and truth centre, px and y-units, for correctly-owned matches
  recall_clear / recall_crowded   recall on markers that are isolated vs. overlapping another marker (any series)

  python -m evaluations.run evaluations/cases evaluations/report
"""

import csv
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.tools import extract as R


def score(case_dir):
    case_dir = pathlib.Path(case_dir)
    meta = json.load(open(case_dir / "meta.json"))
    spec = json.load(open(case_dir / "spec.json"))
    truth = list(csv.DictReader(open(case_dir / "truth.csv")))
    for t in truth:
        t["px"], t["py"], t["x"], t["y"] = (
            float(t["px"]),
            float(t["py"]),
            float(t["x"]),
            float(t["y"]),
        )
    r = meta["marker_r_px"]
    t0 = time.time()
    try:
        res, _ = R.extract(spec, str(case_dir / "out"))
        err = None
    except Exception as e:
        res, err = {"series": [], "calibration": {}}, str(e)
    secs = time.time() - t0
    preds = []
    for s in res["series"]:
        for p in s["points"]:
            if (
                p.get("source") in ("curve", "grid")
                or not p.get("px")
                or p["px"][0] is None
            ):
                continue
            preds.append(
                {
                    "series": s["label"],
                    "px": p["px"][0],
                    "py": p["px"][1],
                    "x": p["x"],
                    "y": p["y"],
                    "conf": p["confidence"],
                }
            )
    # crowded truth: another truth marker (any series) within 1.6 r (markers touch/overlap)
    T = np.array([[t["px"], t["py"]] for t in truth]) if truth else np.zeros((0, 2))
    crowded = np.zeros(len(truth), bool)
    for i in range(len(truth)):
        d = np.hypot(T[:, 0] - T[i, 0], T[:, 1] - T[i, 1])
        d[i] = np.inf
        crowded[i] = d.min() < 2.2 * r
    # greedy matching, nearest first, within 1.0 r (min 3 px)
    tol = max(3.0, 1.0 * r)
    pairs = []
    for j, p in enumerate(preds):
        if len(truth) == 0:
            break
        d = np.hypot(T[:, 0] - p["px"], T[:, 1] - p["py"])
        i = int(d.argmin())
        if d[i] <= tol:
            pairs.append((d[i], j, i))
    pairs.sort()
    used_t, used_p, matches = set(), set(), []
    for d, j, i in pairs:
        if i in used_t or j in used_p:
            continue
        used_t.add(i)
        used_p.add(j)
        matches.append((j, i, d))
    tp = len(matches)
    own_ok = [
        (j, i, d) for j, i, d in matches if preds[j]["series"] == truth[i]["series"]
    ]
    dpx = np.array([d for _, _, d in own_ok]) if own_ok else np.array([np.nan])
    dy = (
        np.array([abs(preds[j]["y"] - truth[i]["y"]) for j, i, _ in own_ok])
        if own_ok
        else np.array([np.nan])
    )
    dx = (
        np.array([abs(preds[j]["x"] - truth[i]["x"]) for j, i, _ in own_ok])
        if own_ok
        else np.array([np.nan])
    )
    matched_t = {i for _, i, _ in matches}
    clear_idx = [i for i in range(len(truth)) if not crowded[i]]
    crowd_idx = [i for i in range(len(truth)) if crowded[i]]
    rec_clear = np.mean([i in matched_t for i in clear_idx]) if clear_idx else np.nan
    rec_crowd = np.mean([i in matched_t for i in crowd_idx]) if crowd_idx else np.nan
    row = {
        "case": case_dir.name,
        **{
            k: meta[k]
            for k in [
                "n_series",
                "palette",
                "width",
                "jpeg_q",
                "double_jpeg",
                "marker_pt",
                "density",
                "loops",
            ]
        },
        "marker_r_px": round(r, 1),
        "n_truth": len(truth),
        "n_pred": len(preds),
        "crowded_frac": round(float(crowded.mean()), 3) if len(truth) else 0,
        "recall": round(tp / max(1, len(truth)), 3),
        "precision": round(tp / max(1, len(preds)), 3),
        "ownership": round(len(own_ok) / max(1, tp), 3),
        "recall_clear": round(float(rec_clear), 3),
        "recall_crowded": round(float(rec_crowd), 3),
        "centre_px_p50": round(float(np.nanmedian(dpx)), 2),
        "centre_px_p95": round(float(np.nanpercentile(dpx, 95)), 2),
        "err_y_p50": round(float(np.nanmedian(dy)), 4),
        "err_y_p95": round(float(np.nanpercentile(dy, 95)), 4),
        "err_x_p50": round(float(np.nanmedian(dx)), 3),
        "seconds": round(secs, 1),
        "error": err or "",
    }
    return row


def report(rows, out_dir):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if not r["error"]]

    def agg(group_key):
        out = {}
        for r in ok:
            out.setdefault(r[group_key], []).append(r)
        lines = [
            f"| {group_key} | cases | recall | precision | ownership | recall clear | recall crowded | centre P50 px | centre P95 px | y err P50 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for k, rs in sorted(out.items(), key=lambda kv: str(kv[0])):
            m = lambda f: np.nanmean([r[f] for r in rs])
            lines.append(
                f"| {k} | {len(rs)} | {m('recall'):.2f} | {m('precision'):.2f} | {m('ownership'):.2f} | {m('recall_clear'):.2f} | {m('recall_crowded'):.2f} | {m('centre_px_p50'):.2f} | {m('centre_px_p95'):.2f} | {m('err_y_p50'):.4f} |"
            )
        return "\n".join(lines)

    m = lambda f: np.nanmean([r[f] for r in ok])
    md = [
        f"# Extractor benchmark — {len(ok)} synthetic cases ({len(rows) - len(ok)} failed)\n",
        f"Overall: recall {m('recall'):.2f} · precision {m('precision'):.2f} · ownership {m('ownership'):.2f} · "
        f"recall on clear markers {m('recall_clear'):.2f} · on crowded markers {m('recall_crowded'):.2f} · "
        f"centre error P50 {m('centre_px_p50'):.2f} px / P95 {m('centre_px_p95'):.2f} px · y error P50 {m('err_y_p50'):.4f} mmol/g\n",
        "Definitions: a truth marker is *crowded* when another marker (any series) lies within 2.2 marker radii. "
        "Matching tolerance = 1 marker radius. Ownership = matched markers with the right series label. "
        "Centre error is measured on correctly owned matches only.\n",
    ]
    for k in [
        "n_series",
        "palette",
        "width",
        "jpeg_q",
        "marker_pt",
        "density",
        "loops",
    ]:
        md.append(f"\n## by {k}\n\n" + agg(k))
    md.append(
        "\n\n## per case\n\n| case | truth | pred | crowded | recall | precision | ownership | rec clear | rec crowded | P50 px | P95 px | s |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in rows:
        md.append(
            f"| {r['case']} | {r['n_truth']} | {r['n_pred']} | {r['crowded_frac']:.2f} | {r['recall']:.2f} | {r['precision']:.2f} | {r['ownership']:.2f} | {r['recall_clear']:.2f} | {r['recall_crowded']:.2f} | {r['centre_px_p50']:.2f} | {r['centre_px_p95']:.2f} | {r['seconds']} |"
            + (f" ERROR {r['error'][:60]}" if r["error"] else "")
        )
    (out_dir / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    return out_dir / "REPORT.md"


if __name__ == "__main__":
    cases = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "evaluations/cases")
    out = sys.argv[2] if len(sys.argv) > 2 else "evaluations/report"
    rows = []
    for d in sorted(p for p in cases.iterdir() if p.is_dir()):
        r = score(d)
        rows.append(r)
        print(
            f"{d.name:55s} recall {r['recall']:.2f}  prec {r['precision']:.2f}  own {r['ownership']:.2f}  clear {r['recall_clear']:.2f}  crowd {r['recall_crowded']:.2f}  P50 {r['centre_px_p50']} px  {r['error'][:50]}"
        )
    print("report:", report(rows, out))

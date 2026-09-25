"""
Where do the misses go? For one benchmark case: one-to-one matching like run.py, then for every missed CLEAR
truth marker, replay the detection stages on that spot and report which stage lost it.

  python -m evaluations.diagnose evaluations/cases/<case>
"""
import csv, json, pathlib, sys
from collections import Counter
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.tools import extract as R


def match(preds, T, r):
    tol = max(3.0, r); pairs = []
    for j, p in enumerate(preds):
        d = np.hypot(T[:, 0] - p[0], T[:, 1] - p[1]); i = int(d.argmin())
        if d[i] <= tol: pairs.append((d[i], j, i))
    pairs.sort(); ut, up, m = set(), set(), {}
    for d, j, i in pairs:
        if i in ut or j in up: continue
        ut.add(i); up.add(j); m[i] = j
    return m


def main(case_dir):
    d = pathlib.Path(case_dir); spec = json.load(open(d / "spec.json")); truth = list(csv.DictReader(open(d / "truth.csv")))
    r = json.load(open(d / "meta.json"))["marker_r_px"]
    T = np.array([[float(t["px"]), float(t["py"])] for t in truth])
    crowded = np.array([(np.hypot(T[:, 0] - T[i, 0], T[:, 1] - T[i, 1]) + np.eye(len(T))[i] * 1e9).min() < 2.2 * r for i in range(len(T))])

    # capture candidates before settlement
    cand = {}
    orig = R.settle_twins
    def spy(all_series, *a, **k):
        for S in all_series:
            cand[S["spec"]["label"]] = [(p[0], p[1]) for p in S["pts"]]
        return orig(all_series, *a, **k)
    R.settle_twins = spy
    res, _ = R.extract(spec)
    R.settle_twins = orig
    preds = [(p["px"][0], p["px"][1], s["label"]) for s in res["series"] for p in s["points"] if not p.get("source") and p["px"][0] is not None]
    m = match(preds, T, r)
    only_clear = "--all" not in sys.argv
    missed_clear = [i for i in range(len(T)) if i not in m and (not crowded[i] or not only_clear)]
    print(f"{d.name}: truth {len(T)}, clear {(~crowded).sum()}, matched {len(m)}, missed clear {len(missed_clear)}")
    im = cv2.imread(spec["image"]); lab = R.to_lab(im)
    cols = {s["label"]: np.array(s["color_lab"], np.float32) for s in res["series"]}
    tols = dict(zip([s["label"] for s in spec["series"]], res["calibration"]["color_tolerance_per_series"]))
    stage = Counter(); examples = {}
    for i in missed_clear:
        t = truth[i]; lbl = t["series"]; px, py = T[i]
        # stage A: was there a candidate of this series within r before settlement?
        if any(abs(cx - px) <= r and abs(cy - py) <= r for cx, cy in cand.get(lbl, [])):
            # stage B: did ANOTHER series get a prediction there (ownership) or was it dropped (settle)?
            other = [p for p in preds if abs(p[0] - px) <= r and abs(p[1] - py) <= r]
            key = "candidate ok -> given to another series" if other else "candidate ok -> dropped in settlement/dedupe"
        else:
            w = int(4 * r); sub = lab[max(0, int(py) - w):int(py) + w + 1, max(0, int(px) - w):int(px) + w + 1]
            mk = R.color_mask(sub, cols[lbl], tols[lbl]); mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            if (mk > 0).sum() < 0.3 * np.pi * r * r:
                key = "no color mask (color/tolerance)"
            else:
                pts, _ = R.marker_centres(mk, np.zeros(mk.shape, np.uint8), r_hint=r)
                key = "local candidate exists but not in full-image pass" if any(abs(a - w) < r and abs(b - w) < r for a, b, *_ in pts) else "rejected by marker_centres"
        stage[key] += 1; examples.setdefault(key, []).append((lbl, round(float(t["x"])), round(float(t["y"]), 2)))
    for k, v in stage.most_common():
        print(f"  {v:4d}  {k}   e.g. {examples[k][:4]}")
    print("  missed clear by series:", Counter(truth[i]["series"] for i in missed_clear).most_common(6))


if __name__ == "__main__":
    main(sys.argv[1])

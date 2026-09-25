"""The Python extraction - extract points (deterministic Python, no model call). Runs every round of the check loop.

Input: spec.json - the agent 01 reading, plus agent 02's corrections from earlier rounds.
src/tools/extract.py calibrates the axes from the printed ticks (and tick anchors once agent 02
has supplied them) and detects every marker of every CO2 series.
Output: the python stage (python_points.csv/json/metadata + overlay, re-plot and side-by-side
images), python_hard_checks.json + python_redraw_diff.png (src/tools/hard_checks.py) and
python_summary.json.  When calibration fails - or succeeds but the values contradict the printed
axis (hard axis-range check) - the error is recorded and the detected tick candidates are drawn
(python_tick_check.png) so agent 02 can fix the mapping.
Always continues to agent 02, which decides whether to accept, adjust and rebuild, or give up.
"""

from __future__ import annotations

import cv2

from src.tools import dense_regions, evidence, extract, hard_checks, legend_markers, recreate, stage_artifacts
from src.workflow.state import CONTINUE, DocumentRun, Panel
from src.workflow.steps.base import PanelStep, read_json, save_json


def legend_marker_check(spec: dict, stage: dict) -> list[dict]:
    """Declared series style (agent 01 / agent 02) next to the glyph measured from the legend pixels."""
    declared = {str(series.get("label")): series for series in spec.get("series", [])}
    rows = []
    for glyph in (stage.get("calibration") or {}).get("legend_markers", []):
        style = declared.get(str(glyph.get("label")), {})
        measured = {key: glyph.get(key) for key in ("shape", "fill", "color_hex", "fill_hex", "edge_hex", "size_px",
                                                     "source", "confidence")}
        rows.append({
            "label": glyph.get("label"),
            "declared": {key: style.get(key) for key in ("marker", "marker_fill", "color_hex")},
            "measured": measured,
            "mismatches": legend_markers.compare(style, glyph),
        })
    return rows


AXIS_CHECK_FAILED = "Axis range check failed"


def count_check(spec: dict, stage: dict) -> list[dict]:
    """Agent 02's expected marker counts per x range next to what Python found."""
    found = {item["label"]: item for item in stage.get("series", [])}
    rows = []
    for series in spec.get("series", []):
        points = (found.get(series.get("label")) or {}).get("points", [])
        for expected in series.get("expected_counts") or []:
            n = sum(expected["x_from"] <= float(p["x"]) <= expected["x_to"] for p in points)
            rows.append({"label": series.get("label"), "x_from": expected["x_from"], "x_to": expected["x_to"],
                         "expected": expected["count"], "found": n, "shortfall": max(0, expected["count"] - n)})
    return rows


AUDIT_KEYS = ("series_assignment_audit", "series_gate", "template_fill", "complex_regions_accepted",
              "complex_regions_rejected", "legend_markers")


def _counts(items, key):
    counts = {}
    for item in items or []:
        counts[str(item.get(key))] = counts.get(str(item.get(key)), 0) + 1
    return counts


def compact_calibration(calibration: dict) -> dict:
    """The calibration an agent can reason about; per-candidate audits stay in python_audit.json."""
    assignment = calibration.get("series_assignment_audit") or {}
    template = calibration.get("template_fill") or {}
    return {
        **{key: value for key, value in calibration.items() if key not in AUDIT_KEYS},
        "series_assignment": {
            "actions": _counts(assignment.get("candidates"), "action"),
            "groups": [{"series_labels": group.get("series_labels"), "assigned_counts": group.get("assigned_counts"),
                        "unresolved_count": group.get("unresolved_count")} for group in assignment.get("groups", [])],
        },
        "series_gate_moves": [{"from": key.split(" -> ")[0], "to": key.split(" -> ")[1], "points": n} for key, n in
                              _counts([{"k": f"{m.get('from')} -> {m.get('to')}"} for m in calibration.get("series_gate") or []],
                                      "k").items()],
        "template_fill": {key: template.get(key) for key in ("template_fit", "template_fill")},
        "complex_regions": {"accepted": len(calibration.get("complex_regions_accepted") or []),
                            "rejected": len(calibration.get("complex_regions_rejected") or [])},
        "detail": "full per-candidate audit in python_audit.json",
    }


def python_summary(spec: dict, stage: dict) -> dict:
    """What agents 02 and 03 need to know about the Python result (saved as python_summary.json).

    Kept compact on purpose: it is pasted into agent prompts.  Per-candidate audits (thousands of
    entries on a 17-series chart) go to python_audit.json instead.
    """
    calibration = stage.get("calibration") or {}
    return {
        "series_counts": {item["label"]: item["n_points"] for item in stage.get("series", [])},
        "calibration": compact_calibration(calibration),
        "proposal_quality": stage.get("proposal_quality", {}),
        "color_ambiguity_groups": [{key: group.get(key) for key in ("series_labels", "pair_distances_lab")}
                                    for group in calibration.get("color_ambiguity_groups", [])],
        "unresolved_slots": stage.get("unresolved_slots", {}),
        "complex_regions": spec.get("complex_regions", []),
        "legend_marker_check": legend_marker_check(spec, stage),
        "source_integrity": evidence.source_integrity(spec),
        "count_check": count_check(spec, stage),
        "sample_colors": calibration.get("sample_colors", []),
        "region_strategies": calibration.get("region_strategies", []),
    }


def write_python_stage(panel_dir, extraction, spec, panel: Panel, source_pdf: str):
    """Write the python stage files and its inspection images."""
    stage, rows, metadata = stage_artifacts.write_stage(
        panel_dir, "python", extraction, spec, panel.figure_id, panel.panel_id, source_pdf=source_pdf,
    )
    dense = dense_regions.write_evidence(cv2.imread(spec["image"]), stage, spec, panel_dir / "adjudicate")
    if dense:
        metadata["dense_region_evidence"] = dense
        save_json(panel_dir / "python_metadata.json", metadata)
    extract.redraw_overlay(stage, spec, str(panel_dir / "python_overlay.png"))
    recreate.recreate(stage, spec, str(panel_dir / "python_recreated.png"))
    recreate.side_by_side(str(panel_dir / "panel.png"), str(panel_dir / "python_recreated.png"),
                          str(panel_dir / "python_compare.png"))
    measured = (stage.get("calibration") or {}).get("legend_markers", [])
    sheet = panel_dir / "python_legend_markers.png"
    sheet.unlink(missing_ok=True)
    if measured:
        legend_markers.legend_sheet(cv2.imread(spec["image"]), [m["legend_box_px"] for m in measured], measured,
                                    [m["label"] for m in measured], sheet)
    return stage, rows


STALE_ON_SUCCESS = ("python_tick_check.png", "python_tick_candidates.json", "python_error.txt")
PYTHON_STAGE_FILES = ("python_overlay.png", "python_compare.png", "python_redraw_diff.png", "python_summary.json",
                      "python_hard_checks.json", "python_audit.json")


def record_error(panel_dir, panel: Panel, spec: dict, message: str) -> None:
    """Keep the error for agent 02 and draw the detected tick candidates so it can fix the mapping."""
    panel.extract_error = message
    (panel_dir / "python_error.txt").write_text(message, encoding="utf-8")
    try:
        candidates = extract.render_tick_check(spec, str(panel_dir / "python_tick_check.png"))
        save_json(panel_dir / "python_tick_candidates.json", candidates)
    except Exception:  # no axis frame -> nothing to draw; agent 02 still sees the error
        for name in ("python_tick_check.png", "python_tick_candidates.json"):
            (panel_dir / name).unlink(missing_ok=True)


class ExtractPoints(PanelStep):
    async def run_panel(self, run: DocumentRun, panel: Panel) -> str:
        panel_dir = run.panel_dir
        spec = read_json(panel_dir / "spec.json")
        try:
            extraction, _ = extract.extract(spec)
        except RuntimeError as error:  # includes TickAnchorMismatchError and axis/tick detection failures
            for name in PYTHON_STAGE_FILES:  # no stale images from an earlier round
                (panel_dir / name).unlink(missing_ok=True)
            record_error(panel_dir, panel, spec, str(error))
            return CONTINUE

        for name in STALE_ON_SUCCESS:
            (panel_dir / name).unlink(missing_ok=True)
        stage, rows = write_python_stage(panel_dir, extraction, spec, panel, run.source)
        hard = hard_checks.run(stage, spec, panel_dir, "python")
        save_json(panel_dir / "python_hard_checks.json", hard)
        save_json(panel_dir / "python_summary.json", {**python_summary(spec, stage), "hard_checks": hard})
        save_json(panel_dir / "python_audit.json", {key: (stage.get("calibration") or {}).get(key) for key in AUDIT_KEYS})
        panel.rows["python"] = len(rows)
        panel.extract_error = ""
        if not hard["passed"]:  # markers on the right pixels, numbers wrong: a calibration error
            record_error(panel_dir, panel, spec, f"{AXIS_CHECK_FAILED}: " + "; ".join(hard["problems"]))
        return CONTINUE

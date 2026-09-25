"""Agent 02 - check extraction (agent, one call per round) - the adjust-and-rebuild loop.

Input: the source panel, the Python overlay and re-plot from the Python extraction (or, when calibration
failed, the error and the drawn tick candidates) and the structure Python used.  The agent
checks calibration, series identity and coverage (schemas/agent02_check_extraction.schema.json).
Routes:
  * ``continue`` - satisfied and Python succeeded: go to agent 03;
  * ``adjust``   - not satisfied: its corrected structure is merged into spec.json and Python
    (the Python extraction) extracts again;
  * after ``max_check_rounds`` (workflow.yaml) it stops looping: a working extraction goes on
    to agent 03 flagged for human review, a failing one marks the panel failed.
Output: agent02_check_extraction.json (latest) and agent02_check_extraction_round<N>.json per
round; agent02_tick_reconciliation_round<N>.json when anchors were snapped to tick candidates.
"""

from __future__ import annotations

import copy
import json

from src.models import naming
from src.settings import CFG
from src.workflow.resources import load_workflow
from src.workflow.state import ADJUST, CONTINUE, FAILED, DocumentRun, Panel
from src.workflow.steps.base import PanelStep, read_json, save_json
from src.workflow.steps.agent01_read_chart import norm_to_px, valid_bbox_norm
from src.workflow.steps.python_extract_points import AXIS_CHECK_FAILED

STYLE_KEYS = ("label", "marker", "marker_fill", "line_style", "color_hex")


def _normalise(value):
    return "".join(naming.series_label(value).lower().split())


def match_checked_series(original, checked):
    """Pair agent 02 series with the current identities without trusting their order."""
    remaining = set(range(len(original)))
    matches = []
    for checked_index, checked_series in enumerate(checked):
        label = _normalise(checked_series.get("label"))
        label_matches = [i for i in remaining if _normalise(original[i].get("label")) == label]
        if len(label_matches) == 1:
            match = label_matches[0]
        else:

            def score(index):
                candidate = original[index]
                return sum(
                    _normalise(candidate.get(key)) == _normalise(checked_series.get(key))
                    and bool(_normalise(checked_series.get(key)))
                    for key in ("color_hex", "marker", "marker_fill", "line_style")
                )

            ranked = sorted(((score(i), i) for i in remaining), reverse=True)
            best_score = ranked[0][0] if ranked else 0
            best = [i for value, i in ranked if value == best_score]
            if best_score > 0 and len(best) == 1:
                match = best[0]
            elif checked_index in remaining:
                match = checked_index
            else:
                match = None
        if match is not None:
            remaining.remove(match)
        matches.append(match)
    return matches


def reconcile_retry_tick_anchors(check, tick_candidates):
    """Snap anchors to the exact tick candidates Python detected and the agent identified.

    This corrects image-coordinate estimation noise, not calibration disagreement: an anchor
    must stay close to one unique candidate, so a shifted tick/value mapping still fails
    the unchanged strict gates when the Python extraction extracts again.
    """
    reconciled = copy.deepcopy(check or {})
    audit = {"axes": {}}
    for axis_id, report_key, extent_key in (("x", "x_axis", "image_width_px"), ("y", "y_axis", "image_height_px")):
        anchors = list((reconciled.get(report_key) or {}).get("tick_anchors") or [])
        candidates = list((tick_candidates or {}).get(axis_id) or [])
        extent = float((tick_candidates or {}).get(extent_key) or 0)
        pixels = sorted(float(item["candidate_pixel"]) for item in candidates)
        spacings = [right - left for left, right in zip(pixels, pixels[1:]) if right > left]
        snap_limit = min(24.0, 0.2 * float(sorted(spacings)[len(spacings) // 2])) if spacings else 0.0
        used, entries = set(), []
        for anchor in anchors:
            entry = {"input": dict(anchor), "snapped": False}
            try:
                anchor_px = float(anchor["pixel_norm"]) * extent
                ranked = sorted(
                    (abs(anchor_px - float(item["candidate_pixel"])), index, item) for index, item in enumerate(candidates)
                )
            except (KeyError, TypeError, ValueError):
                ranked = []
            if ranked and ranked[0][1] not in used and ranked[0][0] <= snap_limit:
                distance, index, candidate = ranked[0]
                anchor["pixel_norm"] = float(candidate["pixel_norm"])
                used.add(index)
                entry.update(snapped=True, candidate_pixel=int(candidate["candidate_pixel"]),
                             snap_distance_px=float(distance), output=dict(anchor))
            entries.append(entry)
        audit["axes"][axis_id] = {"snap_limit_px": snap_limit, "anchors": entries}
    return reconciled, audit


def with_origin(spec, points):
    """An isotherm starts at the origin: zero uptake at zero pressure.

    When both axes are linear and printed from 0, a curve that agent 02 started to the right of
    x = 0 gets (0, 0) in front, so the steep low-pressure part is anchored.  Log axes, axes that
    start above 0 and ``extract.isotherm_origin: false`` keep the agent's points unchanged.
    """
    if not points or not CFG["extract"].get("isotherm_origin", True):
        return points
    axes = [spec.get("x") or {}, spec.get("y") or {}]
    if any(axis.get("scale", "linear") != "linear" or min(axis.get("ticks") or [1]) > 0 for axis in axes):
        return points
    return points if points[0]["x"] <= 0 else [{"x": 0.0, "y": 0.0}, *points]


def apply_extraction_settings(spec, settings, width, height):
    """Copy agent 02's per-series settings onto spec series, clamped to safe ranges."""
    by_label = {_normalise(series.get("label")): series for series in spec.get("series", [])}
    for setting in settings:
        series = by_label.get(_normalise(setting.get("label")))
        if series is None:
            continue
        tolerance = setting.get("color_tolerance")
        series["color_tolerance"] = None if tolerance is None else min(45.0, max(15.0, float(tolerance)))
        series["separate_from"] = [str(label) for label in setting.get("separate_from") or []]
        series["y_bands"] = [dict(band) for band in setting.get("y_bands") or []
                             if band["x_from"] < band["x_to"] and band["y_min"] < band["y_max"]]
        series["template_fill"] = setting.get("template_fill") if setting.get("template_fill") in ("on", "fit_only", "off") else "on"
        series["setting_reason"] = str(setting.get("reason") or "")
        sample = setting.get("sample_marker_norm")
        series["sample_px"] = ([round(float(sample[0]) * width, 1), round(float(sample[1]) * height, 1)]
                               if isinstance(sample, list) and len(sample) == 2 and all(0 <= v <= 1 for v in sample)
                               else None)
        series["expected_counts"] = [dict(item) for item in setting.get("expected_counts") or []
                                     if item["x_from"] < item["x_to"]]
        series["curve_points"] = with_origin(spec, sorted(
            ({"x": float(p["x"]), "y": float(p["y"])} for p in setting.get("curve_points") or []), key=lambda p: p["x"]))


def apply_axes_check(spec, check, width, height):
    """Merge the complete agent 02 structure into the extractor spec for the next Python round."""
    structure = spec["llm_spec"]
    for axis_id, report_key in (("x", "x_axis"), ("y", "y_axis")):
        checked = check.get(report_key) or {}
        axis = spec[axis_id]
        for key in ("title", "unit", "scale"):
            if checked.get(key) not in (None, ""):
                axis[key] = checked[key]
        if len(checked.get("ticks") or []) >= 2:
            axis["ticks"] = list(checked["ticks"])
        if len(checked.get("tick_anchors") or []) >= 2:
            axis["tick_anchors"] = list(checked["tick_anchors"])
        axis["minimum"] = checked.get("minimum")
        axis["maximum"] = checked.get("maximum")
        axis["boundary_limits_confirmed"] = bool(checked.get("boundary_limits_confirmed", False))
        structure[axis_id] = dict(axis)

    legend_norm = valid_bbox_norm(check.get("legend_bbox_norm"))
    if legend_norm is not None:
        spec["legend_bbox"] = norm_to_px(legend_norm, width, height, pad=0.01)
        structure["legend_bbox_norm"] = legend_norm
    else:
        # Optional geometry is supporting evidence; a malformed box must not block calibration.
        structure.pop("legend_bbox_norm", None)

    original = list(structure.get("series") or [])
    checked_series = list(check.get("series") or [])
    if checked_series:
        blank = {"gas": "", "species_evidence": "", "extract": False, "quantity_type": "",
                 "loading_basis": "unknown", "y_anchors": None, "n_markers_estimate": 0}
        original_specs = list(spec.get("series") or [])
        rebuilt_structure, rebuilt_spec = [], []
        for item, match in zip(checked_series, match_checked_series(original, checked_series)):
            prior = dict(original[match]) if match is not None else {**blank, "has_line": item.get("line_style") != "none"}
            prior.update({key: item.get(key, prior.get(key)) for key in STYLE_KEYS})
            prior["label"] = naming.series_label(prior.get("label"))
            prior["has_line"] = prior.get("line_style") != "none"
            rebuilt_structure.append(prior)
            old_spec = dict(original_specs[match]) if match is not None else dict(blank)
            old_spec.update({key: prior.get(key, old_spec.get(key)) for key in STYLE_KEYS})
            rebuilt_spec.append(old_spec)
        structure["series"] = rebuilt_structure
        spec["series"] = rebuilt_spec
    apply_extraction_settings(spec, check.get("extraction_settings") or [], width, height)
    spec["region_strategies"] = [
        {**region, "bbox_px": norm_to_px(box, width, height)}
        for region in check.get("region_strategies") or []
        if (box := valid_bbox_norm(region.get("bbox_norm"))) is not None
    ]
    spec["shared_x_columns"] = sorted(float(value) for value in check.get("shared_x_columns") or [])
    structure["chart_title"] = check.get("chart_title", "")
    structure["branches"] = check.get("branches", structure.get("branches", "unknown"))
    structure["complex_regions"] = list(check.get("complex_regions") or [])
    spec["complex_regions"] = structure["complex_regions"]
    spec["llm_spec"] = structure
    return spec


def _json(value):
    return json.dumps(value, indent=1, default=float)


def current_structure(spec: dict) -> dict:
    """The part of spec.json the agent can correct."""
    return {key: spec.get(key) for key in ("x", "y", "series", "complex_regions")} | {
        "legend_bbox_px": spec.get("legend_bbox"),
        "panel_size_px": spec.get("panel_bbox", [0, 0, 0, 0])[2:],
    }


def application_audit(round_no: int, max_rounds: int, answer: dict, *, state: str, reason: str = "") -> dict:
    """Record whether Agent 02's requested settings reached an extraction.

    The last bounded retry is a review round, not a hidden fourth Python run.
    Keeping the requested settings in a separate audit prevents later readers
    from confusing an agent recommendation with the spec that produced the
    saved points.
    """
    return {
        "round": int(round_no), "max_rounds": int(max_rounds), "checked": True,
        "state": state, "reason": reason,
        "proposed": {
            "extraction_settings": len(answer.get("extraction_settings") or []),
            "region_strategies": len(answer.get("region_strategies") or []),
            "shared_x_columns": len(answer.get("shared_x_columns") or []),
            "complex_regions": len(answer.get("complex_regions") or []),
        },
        "applied": state == "applied_for_next_extraction",
        "unexecuted": state == "unexecuted_retry_limit",
    }


def python_result(panel_dir, panel: Panel):
    """(status text, JSON details, extra images) describing the latest the Python extraction run."""
    if panel.extract_error.startswith(AXIS_CHECK_FAILED):
        candidates = panel_dir / "python_tick_candidates.json"
        text = (f"Python found the markers, but its NUMBERS FAILED the hard axis check: {panel.extract_error}\n\n"
                "The overlay can look perfect while the tick-to-pixel mapping is shifted or stretched. Compare the "
                "re-plotted axis in `python_compare.png` with the source, then fix `ticks` and `tick_anchors` "
                "(read the printed label at each detected tick candidate and copy its `pixel_norm`).")
        images, names = [], []
        for path, name in ((panel_dir / "python_compare.png", "the source next to a chart re-plotted from Python's data - "
                                                              "look at the axis values"),
                           (panel_dir / "python_tick_check.png", "every tick candidate Python detected"),
                           (panel_dir / "python_overlay.png", "Python's markers drawn on the panel")):
            if path.exists():
                images.append(path)
                names.append(f"{len(images) + 1}. `{path.name}` - {name}.")
        details = {"hard_checks": read_json(panel_dir / "python_hard_checks.json"),
                   "tick_candidates": read_json(candidates) if candidates.exists() else {},
                   "python_summary": read_json(panel_dir / "python_summary.json")}
        return text, details, images, "\n".join(names)
    if panel.extract_error:
        candidates = panel_dir / "python_tick_candidates.json"
        text = (f"Python FAILED this round: {panel.extract_error}\n\n"
                + ("Image 2 (`python_tick_check.png`) marks every tick candidate Python detected; the exact candidates are "
                   "listed below. Map the printed values to them." if candidates.exists()
                   else "Python could not find the axis frame, so there are no tick candidates."))
        details = read_json(candidates) if candidates.exists() else {}
        images = [panel_dir / "python_tick_check.png"] if candidates.exists() else []
        names = ["2. `python_tick_check.png` - detected tick candidates."] if images else []
    else:
        text = ("Python succeeded and passed the hard axis check. `hard_checks` gives the printed vs extracted value "
                "range per axis, the redraw score (recall = share of native marker ink covered, per series) and physics "
                "flags; `count_check` compares your expected counts with Python's. Series counts, calibration residuals "
                "and quality warnings:")
        details = read_json(panel_dir / "python_summary.json")
        images = [panel_dir / "python_compare.png", panel_dir / "python_overlay.png"]
        names = ["2. `python_compare.png` - the source next to a chart re-plotted from Python's data (check the numbers first).",
                 "3. `python_overlay.png` - Python's markers drawn on the panel with the measured legend outlines."]
        for path, name in ((panel_dir / "python_redraw_diff.png", "red = native marker ink Python did not extract, green = "
                                                                  "point on ink, blue = point without ink"),
                           (panel_dir / "python_legend_markers.png", "each legend symbol enlarged | the vector glyph "
                                                                     "Python measured from it | label, shape / fill / color")):
            if path.exists():
                images.append(path)
                names.append(f"{len(images) + 1}. `{path.name}` - {name}.")
    return text, details, images, "\n".join(names)


class CheckExtraction(PanelStep):
    async def run_panel(self, run: DocumentRun, panel: Panel) -> str:
        panel_dir = run.panel_dir
        max_rounds = load_workflow().max_check_rounds
        panel.check_rounds += 1
        round_no = panel.check_rounds
        spec = read_json(panel_dir / "spec.json")
        status, details, images, image_list = python_result(panel_dir, panel)
        answer = await self.agent.ask(
            {"round": str(round_no), "max_rounds": str(max_rounds), "image_list": image_list,
             "current_structure": current_structure(spec), "extraction_status": status, "python_result": details},
            images=[panel_dir / "panel.png", *images],
            save_to=panel_dir / f"agent02_check_extraction_round{round_no}.json",
        )
        save_json(panel_dir / "agent02_check_extraction.json", answer)
        axis_failed = [axis for axis in ("x", "y") if answer["axis_check"][axis]["result"] == "fail"]
        satisfied = answer["satisfied"] and not axis_failed  # an agent can not approve numbers it says are wrong

        if satisfied and not panel.extract_error:
            save_json(panel_dir / f"agent02_application_round{round_no}.json",
                      application_audit(round_no, max_rounds, answer, state="checked_no_adjustment",
                                        reason="agent_satisfied"))
            return CONTINUE
        if round_no >= max_rounds:
            save_json(panel_dir / f"agent02_application_round{round_no}.json",
                      application_audit(round_no, max_rounds, answer, state="unexecuted_retry_limit",
                                        reason="bounded_retry_limit_reached; spec and points remain from prior extraction"))
            if panel.extract_error:
                panel.status = FAILED
                panel.message = f"Python extraction still failing after {round_no} check rounds: {panel.extract_error}"
                save_json(panel_dir / "qa.json", {"verdict": "failed", "error": panel.message})
            else:
                panel.needs_review = True
                panel.message = f"Agent 02 not satisfied after {round_no} rounds: " + "; ".join(answer["issues"])
            return CONTINUE

        if panel.extract_error and (panel_dir / "python_tick_candidates.json").exists():
            answer, audit = reconcile_retry_tick_anchors(answer, read_json(panel_dir / "python_tick_candidates.json"))
            save_json(panel_dir / f"agent02_tick_reconciliation_round{round_no}.json", audit)
        width, height = spec["panel_bbox"][2], spec["panel_bbox"][3]
        save_json(panel_dir / "spec.json", apply_axes_check(spec, answer, width, height))
        save_json(panel_dir / f"agent02_application_round{round_no}.json",
                  application_audit(round_no, max_rounds, answer, state="applied_for_next_extraction",
                                    reason="spec_saved_before_next_python_extraction"))
        return ADJUST

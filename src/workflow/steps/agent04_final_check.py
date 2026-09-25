"""Agent 04 - final check (agent, one call per panel).

Input: the agent03 stage.  The agent first records its own source-only reading
of the panel, then judges the candidate and returns a verdict plus any last point operations
(schemas/agent04_final_check.schema.json).
Output: agent04_final_check.json, the agent04 stage (agent04_points.csv/json, agent04_metadata.json), qa.json
(deterministic sanity checks), agent04_hard_checks.json and review.html - the page where a human can drag points.
An agent04 stage that fails the hard axis-range check is blocked and the panel fails: its values
contradict the printed axis, so no verdict can publish them.
Then the workflow moves to the next panel, or to publishing after the last one.
"""

from __future__ import annotations

import pathlib

from src.tools import checks, hard_checks, review_page, stage_artifacts, qa_csv
from src.workflow.review_bundle import build_bundle, csv_text
from src.workflow.state import DONE, FAILED, REVIEW, DocumentRun, Panel
from src.workflow.steps.base import PanelStep, read_json, save_json
from src.workflow.steps.agent03_review_points import draw_stage, read_rows


def crowded_native_evidence(panel_dir, candidate=None, *, ambiguity_limit=3):
    """Return a bounded list of directly attached native crop mappings for QA."""
    evidence_dir = pathlib.Path(panel_dir) / "adjudicate"
    manifest_path = evidence_dir / "review_tiles.json"
    if not manifest_path.is_file():
        return [], []
    manifest = read_json(manifest_path)
    items = list(manifest.get("review_items") or [])
    strips = [item for item in items if item.get("item_id", "").startswith("crowded_strip:")]
    ambiguities = [item for item in items if item.get("item_type") == "ambiguity"]
    selected = []
    for item in strips + ambiguities[:ambiguity_limit]:
        image_name = item.get("source_native_tile") if item.get("item_type") == "ambiguity" else item.get("image") or item.get("native_tile")
        path = evidence_dir / str(image_name or "")
        if not image_name or not path.is_file():
            continue
        record = {
            "item_id": item.get("item_id"),
            "image": str(image_name),
            "kind": "ambiguity" if item.get("item_type") == "ambiguity" else "crowded_strip",
            "source_bbox_px": item.get("source_bbox_px"),
            "native_scale": item.get("native_scale"),
            "candidate_row_ids": item.get("candidate_row_ids") or [],
            "nearby_candidate_rows": item.get("nearby_candidate_rows") or [],
            "series_context": item.get("series_context") or [],
            "competing_series": item.get("competing_series") or [],
        }
        if candidate is not None and record.get("source_bbox_px") and len(record["source_bbox_px"]) == 4:
            x0, y0, x1, y1 = map(float, record["source_bbox_px"])
            center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
            rows = []
            for series in candidate.get("series", []):
                label = str(series.get("label") or "")
                for point in series.get("points", []):
                    px = point.get("px") or []
                    if len(px) != 2 or px[0] is None or px[1] is None:
                        continue
                    px0, py0 = float(px[0]), float(px[1])
                    if x0 <= px0 <= x1 and y0 <= py0 <= y1:
                        rows.append((
                            ((px0 - center_x) ** 2 + (py0 - center_y) ** 2) ** 0.5,
                            {"point_id": point.get("point_id"), "series_label": label,
                             "x": point.get("x"), "y": point.get("y"),
                             "source_pixel": [round(px0, 2), round(py0, 2)],
                             "evidence_type": point.get("evidence_kind") or point.get("source")},
                        ))
            rows.sort(key=lambda item: item[0])
            limit = 12 if record["kind"] == "crowded_strip" else 6
            record["nearby_candidate_rows"] = [row for _, row in rows[:limit]]
            record["candidate_row_ids"] = [row.get("point_id") for _, row in rows[:limit] if row.get("point_id")]
        selected.append((record, path))
    return [record for record, _ in selected], [path for _, path in selected]


def printed_range_mismatch(spec: dict, answer: dict) -> str:
    """Agent 04 reads the printed tick range itself; say so when it differs from the ticks Python used."""
    problems = []
    for axis in ("x", "y"):
        ticks = [float(v) for v in (spec.get(axis) or {}).get("ticks") or []]
        read = answer["axis_check"][axis]
        if ticks and (abs(read["printed_min"] - min(ticks)) > 1e-9 * max(1.0, abs(min(ticks)))
                      or abs(read["printed_max"] - max(ticks)) > 1e-6 * max(1.0, abs(max(ticks)))):
            problems.append(f"{axis}: agent 04 reads printed {read['printed_min']:g}-{read['printed_max']:g}, "
                            f"Python used {min(ticks):g}-{max(ticks):g}")
    return "; ".join(problems)


def reconcile_series_counts(answer: dict, extraction: dict, audit: dict) -> int:
    """Report actual post-edit CO2 counts and return the number of unapplied operations."""
    previous = {str(item.get("label")): item for item in answer.get("series") or [] if item.get("label")}
    reconciled = []
    for series in extraction.get("series", []):
        label = str(series.get("label"))
        report = previous.get(label, {})
        reconciled.append({
            "label": label,
            "coverage": report.get("coverage", "uncertain"),
            "final_count": len(series.get("points") or []),
            "comment": report.get("comment", ""),
        })
    answer["series"] = reconciled
    unapplied = max(0, int(audit.get("requested", 0)) - int(audit.get("applied", 0)))
    if unapplied:
        for series in answer["series"]:
            series["coverage"] = "uncertain"
            note = "One or more requested point operations were not applied; see the Python edit audit."
            series["comment"] = "; ".join(part for part in (series.get("comment", ""), note) if part)
    return unapplied


class FinalCheck(PanelStep):
    def finish(self, run: DocumentRun, route: str) -> DocumentRun:
        return run.next_panel()

    async def run_panel(self, run: DocumentRun, panel: Panel) -> None:
        panel_dir = run.panel_dir
        spec = read_json(panel_dir / "spec.json")
        candidate = read_json(panel_dir / "agent03_points.json")
        candidate_rows = read_rows(panel_dir / "agent03_points.csv")
        bundle = build_bundle(panel_dir, "agent04", schema=self.agent.schema)
        # Offline wiring preview: retain the candidate but never claim source QA passed.
        dry_report = dict(self.agent.example)
        dry_report.update(
            verdict="review", reasoning="Dry run: no source QA performed.",
            calibration=candidate["calibration"], row_count=len(candidate_rows), changes=[],
            unresolved_slots={"total": 0, "by_series": {}, "slots": []},
            series=[{"label": series["label"], "coverage": "uncertain",
                     "final_count": len(series.get("points", [])), "comment": "Dry run only."}
                    for series in candidate["series"]],
            uncertainties=["Dry run: the retained coordinates have not been reviewed."],
        )
        import json
        result = await self.agent.ask_artifacts(
            {"figure_id": panel.figure_id, "panel_id": panel.panel_id,
             "runtime_files": [bundle.name]},
            files=[bundle], images=[panel_dir / "panel.png"],
            output_dir=panel_dir / "agent04_unstructured",
            expected_files=["agent04_points.csv", "agent04_final_check.json"],
            report_name="agent04_final_check.json",
            dry_run_files={"agent04_points.csv": csv_text(candidate_rows),
                           "agent04_final_check.json": json.dumps(dry_report)},
        )
        answer = result["report"]
        final, authored_rows, audit = qa_csv.load_final_csv(
            result["artifacts"]["agent04_points.csv"], answer, candidate, spec,
            panel.figure_id, panel.panel_id,
        )
        # An incomplete inventory cannot be accepted just because its file parses.
        incomplete = any(item["coverage"] != "complete" for item in answer["series"])
        if (answer["unresolved_slots"]["total"] or incomplete or audit.get("requires_review")) and answer["verdict"] == "accept":
            answer["verdict"] = "review"
            answer["uncertainties"].append("Incomplete source coverage or inferred rows require review.")
        save_json(panel_dir / "agent04_final_check.json", answer)
        save_json(panel_dir / "agent04_final_check.provenance.json", result["provenance"])
        agent_axis_fail = [axis for axis in ("x", "y") if answer["axis_check"][axis]["result"] == "fail"]
        status = "blocked" if agent_axis_fail else "partial_review_required" if panel.needs_review else None

        def write(status):
            stage, generated_rows, metadata = stage_artifacts.write_stage(
                panel_dir, "agent04", final, spec, panel.figure_id, panel.panel_id,
                source_pdf=run.source, agent_report=answer, edit_audit=audit,
                # A panel Agent 02 never approved remains subject to review.
                status=status,
            )
            if {row["point_id"] for row in generated_rows} != {row["point_id"] for row in authored_rows}:
                raise ValueError("Stage rebuilding changed the validated QA row inventory")
            # Preserve every authored cell, including notes, uncertainty, and normalized columns.
            stage_artifacts._write_csv(panel_dir / "agent04_points.csv", authored_rows)
            curve_traces, trace_audit = stage_artifacts.demote_deleted_markers_to_curve_traces(
                candidate_rows,
                authored_rows,
                candidate,
                spec,
                panel_dir / "panel.png",
                answer["calibration"],
            )
            if curve_traces:
                stage["curve_traces"] = curve_traces
                stage_artifacts._write_json(panel_dir / "agent04_points.json", stage)
            metadata.update(authority="agent04_code_interpreter_csv", row_count=len(authored_rows),
                            authored_csv=str(result["artifacts"]["agent04_points.csv"]),
                            curve_trace_audit=trace_audit)
            save_json(panel_dir / "agent04_metadata.json", metadata)
            return stage, authored_rows, metadata

        stage, rows, metadata = write(status)
        hard = hard_checks.run(stage, spec, panel_dir, "agent04")
        save_json(panel_dir / "agent04_hard_checks.json", hard)
        if not hard["passed"]:  # values contradict the printed axis: never publish them
            stage, rows, metadata = write("blocked")
        draw_stage(panel_dir, "agent04", stage, spec)

        report = checks.run(stage)
        report.update(role="supporting_diagnostics_only", agent04_verdict=answer["verdict"],
                      rows={**panel.rows, "agent04": len(rows)}, final_operations_applied=audit.get("applied", len(audit.get("operations", []))),
                      hard_checks={"passed": hard["passed"], "problems": hard["problems"], "summary": hard_checks.summary_line(hard)})
        save_json(panel_dir / "qa.json", report)
        accepted = metadata.get("status") == "complete_against_reviewed_evidence"
        review_page.build(
            str(panel_dir / "agent04_points.json"), str(panel_dir / "panel.png"), str(panel_dir / "review.html"),
            comparison_image="agent04_compare.png",
            comparison_label="Agent 04 final reconstruction",
            additional_comparison_image="agent03_compare.png",
            status=metadata.get("status"),
            status_reason="Agent 04 accepted the final extraction." if accepted
            else "The final extraction remains available for human review.",
        )
        panel.rows["agent04"] = len(rows)
        tick_mismatch = printed_range_mismatch(spec, answer)
        if tick_mismatch:  # agent 04 reads different tick labels than Python used: a human must look
            panel.needs_review = True
        panel.verdict = answer["verdict"]
        panel.status = DONE if accepted and not panel.needs_review else REVIEW
        panel.message = "; ".join([panel.message] * panel.needs_review + list(answer.get("uncertainties") or []))
        if tick_mismatch:
            panel.message = "; ".join(part for part in (panel.message, tick_mismatch) if part)
        if agent_axis_fail:
            panel.message = "Agent 04 axis check failed (" + ", ".join(agent_axis_fail) + "): " + "; ".join(
                answer["axis_check"][axis]["reason"] for axis in agent_axis_fail)
        if not hard["passed"]:
            panel.status = FAILED
            panel.message = "Blocked by the hard axis check: " + "; ".join(hard["problems"])
        from src.workflow.artifact_agent import consume_recovery
        consume_recovery(result)

"""Agent 03 - review points (agent, one call per panel).

Input: the python stage that agent 02 accepted. The agent sees the Python overview overlay and
zoomed native evidence tiles for uncertain markers and complex regions, and returns point
operations (schemas/agent03_review_points.schema.json).  src/tools/stage_edits.py applies them
mechanically with the calibrated axes, keeping an audit of every operation.
Output: agent03_review_points.json and the agent03 stage (agent03_points.csv/json, agent03_metadata.json,
agent03_edit_audit.json, overlay / re-plot / side-by-side / contact-sheet images) and
agent03_hard_checks.json + agent03_redraw_diff.png for agent 04.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import logging
from collections import OrderedDict

from src.settings import CFG
from src.tools import adjudicate_tiles, extract, hard_checks, missing_strips, recreate, stage_artifacts, stage_edits
from src.workflow.state import DocumentRun, Panel
from src.workflow.steps.base import PanelStep, read_json, save_json
from src.workflow.steps.agent00_find_figures import request_fingerprint

MAX_IMAGES = 16   # overview overlay + up to 15 focused evidence images per review call
MAX_REVIEW_ITEMS = 40
MAX_CANDIDATE_ROWS = 128
PROGRESS_SECONDS = 30
log = logging.getLogger("chart_extract")


async def _await_batch(call, batch_no, batch_count, timeout_seconds):
    """Await one review call with visible progress and prompt cancellation."""
    task = asyncio.create_task(call)
    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        while True:
            elapsed = loop.time() - started
            remaining = timeout_seconds - elapsed
            if remaining <= 0:
                task.cancel()
                raise TimeoutError(
                    f"agent03-extraction batch {batch_no}/{batch_count} "
                    f"exceeded its {timeout_seconds}s hard timeout"
                )
            done, _ = await asyncio.wait(
                {task}, timeout=min(PROGRESS_SECONDS, remaining)
            )
            if done:
                return await task
            elapsed = round(loop.time() - started)
            log.info(
                "agent03-extraction: batch %d/%d still waiting for Foundry (%ds elapsed)",
                batch_no,
                batch_count,
                elapsed,
            )
    except BaseException:
        task.cancel()
        raise


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def draw_stage(panel_dir, stage_name, stage, spec):
    """Overlay, re-plot and side-by-side images for the agent03 or agent04 stage."""
    extract.redraw_overlay(stage, spec, str(panel_dir / f"{stage_name}_overlay.png"))
    recreate.recreate(
        stage,
        spec,
        str(panel_dir / f"{stage_name}_recreated.png"),
    )
    recreate.side_by_side(str(panel_dir / "panel.png"), str(panel_dir / f"{stage_name}_recreated.png"),
                          str(panel_dir / f"{stage_name}_compare.png"))


def _review_groups(manifest, evidence_dir):
    """Group ledger entries by native evidence without dropping box/slot IDs."""
    items = list(manifest.get("review_items") or [])
    if not items:  # Backward-compatible manifests written by older runs.
        for tile in manifest.get("tiles", []):
            tile_id = tile.get("tile_id")
            items.append({"item_id": f"tile:{tile_id}", "item_type": "tile", "tile_id": tile_id,
                          "native_tile": tile.get("native_tile"), "candidate_tile": tile.get("candidate_tile")})
            for box in tile.get("boxes", []):
                items.append({"item_id": f"box:{tile_id}:{box.get('box')}", "item_type": "box",
                              "tile_id": tile_id, "box": box.get("box"), "point_id": box.get("point_id"),
                              "series_label": box.get("series_label"), "native_tile": tile.get("native_tile"),
                              "candidate_tile": tile.get("candidate_tile")})
        for name in manifest.get("dense_region_evidence", []):
            items.append({"item_id": f"evidence:{name}", "item_type": "evidence", "native_tile": name})
        for name in manifest.get("ambiguity_native_evidence_tiles", []):
            items.append({"item_id": f"ambiguity:{name}", "item_type": "ambiguity", "native_tile": name})

    groups = OrderedDict()
    for item in items:
        native_name = item.get("source_native_tile") if item.get("item_type") == "ambiguity" else None
        native_name = native_name or item.get("native_tile") or item.get("diagnostic_tile")
        native_path = evidence_dir / native_name if native_name else None
        if native_path is not None and native_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            # JSON diagnostics belong in the ledger but cannot be sent as an
            # image input.  The item is therefore retained and recorded as
            # unresolved unless a later reviewer covers it textually.
            native_path = None
        # A missing file still receives a ledger entry, and is therefore
        # deterministically unresolved rather than silently disappearing.
        key = str(native_name or "__no_native_evidence__")
        group = groups.setdefault(key, {"native": native_path, "candidates": [], "items": []})
        group["items"].append(item)
        candidate_name = item.get("candidate_tile")
        candidate = evidence_dir / candidate_name if candidate_name else None
        if candidate and candidate.is_file() and candidate not in group["candidates"]:
            group["candidates"].append(candidate)
    result = []
    for group in groups.values():
        if len(group["items"]) <= MAX_REVIEW_ITEMS:
            result.append(group)
            continue
        for offset in range(0, len(group["items"]), MAX_REVIEW_ITEMS):
            result.append({**group, "items": group["items"][offset:offset + MAX_REVIEW_ITEMS]})
    return result


def _batch_groups(groups, evidence_slots=MAX_IMAGES - 1, max_items=MAX_REVIEW_ITEMS,
                  max_candidate_rows=MAX_CANDIDATE_ROWS):
    """Pack image groups under image, ledger-item and candidate-row limits."""
    batches, current, used_images, item_count = [], [], 0, 0
    candidate_ids = set()
    for group in groups:
        weight = (1 if group["native"] is not None and group["native"].is_file() else 0) + sum(
            candidate.is_file() for candidate in group["candidates"]
        )
        group_items = len(group["items"])
        group_ids = {
            str(point_id)
            for item in group["items"]
            for point_id in (item.get("candidate_row_ids") or [])
            if point_id
        }
        group_ids.update(
            str(row.get("point_id")) for item in group["items"]
            for row in (item.get("nearby_candidate_rows") or []) if row.get("point_id")
        )
        # A group with no existing file remains in a batch for its unresolved
        # ledger entries; a pathological group is never allowed to overflow.
        would_overflow = (
            used_images + weight > evidence_slots
            or item_count + group_items > max_items
            or len(candidate_ids | group_ids) > max_candidate_rows
        )
        if current and would_overflow:
            batches.append(current)
            current, used_images, item_count, candidate_ids = [], 0, 0, set()
        current.append(group)
        used_images += min(weight, evidence_slots)
        item_count += group_items
        candidate_ids.update(group_ids)
        if used_images >= evidence_slots or item_count >= max_items or len(candidate_ids) >= max_candidate_rows:
            batches.append(current)
            current, used_images, item_count, candidate_ids = [], 0, 0, set()
    if current or not batches:
        batches.append(current)
    return batches


def _batch_images(batch, review_overlay):
    """Attach the overview and all selected evidence, without repeating panel.png."""
    images = [review_overlay]
    attached = []
    for group in batch:
        for path in [group["native"], *group["candidates"]]:
            if path is None or not path.is_file() or path in images:
                continue
            images.append(path)
            attached.append(path.name)
    if len(images) > MAX_IMAGES:
        raise ValueError("review batch exceeds the image limit; evidence must not be silently dropped")
    return images, attached


def _review_check_summary(check):
    """Keep only the Agent 02 context needed for evidence review."""
    check = check if isinstance(check, dict) else {}
    styles = []
    for series in check.get("series") or []:
        styles.append({key: series.get(key) for key in (
            "label", "marker", "marker_fill", "line_style", "color_hex", "notes",
        ) if series.get(key) is not None})
    return {
        "satisfied": check.get("satisfied"),
        "structure_ok": check.get("structure_ok"),
        "axis_check": check.get("axis_check"),
        "issues": check.get("issues") or [],
        "series": styles,
        "complex_regions": check.get("complex_regions") or [],
        "notes": check.get("notes") or "",
    }


def _review_python_evidence(summary, items):
    """Bound per-batch Python diagnostics while retaining relevant leads."""
    summary = summary if isinstance(summary, dict) else {}
    labels = {
        str(label) for item in items
        for label in ([item.get("series_label")] + list(item.get("competing_series") or []))
        if label
    }
    labels.update(
        str(row.get("series_label")) for item in items
        for row in (item.get("nearby_candidate_rows") or []) if row.get("series_label")
    )
    counts = summary.get("series_counts") or {}
    calibration = summary.get("calibration") or {}
    hard = summary.get("hard_checks") or {}
    axis_range = hard.get("axis_range") or {}
    compact_hard = {
        "passed": hard.get("passed"),
        "problems": list(hard.get("problems") or [])[:8],
        "axes": {
            axis: {key: (axis_range.get(axis) or {}).get(key) for key in ("unit", "scale", "printed", "data", "points_outside", "passed")
                   if key in (axis_range.get(axis) or {})}
            for axis in ("x", "y")
        },
    }
    redraw = (hard.get("redraw") or {}).get("series") or {}
    quality = summary.get("proposal_quality") or {}
    warnings = [
        {key: warning.get(key) for key in ("check", "series", "detail") if warning.get(key) is not None}
        for warning in (quality.get("warnings") or [])
    ]
    legend_checks = []
    for check in summary.get("legend_marker_check") or []:
        if not labels or check.get("label") in labels:
            legend_checks.append({
                "label": check.get("label"),
                "declared": check.get("declared"),
                "measured": check.get("measured"),
                "mismatches": check.get("mismatches") or [],
            })
    ambiguity_groups = []
    for group in summary.get("color_ambiguity_groups") or []:
        group_labels = list(group.get("series_labels") or [])
        if not labels or labels.intersection(group_labels):
            ambiguity_groups.append({
                "series_labels": group_labels,
                "pair_distances_lab": group.get("pair_distances_lab"),
                "threshold_lab": group.get("threshold_lab"),
            })
    return {
        "series_counts": counts,
        "calibration": {
            "frame_px": calibration.get("frame_px"),
            "axis_models": calibration.get("axis_models"),
            "tick_fit_resid_x_px": calibration.get("tick_fit_resid_x_px"),
            "tick_fit_resid_y_px": calibration.get("tick_fit_resid_y_px"),
        },
        "proposal_quality": {"review_required": quality.get("review_required"), "warnings": warnings},
        "color_ambiguity_groups": ambiguity_groups,
        "unresolved_slots": {
            "total": (summary.get("unresolved_slots") or {}).get("total", 0),
            "by_series": (summary.get("unresolved_slots") or {}).get("by_series", {}),
        },
        "legend_marker_check": legend_checks[:16],
        "redraw": {label: redraw[label] for label in sorted(labels) if label in redraw},
        "hard_checks": compact_hard,
    }


def _batch_manifest(manifest, items, attached, batch_no, batch_count):
    """Keep each request's manifest bounded; the full ledger is saved on disk."""
    selected_tiles = {item.get("tile_id") for item in items if item.get("tile_id")}
    tiles = [
        {key: tile.get(key) for key in ("tile_id", "native_tile", "candidate_tile", "reason") if key in tile}
        for tile in manifest.get("tiles", []) if tile.get("tile_id") in selected_tiles
    ]
    return {
        "schema_version": manifest.get("schema_version", "1.0"),
        "authority": manifest.get("authority", "supporting_evidence_only"),
        "native_image": manifest.get("native_image"),
        "panel_size_px": manifest.get("panel_size_px"),
        "legend": manifest.get("legend"),
        "tiles": tiles,
        "dense_region_evidence": [name for name in manifest.get("dense_region_evidence", []) if any(
            name in (item.get("native_tile"), item.get("diagnostic_tile")) for item in items
        )],
        "ambiguity_native_evidence_tiles": [
            item.get("source_native_tile") or item.get("native_tile")
            for item in items if item.get("item_type") == "ambiguity"
        ],
        "ambiguity_contact_sheets": [
            sheet for sheet in manifest.get("ambiguity_contact_sheets", [])
            if sheet.get("native_tile") in attached
        ],
        "review_items": items,
        "review_item_ids": [item.get("item_id") for item in items],
        "tiles_attached": attached,
        "batch_id": f"{batch_no:03d}",
        "batch_count": batch_count,
        "instructions": manifest.get("instructions", ""),
    }


def _load_batch_checkpoint(path, checkpoint, fingerprint, validator):
    """Reuse only a schema-valid answer to this exact prompt, data and images."""
    if checkpoint.get("batch_fingerprints", {}).get(path.name) != fingerprint or not path.is_file():
        return None
    try:
        answer = read_json(path)
        return None if list(validator.iter_errors(answer)) else answer
    except (OSError, ValueError, TypeError):
        return None


def _merge_answer(responses, expected_items, extraction):
    """Consolidate batches deterministically and close all omitted coverage."""
    operations, seen_ops, uncertainties = [], set(), []
    series_by_label = OrderedDict()
    coverage_by_id = {}
    verdicts = []
    reasoning = []
    for batch_no, answer in enumerate(responses, start=1):
        verdicts.append(answer.get("verdict", "review"))
        if answer.get("reasoning"):
            reasoning.append(f"batch {batch_no}: {answer['reasoning']}")
        for operation_no, operation in enumerate(answer.get("operations") or [], start=1):
            # Every review call commonly starts numbering at ``op-1``.  An
            # operation ID is therefore unique only inside its batch.  Use
            # the operation content for true duplicate detection and prefix
            # retained IDs with the batch number before the shared edit layer
            # records them in one audit.
            semantic_key = json.dumps(
                {key: value for key, value in operation.items() if key != "operation_id"},
                sort_keys=True,
                ensure_ascii=False,
            )
            if semantic_key not in seen_ops:
                seen_ops.add(semantic_key)
                operation = dict(operation)
                local_id = str(operation.get("operation_id") or f"op-{operation_no}")
                operation["operation_id"] = f"batch-{batch_no:03d}-{local_id}"
                operations.append(operation)
        for item in answer.get("uncertainties") or []:
            if item not in uncertainties:
                uncertainties.append(item)
        for item in answer.get("series") or []:
            label = str(item.get("label") or "")
            if label:
                series_by_label[label] = item
        for item in answer.get("coverage") or []:
            item_id = item.get("item_id")
            if item_id and item_id not in coverage_by_id:
                coverage_by_id[item_id] = item

    coverage = []
    unresolved_labels = set()
    for item in expected_items:
        item_id = item.get("item_id")
        disposition = coverage_by_id.get(item_id)
        if disposition is None:
            disposition = {"item_id": item_id, "item_type": item.get("item_type", "evidence"),
                           "status": "unresolved", "detail": "not returned by the review batch"}
        else:
            # Do not allow a model to change the manifest's item type or add
            # an arbitrary ID to make coverage appear complete.
            disposition = {"item_id": item_id, "item_type": item.get("item_type", "evidence"),
                           "status": disposition.get("status") if disposition.get("status") in ("reviewed", "unresolved") else "unresolved",
                           "detail": str(disposition.get("detail") or "")}
        coverage.append(disposition)
        if disposition["status"] == "unresolved":
            if item.get("series_label"):
                unresolved_labels.add(str(item["series_label"]))
            uncertainties.append(f"Unresolved review item {item_id}: {disposition['detail']}")

    # Keep one coverage row per eligible series even when a model batch did
    # not return series metadata (e.g. a checkpoint from an older example).
    for series in extraction.get("series", []):
        label = str(series.get("label") or "")
        if not label or not series.get("extract"):
            continue
        existing = series_by_label.get(label, {})
        series_by_label[label] = {
            "label": label,
            "coverage": "incomplete" if label in unresolved_labels else existing.get("coverage", "complete"),
            "final_count": int(existing.get("final_count", len(series.get("points", [])))),
            "comment": str(existing.get("comment") or ("Some evidence items remain unresolved." if label in unresolved_labels else "All assigned review evidence was covered.")),
        }
    verdict = "reject" if "reject" in verdicts else ("review" if "review" in verdicts or unresolved_labels or any(c["status"] == "unresolved" for c in coverage) else "accept")
    return {
        "reasoning": "\n".join(reasoning),
        "verdict": verdict,
        "series": list(series_by_label.values()),
        "operations": operations,
        "uncertainties": list(dict.fromkeys(uncertainties)),
        "coverage": coverage,
    }


def _recovery_application_summary(requested_operations, application_audit):
    """Keep source confirmation distinct from mechanically applied edits."""
    operations = list((application_audit or {}).get("operations") or [])
    applied = [item for item in operations if item.get("status") == "applied"]
    rejected = [item for item in operations if item.get("status") != "applied"]
    return {
        "requested_count": len(requested_operations),
        "requested_operations": requested_operations,
        "applied_count": len(applied),
        "applied_operations": applied,
        "rejected_count": len(rejected),
        "rejected_operations": rejected,
    }


class ReviewPoints(PanelStep):
    async def run_panel(self, run: DocumentRun, panel: Panel) -> None:
        panel_dir = run.panel_dir
        spec = read_json(panel_dir / "spec.json")
        python_stage = read_json(panel_dir / "python_points.json")
        check = read_json(panel_dir / "agent02_check_extraction.json")
        # Optional deterministic recovery sits after Agent 02's stabilized
        # check and before any Agent 03 edits.  It is disabled by default and
        # can only emit an add operation after native pixel confirmation.
        recovery_ops, recovery_audit = missing_strips.recover_confirmed_slots(
            python_stage,
            spec,
            check,
            enabled=bool(CFG["extract"].get("missing_strip_recovery", False)),
        )
        review_stage = python_stage
        recovery_application_audit = {"operations": []}
        if recovery_ops:
            review_stage, recovery_application_audit = stage_edits.apply_agent_edits(
                python_stage, spec, {"operations": recovery_ops}, "missing_strip_recovery"
            )
        recovery_application = _recovery_application_summary(recovery_ops, recovery_application_audit)
        save_json(panel_dir / "agent03_missing_strip_recovery.json", {
            "enabled": bool(CFG["extract"].get("missing_strip_recovery", False)),
            "operations": recovery_ops,
            "audit": recovery_audit,
            "application": recovery_application,
        })
        # Build canonical point IDs and rows for the exact stage the reviewer
        # sees. Recovery additions remain in a separate stage; the original
        # python proposal files are left intact for auditability.
        review_stage, all_rows, _ = stage_artifacts.build_candidate(
            review_stage, spec, panel.figure_id, panel.panel_id, source_pdf=run.source,
        )
        review_overlay = panel_dir / "adjudicate" / "review_python_overlay.png"
        review_overlay.parent.mkdir(parents=True, exist_ok=True)
        extract.redraw_overlay(review_stage, spec, str(review_overlay))
        manifest_path, manifest = adjudicate_tiles.build_review_tiles(
            review_stage, spec, panel_dir / "adjudicate", check.get("complex_regions", []),
            candidate_overlay=review_overlay,
        )
        bbox = spec["panel_bbox"]
        manifest["panel_size_px"] = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
        all_items = list(manifest.get("review_items") or [])
        # One complete file review. Code Interpreter can open every tile, create
        # additional native crops, and inspect the full table without prompt batching.
        from src.workflow.review_bundle import build_bundle

        save_json(manifest_path, manifest)
        bundle = build_bundle(panel_dir, "agent03", schema=self.agent.schema,
                              candidate=review_stage, rows=all_rows)
        result = await self.agent.ask_artifacts(
            {"figure_id": panel.figure_id, "panel_id": panel.panel_id,
             "runtime_files": [bundle.name]},
            files=[bundle], images=[panel_dir / "panel.png"],
            output_dir=panel_dir / "agent03_unstructured",
            expected_files=["agent03_review_points.json"],
            report_name="agent03_review_points.json",
            dry_run_files={"agent03_review_points.json": json.dumps(self.agent.example)},
        )
        answer = _merge_answer([result["report"]], all_items, review_stage)
        candidate, audit = stage_edits.apply_agent_edits(review_stage, spec, answer, self.id)
        rejected_edits = [item for item in audit["operations"] if item["status"] != "applied"]
        if rejected_edits:
            if answer["verdict"] == "accept":
                answer["verdict"] = "review"
            answer["uncertainties"].extend(
                f"Edit {item['operation_id']} was not applied: {item['detail']}" for item in rejected_edits
            )
        if recovery_ops:
            audit["recovery"] = {
                "actor": "missing_strip_recovery",
                "source_audit": recovery_audit,
                **recovery_application,
            }
        # Counts are consolidated after all batch operations, rather than
        # trusting a per-batch count that only describes that batch's rows.
        candidate_counts = {
            str(series.get("label")): len(series.get("points", []))
            for series in candidate.get("series", [])
        }
        for series in answer.get("series", []):
            if series.get("label") in candidate_counts:
                series["final_count"] = candidate_counts[series["label"]]
        save_json(panel_dir / "agent03_review_points.json", answer)
        save_json(panel_dir / "agent03_review_points.provenance.json", result["provenance"])
        stage, rows, _ = stage_artifacts.write_stage(
            panel_dir, "agent03", candidate, spec, panel.figure_id, panel.panel_id,
            source_pdf=run.source, agent_report=answer, edit_audit=audit,
        )
        draw_stage(panel_dir, "agent03", stage, spec)
        recreate.contact_sheet(str(panel_dir / "panel.png"), str(panel_dir / "agent03_overlay.png"),
                               str(panel_dir / "agent03_recreated.png"), str(panel_dir / "agent03_contact_sheet.png"))
        save_json(panel_dir / "agent03_hard_checks.json", hard_checks.run(stage, spec, panel_dir, "agent03"))
        panel.rows["agent03"] = len(rows)
        from src.workflow.artifact_agent import consume_recovery
        consume_recovery(result)

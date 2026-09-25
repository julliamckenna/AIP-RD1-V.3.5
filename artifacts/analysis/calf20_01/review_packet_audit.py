"""Offline audit of reviewer inputs using saved calf20_01 outputs; no model calls."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.tools import adjudicate_tiles, dense_regions, stage_artifacts, stage_edits
from src.workflow.agents import StepAgent
from src.workflow.resources import load_workflow
from src.workflow.steps.agent03_review_points import (
    _review_groups, _batch_groups, _batch_manifest,
    _review_check_summary, _review_python_evidence,
)
from src.workflow.steps.agent04_final_check import crowded_native_evidence


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    report = {}
    agent = StepAgent(load_workflow().steps["agent03_review_points"].agent)
    for panel_id in ("figp1a", "figp1b"):
        source = ROOT / "artifacts/runs/calf20_01" / panel_id
        out = Path(__file__).parent / "packet_replay" / panel_id / "adjudicate"
        out.mkdir(parents=True, exist_ok=True)
        spec = read(source / "spec.json")
        check = read(source / "agent02_check_extraction.json")
        summary = read(source / "python_summary.json")
        stage, rows, _ = stage_artifacts.build_candidate(
            read(source / "python_points.json"), spec, panel_id, panel_id[-1]
        )
        panel = cv2.imread(spec["image"])
        dense_regions.write_evidence(panel, stage, spec, out)
        path, manifest = adjudicate_tiles.build_review_tiles(
            stage, spec, out, check.get("complex_regions", [])
        )
        batches = _batch_groups(_review_groups(manifest, path.parent))
        items_all = manifest["review_items"]
        packets = []
        for number, batch in enumerate(batches, 1):
            items = [item for group in batch for item in group["items"]]
            images = []
            ids = set()
            for group in batch:
                for image in [group["native"], *group["candidates"]]:
                    if image and image.is_file() and image.name not in images:
                        images.append(image.name)
            for item in items:
                if item.get("point_id"):
                    ids.add(str(item["point_id"]))
                ids.update(str(pid) for pid in item.get("candidate_row_ids", []) if pid)
                ids.update(str(row["point_id"]) for row in item.get("nearby_candidate_rows", []) if row.get("point_id"))
            selected = [row for row in rows if row["point_id"] in ids]
            values = {
                "figure_id": panel_id, "panel_id": panel_id[-1],
                "python_rows": stage_edits.rows_table(selected),
                "python_evidence": _review_python_evidence(summary, items),
                "agent02_check_extraction": _review_check_summary(check),
                "tile_manifest": _batch_manifest(manifest, items, images, number, len(batches)),
            }
            prompt = agent.prompt(values)
            packets.append({"batch": number, "items": len(items), "rows": len(selected),
                            "images": len(images) + 2, "prompt_characters": len(prompt),
                            "missing_row_ids": sorted(ids - {row["point_id"] for row in selected})})
        crop_checks = []
        for item in items_all:
            if item.get("kind") != "crowded_strip":
                continue
            x0, y0, x1, y1 = item["source_bbox_px"]
            actual = cv2.imread(str(out / item["native_tile"]))
            scale = item["native_scale"]
            expected = cv2.resize(panel[y0:y1, x0:x1], None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_NEAREST)
            crop_checks.append({"item_id": item["item_id"], "bbox": item["source_bbox_px"],
                                "shape": list(actual.shape), "exact_source_pixels": np.array_equal(actual, expected)})
        ambiguity = [item for item in items_all if item.get("item_type") == "ambiguity"]
        qa_stage_path = source / "agent03_points.json"
        qa_stage = read(qa_stage_path) if qa_stage_path.is_file() else stage
        qa_records, qa_paths = crowded_native_evidence(out.parent, qa_stage)
        qa_ids = {point["point_id"] for series in qa_stage["series"] for point in series["points"]}
        qa_bad_ids = sorted({pid for item in qa_records for pid in item["candidate_row_ids"]} - qa_ids)
        unmapped_slots = [item["item_id"] for item in items_all
                          if item.get("item_type") == "slot"
                          and not (out / str(item.get("native_tile") or "")).is_file()]
        assert not qa_bad_ids, qa_bad_ids
        assert not unmapped_slots, unmapped_slots
        assert all(item["exact_source_pixels"] for item in crop_checks)
        assert all(not packet["missing_row_ids"] for packet in packets)
        report[panel_id] = {
            "batch_count": len(batches), "review_items": len(items_all), "packets": packets,
            "ambiguity_items": len(ambiguity),
            "ambiguity_without_bbox": [item["item_id"] for item in ambiguity if not item.get("source_bbox_px")],
            "strip_crops": crop_checks,
            "slots_without_focused_crop": unmapped_slots,
            "qa_native_crop_count": len(qa_paths), "qa_invalid_row_ids": qa_bad_ids,
            "qa_stage": "saved agent03" if qa_stage_path.is_file() else "Python proposal (panel B review did not finish)",
        }
    output = Path(__file__).parent / "review_packet_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({panel: {
        "batches": data["batch_count"], "items": data["review_items"],
        "prompt_characters_min": min(p["prompt_characters"] for p in data["packets"]),
        "prompt_characters_max": max(p["prompt_characters"] for p in data["packets"]),
        "max_images": max(p["images"] for p in data["packets"]),
        "qa_native_crops": data["qa_native_crop_count"],
        "ambiguity_without_bbox": data["ambiguity_without_bbox"],
        "slots_without_focused_crop": data["slots_without_focused_crop"],
    } for panel, data in report.items()}, indent=2))


if __name__ == "__main__":
    main()

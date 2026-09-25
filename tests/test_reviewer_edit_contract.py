"""Reviewer operation contracts and Python's post-edit rebuild path."""

import copy
import csv
import json
import asyncio
import shutil

import cv2
import numpy as np
from jsonschema import Draft202012Validator

from src.tools import hard_checks, stage_artifacts, stage_edits, suspects
from src.workflow.resources import load_json, load_schema
from src.workflow.steps.agent03_review_points import draw_stage
from src.workflow.steps.agent04_final_check import FinalCheck, reconcile_series_counts
from src.workflow.resources import load_workflow
from src.workflow.state import DocumentRun, Panel
from src.workflow.steps.base import read_json, save_json


def test_agent03_file_schema_restricts_each_action_and_requires_native_evidence():
    schema = load_schema("schemas/agent03_review_points.schema.json")
    Draft202012Validator.check_schema(schema)
    valid = load_json("examples/agent03_review_points.example.json")
    validator = Draft202012Validator(schema)
    assert not list(validator.iter_errors(valid))

    def accepts(operation):
        answer = copy.deepcopy(valid)
        answer["operations"] = [operation]
        return not list(validator.iter_errors(answer))

    base = {
        "operation_id": "op",
        "point_id": "pt-1",
        "series_label": None,
        "x_norm": 0.4,
        "y_norm": 0.5,
        "overlap": False,
        "reason": "Source-backed correction.",
        "source_evidence": "Source marker centre at panel pixel (40, 50).",
        "evidence_kind": "native_visible",
        "uncertainty_px": 1.0,
        "evidence_ref": "panel.png",
    }
    assert accepts({**base, "action": "move"})
    assert not accepts({**base, "action": "add"})  # add needs null point_id and a series label
    assert accepts({**base, "action": "add", "point_id": None, "series_label": "CO2 273K"})
    assert accepts({**base, "action": "add", "point_id": None, "series_label": "CO2 273K",
                    "evidence_kind": "partial_marker", "uncertainty_px": 2.5, "evidence_ref": "tile:1"})
    assert not accepts({**base, "action": "add", "point_id": None, "series_label": "CO2 273K", "evidence_ref": " "})
    assert not accepts({**base, "action": "add", "point_id": None, "series_label": "CO2 273K", "uncertainty_px": None})
    assert accepts({**base, "action": "delete", "x_norm": None, "y_norm": None,
                    "evidence_kind": "native_absence", "uncertainty_px": None})
    assert not accepts({**base, "action": "delete", "point_id": "  ", "x_norm": None, "y_norm": None,
                        "evidence_kind": "native_absence", "uncertainty_px": None})
    assert accepts({**base, "action": "reassign", "series_label": "CO2 273K", "x_norm": None,
                    "y_norm": None, "uncertainty_px": None})
    assert not accepts({**base, "action": "reassign", "series_label": "CO2 273K", "x_norm": None})
    assert not accepts({**base, "action": "move", "source_evidence": "   "})


def test_agent04_candidate_index_contains_unflagged_and_flagged_points():
    rows = [
        {"point_id": "ordinary-1", "series_name": "CO2 273K", "x": "1", "y": "1",
         "source_pixel_x": "10", "source_pixel_y": "20", "evidence_type": "visible_marker"},
        {"point_id": "ordinary-2", "series_name": "CO2 273K", "x": "2", "y": "1",
         "source_pixel_x": "20", "source_pixel_y": "20", "evidence_type": "visible_marker"},
        {"point_id": "template-3", "series_name": "CO2 273K", "x": "3", "y": "1",
         "source_pixel_x": "30", "source_pixel_y": "20", "evidence_type": "template_fill"},
    ]
    spec = {"x": {"ticks": [0, 10]}, "y": {"ticks": [0, 10]}, "series": []}
    table = suspects.suspects_table(rows, spec)
    indexed = list(csv.DictReader(table.splitlines()))
    assert [row["point_id"] for row in indexed] == ["ordinary-1", "ordinary-2", "template-3"]
    assert indexed[0]["why"] == indexed[1]["why"] == ""
    assert "inferred" in indexed[2]["why"]


def _test_stage(tmp_path):
    image = np.full((100, 100, 3), 255, np.uint8)
    cv2.circle(image, (20, 80), 4, (220, 40, 30), -1)
    cv2.circle(image, (70, 20), 4, (220, 40, 30), -1)
    cv2.circle(image, (80, 70), 4, (30, 40, 220), -1)
    image_path = tmp_path / "panel.png"
    assert cv2.imwrite(str(image_path), image)
    red_lab = cv2.cvtColor(np.uint8([[[30, 40, 220]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float).tolist()
    blue_lab = cv2.cvtColor(np.uint8([[[220, 40, 30]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float).tolist()
    extraction = {
        "calibration": {
            "frame_px": [0, 0, 100, 100],
            "unit_per_px_x": 1,
            "unit_per_px_y": 1,
            "axis_models": {
                "x": {"scale": "linear", "slope": 1, "intercept": 0},
                "y": {"scale": "linear", "slope": -1, "intercept": 100},
            },
        },
        "series": [
            {"label": "CO2 273K", "gas": "CO2", "extract": True, "marker_radius_px": 2,
             "color_lab": blue_lab, "points": [
                 {"point_id": "a-1", "px": [20, 80], "x": 20, "y": 20, "source": "visible_marker", "confidence": 0.9},
                 {"point_id": "a-2", "px": [70, 20], "x": 70, "y": 80, "source": "visible_marker", "confidence": 0.9},
             ]},
            {"label": "CO2 298K", "gas": "CO2", "extract": True, "marker_radius_px": 2,
             "color_lab": red_lab, "points": [
                 {"point_id": "b-1", "px": [80, 70], "x": 80, "y": 30, "source": "visible_marker", "confidence": 0.9},
             ]},
        ],
    }
    spec = {
        "image": str(image_path), "panel_bbox": [0, 0, 100, 100],
        "x": {"ticks": [0, 50, 100], "unit": "kPa", "title": "Pressure", "scale": "linear"},
        "y": {"ticks": [0, 50, 100], "unit": "mmol/g", "title": "Uptake", "scale": "linear"},
        "series": [
            {"label": "CO2 273K", "gas": "CO2", "extract": True, "marker": "circle", "marker_fill": "filled", "color_hex": "#1e28dc",
             "curve_points": [{"x": 20, "y": 20}, {"x": 70, "y": 80}]},
            {"label": "CO2 298K", "gas": "CO2", "extract": True, "marker": "square", "marker_fill": "filled", "color_hex": "#dc281e",
             "curve_points": [{"x": 0, "y": 30}, {"x": 100, "y": 30}]},
        ],
        "llm_spec": {"series": [
            {"label": "CO2 273K", "gas": "CO2", "species_evidence": "CO2 label in source legend", "marker": "circle", "marker_fill": "filled", "color_hex": "#1e28dc"},
            {"label": "CO2 298K", "gas": "CO2", "species_evidence": "CO2 label in source legend", "marker": "square", "marker_fill": "filled", "color_hex": "#dc281e"},
        ]},
    }
    response = {"operations": [
        {"operation_id": "add", "action": "add", "point_id": None, "series_label": "CO2 273K", "x_norm": 0.4, "y_norm": 0.6,
         "evidence_kind": "native_visible", "uncertainty_px": 1.0, "evidence_ref": "panel.png", "source_evidence": "native marker at (40, 60)"},
        {"operation_id": "move", "action": "move", "point_id": "a-1", "x_norm": 0.22, "y_norm": 0.8,
         "evidence_kind": "native_visible", "uncertainty_px": 1.0, "evidence_ref": "panel.png", "source_evidence": "native marker at (22, 80)"},
        {"operation_id": "delete", "action": "delete", "point_id": "a-2", "uncertainty_px": None,
         "evidence_kind": "native_absence", "evidence_ref": "panel.png", "source_evidence": "no marker at (70, 20)"},
        {"operation_id": "reassign", "action": "reassign", "point_id": "b-1", "series_label": "CO2 273K", "uncertainty_px": None,
         "evidence_kind": "native_visible", "evidence_ref": "panel.png", "source_evidence": "marker shape identifies series"},
    ]}
    return image_path, extraction, spec, response


def test_python_applies_edits_then_rebuilds_rows_counts_and_visual_diagnostics(tmp_path):
    image_path, extraction, spec, response = _test_stage(tmp_path)
    final, audit = stage_edits.apply_agent_edits(extraction, spec, response, "agent04_final_check")
    assert audit["applied"] == audit["requested"] == 4

    answer = {"series": [
        {"label": "CO2 273K", "coverage": "complete", "final_count": 2, "comment": ""},
        {"label": "CO2 298K", "coverage": "complete", "final_count": 1, "comment": ""},
    ]}
    assert reconcile_series_counts(answer, final, audit) == 0
    assert {item["label"]: item["final_count"] for item in answer["series"]} == {"CO2 273K": 3, "CO2 298K": 0}

    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    shutil.copyfile(image_path, stage_dir / "panel.png")
    stage, rows, metadata = stage_artifacts.write_stage(
        stage_dir, "agent04", final, spec, "fig1", "a", agent_report={"verdict": "accept", **answer},
        edit_audit=audit, status="complete_against_reviewed_evidence",
    )
    assert len(rows) == metadata["row_count"] == 3
    assert {row["series_name"] for row in rows} == {"CO2 273K"}
    assert len(stage["series"][0]["points"]) == 3
    added = next(point for point in stage["series"][0]["points"]
                 if point.get("source_evidence") == "native marker at (40, 60)")
    added_row = next(row for row in rows if row["point_id"] == added["point_id"])
    added_instance = next(item for item in metadata["instances"] if item["point_id"] == added["point_id"])
    assert added["confidence"] < 0.85
    assert added["uncertainty_px"] == 1.0
    assert added_row["uncertainty_x"] == 1.0 and added_row["uncertainty_y"] == 1.0
    assert "evidence_ref=panel.png" in added_row["notes"]
    assert added_instance["evidence"] == "native marker at (40, 60)"
    assert {point["point_id"] for point in stage["series"][0]["points"]} >= {"a-1", "b-1"}
    assert (stage_dir / "agent04_points.csv").is_file()
    assert json.loads((stage_dir / "agent04_points.json").read_text(encoding="utf-8"))["series"][0]["n_points"] == 3

    draw_stage(stage_dir, "agent04", stage, spec)
    hard = hard_checks.run(stage, spec, stage_dir, "agent04")
    assert hard["passed"], hard["problems"]
    assert (stage_dir / "agent04_overlay.png").is_file()
    assert (stage_dir / "agent04_recreated.png").is_file()
    assert (stage_dir / "agent04_compare.png").is_file()
    assert (stage_dir / "agent04_redraw_diff.png").is_file()


def test_logarithmic_axis_uncertainty_is_converted_in_data_units():
    calibration = {
        "axis_models": {
            "x": {"scale": "log", "slope": 0.01, "intercept": -1},
            "y": {"scale": "linear", "slope": -0.1, "intercept": 5},
        }
    }
    points = [{"px": [50, 20], "x": 0.316227766, "y": 3, "source": "agent04", "evidence_kind": "native_visible",
               "uncertainty_px": 2, "evidence_ref": "tile:log-1", "source_evidence": "native glyph at (50, 20)"}]
    extraction = {"calibration": calibration, "series": [{"label": "CO2 273K", "gas": "CO2", "extract": True,
                  "points": points}]}
    spec = {"panel_bbox": [0, 0, 100, 100], "series": [{"label": "CO2 273K", "gas": "CO2", "extract": True}],
            "llm_spec": {"series": [{"label": "CO2 273K", "gas": "CO2"}]}}
    _, csv_rows, metadata = stage_artifacts.build_candidate(extraction, spec, "fig1", "a")
    assert csv_rows[0]["uncertainty_x"] > 0
    assert csv_rows[0]["uncertainty_y"] == 0.2
    assert metadata["instances"][0]["evidence_ref"] == "tile:log-1"


def test_final_check_orchestrates_full_candidate_edits_and_python_rebuild(tmp_path, monkeypatch):
    panel_dir = tmp_path / "fig1a"
    panel_dir.mkdir()
    image_path, extraction, spec, _ = _test_stage(panel_dir)
    # The source contains a marker that Python missed at (40, 50), while its
    # candidate at (70, 20) has no source marker and should be deleted.
    source = cv2.imread(str(image_path))
    cv2.rectangle(source, (64, 14), (76, 26), (255, 255, 255), -1)
    cv2.circle(source, (40, 50), 4, (220, 40, 30), -1)
    assert cv2.imwrite(str(image_path), source)
    spec["series"][0]["curve_points"] = [{"x": 20, "y": 20}, {"x": 70, "y": 80}]
    spec["series"][1]["curve_points"] = [{"x": 0, "y": 30}, {"x": 100, "y": 30}]
    save_json(panel_dir / "spec.json", spec)

    stage_artifacts.write_stage(panel_dir, "python", extraction, spec, "fig1", "a")
    agent03, rows, _ = stage_artifacts.write_stage(panel_dir, "agent03", extraction, spec, "fig1", "a")
    draw_stage(panel_dir, "agent03", agent03, spec)
    save_json(panel_dir / "agent03_hard_checks.json", hard_checks.run(agent03, spec, panel_dir, "agent03"))
    save_json(panel_dir / "agent03_review_points.json", {
        "verdict": "accept", "reasoning": "Initial review complete.", "operations": [], "uncertainties": [],
    })
    assert suspects.find(rows, spec) == []

    answer = copy.deepcopy(load_json("examples/agent04_final_check.example.json"))
    answer["independent_read"]["x_axis"] = "Pressure (kPa), linear 0-100"
    answer["independent_read"]["y_axis"] = "CO2 uptake (mmol/g), linear 0-100"
    answer["independent_read"]["series"] = [
        {"label": "CO2 273K", "marker": "circle", "marker_fill": "filled", "visible_markers_estimate": 2, "notes": ""},
        {"label": "CO2 298K", "marker": "square", "marker_fill": "filled", "visible_markers_estimate": 1, "notes": ""},
    ]
    answer["axis_check"] = {
        "x": {"printed_min": 0, "printed_max": 100, "data_min": 20, "data_max": 80,
              "result": "pass", "reason": "The source ticks and candidate positions align."},
        "y": {"printed_min": 0, "printed_max": 100, "data_min": 20, "data_max": 80,
              "result": "pass", "reason": "The source ticks and candidate positions align."},
    }
    answer["series"] = [
        {"label": "CO2 273K", "coverage": "complete", "final_count": 2, "comment": ""},
        {"label": "CO2 298K", "coverage": "complete", "final_count": 1, "comment": ""},
    ]
    replacement = copy.deepcopy(agent03)
    replacement["series"][0]["points"] = [replacement["series"][0]["points"][0], {
        "point_id": "qa-new", "source_instance_id": "qa-source-new", "px": [40, 50], "x": 40, "y": 50,
        "source": "agent04", "confidence": 0.7, "evidence_kind": "native_visible",
        "uncertainty_px": 1.0, "evidence_ref": "panel.png", "source_evidence": "Blue circle at native (40, 50).",
    }]
    _, final_rows, _ = stage_artifacts.build_candidate(replacement, spec, "fig1", "a")
    answer.update(calibration=agent03["calibration"], row_count=3, changes=[
        {"action": "add", "point_id": "qa-new", "reason": "Missing marker", "source_evidence": "Blue circle at native (40, 50)."},
        {"action": "delete", "point_id": "a-2", "reason": "False detection", "source_evidence": "Native (70, 20) is blank."},
    ])

    class FakeAgent:
        schema = load_schema("schemas/agent04_final_check.schema.json")
        example = load_json("examples/agent04_final_check.example.json")

        async def ask_artifacts(self, values, **kwargs):
            import zipfile
            self.values = values
            with zipfile.ZipFile(kwargs["files"][0]) as archive:
                sent = list(csv.DictReader(archive.read("agent03_points.csv").decode().splitlines()))
                assert {row["point_id"] for row in sent} == {"a-1", "a-2", "b-1"}
                assert "panel.png" in archive.namelist()
            errors = list(Draft202012Validator(self.schema).iter_errors(answer))
            assert not errors, [error.message for error in errors]
            output = panel_dir / "fake-output"
            output.mkdir()
            report_path = output / "agent04_final_check.json"
            csv_path = output / "agent04_points.csv"
            save_json(report_path, answer)
            stage_artifacts._write_csv(csv_path, final_rows)
            return {"report": copy.deepcopy(answer), "provenance": {"response_mode": "artifacts"},
                    "artifacts": {"agent04_final_check.json": report_path, "agent04_points.csv": csv_path}}

    fake_agent = FakeAgent()
    step = FinalCheck(load_workflow().steps["agent04_final_check"])
    step.agent = fake_agent
    monkeypatch.setattr("src.workflow.steps.agent04_final_check.review_page.build", lambda *args, **kwargs: None)
    panel = Panel(folder="fig1a", figure={}, discovery={}, panel={}, figure_id="fig1", panel_id="a")
    run = DocumentRun(source="", document_dir=str(tmp_path), run_id="test", panels=(panel,))

    asyncio.run(step.run_panel(run, panel))

    saved_answer = read_json(panel_dir / "agent04_final_check.json")
    assert {item["label"]: item["final_count"] for item in saved_answer["series"]} == {"CO2 273K": 2, "CO2 298K": 1}
    with (panel_dir / "agent04_points.csv").open(encoding="utf-8", newline="") as handle:
        saved_rows = list(csv.DictReader(handle))
    assert len(saved_rows) == 3
    assert {row["point_id"] for row in saved_rows} == {"a-1", "b-1", "qa-new"}
    assert saved_rows[-1]["notes"] == str(final_rows[-1]["notes"])
    final_json = read_json(panel_dir / "agent04_points.json")
    counts = {series["label"]: series["n_points"] for series in final_json["series"]}
    assert counts == {"CO2 273K": 2, "CO2 298K": 1}
    assert read_json(panel_dir / "agent04_edit_audit.json")["applied"] == 2
    metadata = read_json(panel_dir / "agent04_metadata.json")
    assert {item["label"]: item["final_count"] for item in metadata["agent_report"]["series"]} == counts
    qa = read_json(panel_dir / "qa.json")
    assert qa["rows"]["agent04"] == 3 and qa["final_operations_applied"] == 2
    assert read_json(panel_dir / "agent04_hard_checks.json")["passed"]


def test_unapplied_operations_are_counted_as_requiring_review():
    answer = {"series": [{"label": "CO2 273K", "coverage": "complete", "final_count": 1, "comment": ""}]}
    extraction = {"series": [{"label": "CO2 273K", "points": [{"point_id": "pt-1"}]}]}
    unapplied = reconcile_series_counts(answer, extraction, {"requested": 1, "applied": 0})
    assert unapplied == 1
    assert answer["series"][0]["coverage"] == "uncertain"
    assert "not applied" in answer["series"][0]["comment"]

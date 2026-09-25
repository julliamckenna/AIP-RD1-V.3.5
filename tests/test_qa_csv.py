import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.tools import qa_csv, stage_artifacts


class _WorkspaceTemp:
    """Use direct workspace files; child temp directories inherit a restricted ACL here."""
    def __enter__(self):
        self.root = Path(__file__).resolve().parents[1]
        return self.root

    def __exit__(self, *_):
        for name in ("qa_csv_test_final.csv", "qa_csv_test_panel.png"):
            (self.root / name).unlink(missing_ok=True)


def _case(root):
    image = root / "qa_csv_test_panel.png"
    cv2.imwrite(str(image), np.full((100, 100, 3), 255, dtype=np.uint8))
    calibration = {
        "frame_px": [10, 10, 90, 90],
        "axis_models": {
            "x": {"slope": 0.1, "intercept": -1.0, "scale": "linear"},
            "y": {"slope": -0.1, "intercept": 9.0, "scale": "linear"},
        },
        "unit_per_px_x": 0.1,
        "unit_per_px_y": 0.1,
    }
    spec = {
        "image": str(image), "panel_bbox": [0, 0, 100, 100],
        "x": {"unit": "bar", "scale": "linear", "ticks": [0, 1, 2]},
        "y": {"unit": "mmol/g", "scale": "linear", "ticks": [0, 1, 2]},
        "series": [
            {"label": "293K", "extract": True, "gas": "CO2"},
            {"label": "313K", "extract": True, "gas": "CO2"},
        ],
        "llm_spec": {"series": [
            {"label": "293K", "gas": "CO2", "marker": "circle", "line_style": "solid",
             "color_hex": "#ff0000", "species_evidence": "CO2 in legend", "quantity_type": "loading",
             "loading_basis": "unknown"},
            {"label": "313K", "gas": "CO2", "marker": "triangle_up", "line_style": "dashed",
             "color_hex": "#0000ff", "species_evidence": "CO2 in legend", "quantity_type": "loading",
             "loading_basis": "unknown"},
        ]},
    }
    candidate = {
        "calibration": calibration,
        "series": [
            {"label": "293K", "extract": True, "gas": "CO2", "marker": "circle", "points": [
                {"point_id": "p1", "source_instance_id": "s1", "x": 1.0, "y": 1.0,
                 "px": [20.0, 80.0], "source": "manual", "confidence": 0.9},
            ]},
            {"label": "313K", "extract": True, "gas": "CO2", "marker": "triangle_up", "points": [
                {"point_id": "p2", "source_instance_id": "s2", "x": 2.0, "y": 2.0,
                 "px": [30.0, 70.0], "source": "manual", "confidence": 0.9},
            ]},
        ],
        "unresolved_slots": {"total": 0, "by_series": {}, "slots": []},
    }
    report = {
        "verdict": "accept", "series": [], "calibration": calibration,
        "unresolved_slots": {"total": 0, "by_series": {}, "slots": []},
        "row_count": 2, "changes": [],
    }
    _, rows, _ = stage_artifacts.build_candidate(candidate, spec, "fig-1", "panel-a")
    return spec, candidate, report, rows


def _write(path, rows, *, header=None):
    columns = stage_artifacts.STAGE_COLUMNS if header is None else header
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _with_native_support(row, *, px, py, point_id, source_id, notes=None):
    row = dict(row)
    row.update({
        "point_id": point_id, "source_instance_id": source_id,
        "source_pixel_x": str(px), "source_pixel_y": str(py),
        "x": str(0.1 * px - 1), "y": str(-0.1 * py + 9),
        "notes": notes or "evidence_kind=native_visible; uncertainty_px=1.5; evidence_ref=panel.png; source_evidence=Visible native marker; source=agent04",
    })
    return row


def _load(root, rows, *, spec=None, candidate=None, report=None, header=None):
    path = root / "qa_csv_test_final.csv"
    _write(path, rows, header=header)
    if spec is None or candidate is None or report is None:
        base_spec, base_candidate, base_report, _ = _case(root)
        spec = spec or base_spec
        candidate = candidate or base_candidate
        report = report or base_report
    report = dict(report)
    report["row_count"] = len(rows)
    labels = [row["series_name"] for row in rows]
    series_labels = [series["label"] for series in candidate["series"]
                     if series.get("extract") and str(series.get("gas", "")).upper() == "CO2"]
    report["series"] = [
        {"label": label, "coverage": "uncertain", "final_count": labels.count(label), "comment": "Reviewed."}
        for label in series_labels
    ]
    return qa_csv.load_final_csv(path, report, candidate, spec, "fig-1", "panel-a")


def test_roundtrip_preserves_authored_rows_and_notes():
    with _WorkspaceTemp() as root:
        spec, candidate, report, rows = _case(root)
        rows[0]["notes"] = "human checked marker; evidence_ref=panel.png; source_evidence=ring visible"
        rows[0]["normalized_x"] = "100"
        rows[0]["normalized_x_unit"] = "kPa"
        rows[0]["conversion_id"] = "bar_to_kpa-v1"
        extraction, authored, audit = _load(root, rows, spec=spec, candidate=candidate, report=report)
        assert extraction["series"][0]["points"][0]["csv_row"] == authored[0]
        assert extraction["authored_rows"] == authored
        assert extraction["series"][0]["points"][0]["confidence"] == 0.9
        assert audit["authoritative_csv"] is True
        assert audit["deleted"] == []


def test_inferred_rows_are_marked_for_review_and_native_notes_are_restored():
    with _WorkspaceTemp() as root:
        spec, candidate, report, rows = _case(root)
        rows[0]["is_inferred"] = "true"
        rows[0]["evidence_type"] = "line_sample"
        rows[0]["marker_shape"] = "none"
        rows[0]["notes"] = "branch=adsorption; assigned_by=Agent03; source=curve"
        extraction, _, audit = _load(root, rows, spec=spec, candidate=candidate, report=report)
        point = extraction["series"][0]["points"][0]
        assert point["branch"] == "adsorption"
        assert point["assigned_by"] == "Agent03"
        assert point["is_inferred"] is True
        assert audit["contains_inferred"] and audit["requires_review"]


def test_accepts_python_bool_csv_tokens_and_normalizes_them():
    with _WorkspaceTemp() as root:
        spec, candidate, report, rows = _case(root)
        for row in rows:
            row["is_inferred"] = "False"
            row["export_eligible"] = "True"
        _, authored, _ = _load(root, rows, spec=spec, candidate=candidate, report=report)
        assert {row["is_inferred"] for row in authored} == {"false"}
        assert {row["export_eligible"] for row in authored} == {"true"}


def test_csv_is_full_replacement_and_audits_add_delete_move_and_reassignment():
    with _WorkspaceTemp() as root:
        spec, candidate, report, rows = _case(root)
        expected_b = next(row for row in rows if row["series_name"] == "313K")
        reassigned = _with_native_support(
            expected_b, px=21, py=79, point_id="p1", source_id="s1",
        )
        added = _with_native_support(
            dict(rows[0]), px=40, py=60, point_id="p3", source_id="s3",
        )
        added["series_id"] = rows[0]["series_id"]
        added["series_name"] = "293K"
        added["marker_shape"] = "circle"
        added["line_style"] = rows[0]["line_style"]
        added["color_hex"] = rows[0]["color_hex"]
        added["species"] = "CO2"
        added["series_id"] = rows[0]["series_id"]
        report["changes"] = [
            {"action": "move", "point_id": "p1", "reason": "Marker center corrected.", "source_evidence": "Native panel marker."},
            {"action": "reassign", "point_id": "p1", "reason": "Marker matches the 313K series.", "source_evidence": "Native marker shape and legend."},
            {"action": "add", "point_id": "p3", "reason": "Visible marker omitted from candidate.", "source_evidence": "Native panel marker."},
            {"action": "delete", "point_id": "p2", "reason": "Candidate point is unsupported.", "source_evidence": "No native marker at candidate location."},
        ]
        extraction, final_rows, audit = _load(
            root, [reassigned, added], spec=spec, candidate=candidate, report=report,
        )
        assert [row["point_id"] for row in final_rows] == ["p1", "p3"]
        assert audit["added"] == ["p3"]
        assert audit["deleted"] == ["p2"]
        assert audit["moved"][0]["point_id"] == "p1"
        assert audit["reassigned"] == [{"point_id": "p1", "from": "293K", "to": "313K"}]
        assert audit["requested"] == audit["applied"] == 4
        assert len(audit["operations"]) == 4
        assert extraction["series"][0]["n_points"] == 1
        assert extraction["series"][1]["n_points"] == 1


@pytest.mark.parametrize("change, message", [
    (lambda rows: rows[0].update(x="nan"), "finite number"),
    (lambda rows: rows[0].update(species="N2"), "CO2"),
    (lambda rows: rows[0].update(source_pixel_x="91"), "outside the plot frame"),
    (lambda rows: rows[0].update(x="1.25"), "disagrees with its native pixel"),
    (lambda rows: rows[0].update(x="1.0001"), "disagrees with its native pixel"),
    (lambda rows: rows[0].update(point_id=""), "nonempty point_id"),
])
def test_rejects_malformed_values_species_bounds_calibration_and_ids(change, message):
    with _WorkspaceTemp() as root:
        spec, candidate, report, rows = _case(root)
        changed = [dict(row) for row in rows[:1]]
        change(changed)
        with pytest.raises(qa_csv.FinalCSVError, match=message):
            _load(root, changed, spec=spec, candidate=candidate, report=report)


def test_requires_native_uncertainty_for_new_points_and_exact_duplicate_pixels_fail():
    with _WorkspaceTemp() as root:
        spec, candidate, report, rows = _case(root)
        new_row = _with_native_support(dict(rows[0]), px=40, py=60, point_id="p3", source_id="s3")
        new_row["notes"] = "evidence_kind=native_visible; evidence_ref=panel.png; source_evidence=marker visible"
        with pytest.raises(qa_csv.FinalCSVError, match="positive uncertainty_px"):
            _load(root, [new_row], spec=spec, candidate=candidate, report=report)

        one, two = dict(rows[0]), dict(rows[0])
        two.update(point_id="p3", source_instance_id="s3")
        two["notes"] = "evidence_kind=native_visible; uncertainty_px=1; evidence_ref=panel.png; source_evidence=marker visible"
        with pytest.raises(qa_csv.FinalCSVError, match="duplicate native pixel"):
            _load(root, [one, two], spec=spec, candidate=candidate, report=report)


def test_rejects_noncanonical_header_and_duplicate_candidate_ids():
    with _WorkspaceTemp() as root:
        spec, candidate, report, rows = _case(root)
        bad_header = list(stage_artifacts.STAGE_COLUMNS)
        bad_header[0], bad_header[1] = bad_header[1], bad_header[0]
        with pytest.raises(qa_csv.FinalCSVError, match="canonical points column order"):
            _load(root, rows, spec=spec, candidate=candidate, report=report, header=bad_header)
        duplicate = {**candidate, "series": [dict(s) for s in candidate["series"]]}
        duplicate["series"][1] = {**duplicate["series"][1], "points": [dict(candidate["series"][0]["points"][0])]}
        with pytest.raises(qa_csv.FinalCSVError, match="duplicate point_id"):
            _load(root, rows, spec=spec, candidate=duplicate, report=report)


def test_report_series_counts_and_calibration_scale_are_authoritative():
    with _WorkspaceTemp() as root:
        spec, candidate, report, rows = _case(root)
        path = root / "qa_csv_test_final.csv"
        _write(path, rows)
        report["row_count"] = len(rows)
        report["series"] = [
            {"label": "293K", "final_count": 1},
        ]
        with pytest.raises(qa_csv.FinalCSVError, match="every eligible candidate series"):
            qa_csv.load_final_csv(path, report, candidate, spec, "fig-1", "panel-a")

        _, _, report, rows = _case(root)
        _write(path, rows)
        report["row_count"] = len(rows)
        report["series"] = [
            {"label": "293K", "final_count": 2},
            {"label": "313K", "final_count": 0},
        ]
        with pytest.raises(qa_csv.FinalCSVError, match="final_count does not match"):
            qa_csv.load_final_csv(path, report, candidate, spec, "fig-1", "panel-a")

        _, _, report, rows = _case(root)
        _write(path, rows)
        report["row_count"] = len(rows)
        report["series"] = [
            {"label": "293K", "final_count": 1},
            {"label": "313K", "final_count": 1},
        ]
        report["calibration"] = {
            **report["calibration"],
            "axis_models": {
                **report["calibration"]["axis_models"],
                "x": {**report["calibration"]["axis_models"]["x"], "scale": "log"},
            },
        }
        with pytest.raises(qa_csv.FinalCSVError, match="scale conflicts"):
            qa_csv.load_final_csv(path, report, candidate, spec, "fig-1", "panel-a")

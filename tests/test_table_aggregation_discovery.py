import csv
import json
import shutil
import uuid

import pytest

from src.tools import table
from src.tools.stage_artifacts import STAGE_COLUMNS
from src.settings import ROOT


@pytest.fixture
def workspace_tmp_path():
    path = ROOT / "artifacts" / "test-tmp" / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def _write_points(path, point_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, point_id in enumerate(point_ids):
        row = {column: "" for column in STAGE_COLUMNS}
        row.update(
            point_id=point_id,
            source_instance_id=f"src-{point_id}",
            figure_id="f03a",
            panel_id="panel-a",
            series_id="co2-195k",
            series_name="CO2 195K",
            species="CO2",
            export_eligible="true",
            evidence_type="visible_marker",
            marker_shape="circle",
            x=str(index / 10),
            y=str(index),
            confidence="high",
            source_pixel_x=str(100 + index),
            source_pixel_y=str(200 - index),
        )
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STAGE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_aggregation_ignores_agent04_input_copy_and_keeps_other_panels(workspace_tmp_path):
    runs = workspace_tmp_path / "runs"
    first = runs / "doc-a" / "run-a" / "panel-a"
    second = runs / "doc-b" / "run-b" / "panel-b"
    _write_points(first / "agent03_points.csv", ["keep-1", "keep-2"])
    _write_points(second / "agent03_points.csv", ["keep-3"])
    # Agent 04's saved input bundle contains a copied pre-review table that
    # must not be discovered as a third panel or restore removed row IDs.
    _write_points(
        first / "agent04_unstructured" / "agent04_inputs" / "agent03_points.csv",
        ["deleted-1", "deleted-2"],
    )
    (first / "agent04_unstructured" / "attempt.json").write_text("{}", encoding="utf-8")

    master_path, rows = table.write_stage_aggregations(runs)
    report = json.loads((runs / "aggregation_report.json").read_text(encoding="utf-8"))
    master_rows = table._read_csv(master_path)

    assert len(rows) == 3
    assert {row["point_id"] for row in master_rows} == {"keep-1", "keep-2", "keep-3"}
    assert report["panel_count"] == 2
    assert report["review_row_count"] == 3
    assert report["accepted_row_count"] == 0

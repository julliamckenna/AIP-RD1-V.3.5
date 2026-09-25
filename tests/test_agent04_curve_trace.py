import csv
import json
import shutil
import uuid

import cv2
import numpy as np
import pytest

from src.settings import ROOT
from src.tools import review_page, stage_artifacts


@pytest.fixture
def workspace_tmp_path():
    path = ROOT / "artifacts" / "test-tmp" / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def _row(point_id, x, y):
    return {
        "point_id": point_id,
        "source_instance_id": f"src-{point_id}",
        "figure_id": "f03a",
        "panel_id": "panel-a",
        "series_id": "co2-195k",
        "series_name": "CO2 195K",
        "species": "CO2",
        "export_eligible": "true",
        "evidence_type": "visible_marker",
        "marker_shape": "circle",
        "line_style": "solid",
        "color_hex": "#5462fb",
        "x": str(x / 100),
        "y": str(60 - y),
        "confidence": "high",
        "source_pixel_x": str(x),
        "source_pixel_y": str(y),
    }


def test_agent04_demotes_only_color_supported_deletes_to_separate_line_samples(workspace_tmp_path):
    panel = workspace_tmp_path / "panel.png"
    image = np.full((60, 60, 3), 255, dtype=np.uint8)
    cv2.line(image, (10, 10), (10, 32), (251, 98, 84), 4)  # blue CO2 ink
    cv2.line(image, (30, 42), (56, 42), (50, 70, 220), 4)  # red N2 ink
    cv2.imwrite(str(panel), image)
    candidate = [_row("keep", 10, 12), _row("drop-1", 10, 18), _row("drop-2", 10, 24), _row("drop-3", 10, 30)]
    final = [candidate[0]]
    extraction = {"series": [{"label": "CO2 195K", "marker_radius_px": 3}]}
    spec = {
        "mask_bboxes": [[40, 0, 58, 20]],
        "region_strategies": [{"strategy": "ignore", "series_labels": [], "bbox_px": [40, 0, 58, 20], "reason": "inset"}],
    }
    calibration = {
        "frame_px": [0, 0, 60, 60],
        "axis_models": {
            "x": {"scale": "linear", "slope": 0.01, "intercept": 0},
            "y": {"scale": "linear", "slope": -1, "intercept": 60},
        },
    }

    traces, audit = stage_artifacts.demote_deleted_markers_to_curve_traces(
        candidate, final, extraction, spec, panel, calibration,
    )

    assert len(final) == 1
    assert audit["demoted_candidate_count"] == 3
    assert traces[0]["evidence_type"] == "line_sample"
    assert traces[0]["marker_shape"] == "none"
    assert traces[0]["orientation"] == "vertical"
    assert all(point["source"] == "curve" for point in traces[0]["points"])
    assert all(7 <= point["px"][0] <= 13 for point in traces[0]["points"])
    assert min(point["px"][1] for point in traces[0]["points"]) <= 12
    assert traces[0]["provenance"]["excluded_regions"]
    assert traces[0]["provenance"]["source_candidate_ids"] == ["drop-1", "drop-2", "drop-3"]


def test_agent04_review_renders_curve_trace_without_adding_marker_rows(workspace_tmp_path):
    panel = workspace_tmp_path / "panel.png"
    cv2.imwrite(str(panel), np.full((60, 60, 3), 255, dtype=np.uint8))
    result_path = workspace_tmp_path / "agent04_points.json"
    result = {
        "calibration": {
            "frame_px": [0, 0, 60, 60],
            "unit_per_px_x": 0.01,
            "unit_per_px_y": 1,
            "tick_fit_resid_x": 0,
            "tick_fit_resid_y": 0,
            "x_ticks_px": [],
            "y_ticks_px": [],
            "axis_models": {
                "x": {"scale": "linear", "slope": 0.01, "intercept": 0},
                "y": {"scale": "linear", "slope": -1, "intercept": 60},
            },
        },
        "series": [{
            "label": "CO2 195K",
            "marker_shape": "circle",
            "color_lab": None,
            "points": [{"x": 0.1, "y": 48, "px": [10, 12], "confidence": 0.9}],
        }],
        "curve_traces": [{
            "series_name": "CO2 195K",
            "color_hex": "#5462fb",
            "source": "curve",
            "evidence_type": "line_sample",
            "marker_shape": "none",
            "points": [
                {"x": 0.1, "y": 42, "px": [10, 18], "source": "curve"},
                {"x": 0.1, "y": 36, "px": [10, 24], "source": "curve"},
                {"x": 0.1, "y": 30, "px": [10, 30], "source": "curve"},
            ],
        }],
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")

    html_path = workspace_tmp_path / "review.html"
    review_page.build(str(result_path), str(panel), str(html_path))
    html = html_path.read_text(encoding="utf-8")

    assert "line samples, not marker centers" in html
    rebuilt = json.loads(result_path.read_text(encoding="utf-8"))
    assert len(rebuilt["series"][0]["points"]) == 1
    assert len(rebuilt["curve_traces"][0]["points"]) == 3

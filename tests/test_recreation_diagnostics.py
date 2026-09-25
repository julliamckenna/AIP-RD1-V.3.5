"""Physics diagnostics must not change the evidence used by reconstruction."""

import copy
import json
import runpy
import sys

import pytest

from src.tools import checks, hard_checks, recreate


@pytest.fixture
def extraction():
    # A steep rise followed by a plateau makes residual clustering infer two
    # branches even though the supplied markers carry no branch assignment.
    xy = [
        (0.0023, 19.8419), (0.0044, 3.035), (0.0064, 14.9399),
        (0.0147, 59.7583), (0.0353, 66.411), (0.0374, 95.1228),
        (0.0415, 86.0191), (0.0456, 103.5262), (0.0622, 117.532),
        (0.091, 121.7337), (0.1757, 122.0839), (0.2974, 122.434),
        (0.543, 123.8346), (0.7442, 124.7169), (0.9938, 126.8213),
    ]
    return {
        "calibration": {
            "frame_px": [100, 20, 600, 420],
            "unit_per_px_y": 0.35,
            "axis_models": {
                "x": {"scale": "linear", "slope": 0.002, "intercept": -0.2},
                "y": {"scale": "linear", "slope": -0.35, "intercept": 147},
            },
        },
        "series": [{
            "label": "CO2 195K",
            "points": [{"x": x, "y": y, "confidence": 0.95} for x, y in xy],
        }],
    }


@pytest.mark.parametrize("existing_branch", [None, "ads"])
def test_checks_leave_point_evidence_unchanged(extraction, existing_branch):
    if existing_branch:
        extraction["series"][0]["points"][0]["branch"] = existing_branch
    original = copy.deepcopy(extraction)
    diagnostic = copy.deepcopy(extraction["series"][0])
    assert checks.split_branches(diagnostic, 0.35) > 0
    assert {p["branch"] for p in diagnostic["points"]} == {"ads", "des"}

    checks.check_series(extraction["series"][0], 0.35)
    assert extraction == original
    report = checks.run(extraction)
    assert report["series"]["CO2 195K"]["flags"]
    assert extraction == original


def test_hard_checks_do_not_change_rendered_connections(extraction, monkeypatch, tmp_path):
    spec = {
        "x": {"title": "Relative Pressure (P/P₀)", "unit": "P/P₀", "ticks": [0, 0.5, 1]},
        "y": {"title": "Uptake", "unit": "cm³ g⁻¹", "ticks": [0, 70, 140]},
        "llm_spec": {"series": [{"label": "CO2 195K", "marker_fill": "open"}]},
    }
    # The native ink comparison is independent of diagnostic branch inference.
    monkeypatch.setattr(hard_checks, "redraw", lambda *args: {})
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    recreate.recreate(extraction, spec, before)
    report = hard_checks.run(extraction, spec)
    assert report["passed"]
    recreate.recreate(extraction, spec, after)
    assert after.read_bytes() == before.read_bytes()


def test_checks_cli_does_not_rewrite_input(extraction, monkeypatch, tmp_path, capsys):
    path = tmp_path / "points.json"
    original = json.dumps(extraction, ensure_ascii=False).encode("utf-8")
    path.write_bytes(original)
    monkeypatch.setattr(sys, "argv", [checks.__file__, str(path)])
    runpy.run_path(checks.__file__, run_name="__main__")
    assert json.loads(capsys.readouterr().out)["verdict"] == "review"
    assert path.read_bytes() == original

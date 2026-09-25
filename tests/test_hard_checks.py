"""Hard checks and agent 02 coaching - deterministic, no Azure access needed."""

import cv2
import numpy as np

from src.tools import hard_checks, region_strategies, series_gate
from src.workflow.steps.agent02_check_extraction import apply_axes_check


def _extraction(x_slope, x_intercept, xs, frame=(100, 20, 620, 420)):
    """Frame 100..620 px; the true axis puts 0 at 100 px and 120 at 620 px (4.333 px per kPa)."""
    return {
        "calibration": {
            "axis_models": {"x": {"scale": "linear", "slope": x_slope, "intercept": x_intercept},
                            "y": {"scale": "linear", "slope": -0.0075, "intercept": 3.15}},
            "frame_px": list(frame),
        },
        "series": [{"label": "CO2 303K", "points": [{"x": x, "y": 1.0} for x in xs]}],
    }


SPEC = {"x": {"ticks": [0, 40, 80, 120], "unit": "kPa", "scale": "linear"},
        "y": {"ticks": [0, 1, 2, 3], "unit": "mmol/g", "scale": "linear"}}


def test_correct_calibration_passes():
    report = hard_checks.axis_range(_extraction(120 / 520, -100 * 120 / 520, [0.5, 60, 118]), SPEC)
    assert report["passed"], report["problems"]


def test_shifted_tick_mapping_fails_even_with_values_in_range():
    # fig2b: minor ticks taken as the printed ones - the frame reads -40..199.5 kPa
    slope = 40 / 87
    report = hard_checks.axis_range(_extraction(slope, -40 - 100 * slope, [10, 60, 110]), SPEC)
    assert not report["passed"]
    assert any("frame" in problem for problem in report["problems"])


def test_values_outside_the_printed_axis_fail():
    report = hard_checks.axis_range(_extraction(120 / 520, -100 * 120 / 520, [-37.0, 60, 188]), SPEC)
    assert not report["passed"]
    assert report["x"]["points_outside"] == 2


def test_data_inside_a_frame_that_runs_past_the_last_tick_passes():
    # fig10: y printed 0-50, the frame runs to ~55 and the top markers sit at 53.6
    spec = {"x": SPEC["x"], "y": {"ticks": [0, 10, 20, 30, 40, 50], "unit": "cm3/g", "scale": "linear"}}
    extraction = _extraction(120 / 520, -100 * 120 / 520, [10, 60])
    extraction["calibration"]["axis_models"]["y"] = {"scale": "linear", "slope": -55 / 400, "intercept": 55 + 20 * 55 / 400}
    extraction["series"][0]["points"] = [{"x": 10, "y": 53.6}, {"x": 60, "y": 20}]
    assert hard_checks.axis_range(extraction, spec)["passed"]


def test_curve_points_hold_over_their_whole_range():
    curve, x_from, x_to = series_gate.anchor_curve(
        {"curve_points": [{"x": 0, "y": 0}, {"x": 5, "y": 2}, {"x": 100, "y": 4}]}, 0, 100)
    assert (x_from, x_to) == (0, 100)
    assert abs(curve(2.5) - 1.0) < 1e-9
    _, x_from, _ = series_gate.anchor_curve({"y_anchors": {"p25": 1, "p50": 2, "p75": 3, "p100": 4}}, 0, 100)
    assert x_from == 25  # coarse anchors stay unreliable in the steep first quarter


def _series(idx, label, fill, points, color):
    return {"idx": idx, "spec": {"label": label, "marker_fill": fill}, "col": np.float32(color), "r": 5.0,
            "tol": 25.0, "pts": [(float(x), float(y), 0.9, False, True) for x, y in points]}


def test_split_filled_open_by_centre_ink():
    image = np.full((80, 120, 3), 255, np.uint8)
    cv2.circle(image, (30, 40), 5, (40, 40, 220), -1)     # filled red
    cv2.circle(image, (80, 40), 5, (40, 40, 220), 1)      # open red ring
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    red = lab[40, 30]
    ads = _series(0, "a ads", "filled", [(80, 40)], red)  # both points start on the wrong series
    des = _series(1, "b des", "open", [(30, 40)], red)
    spec = {"region_strategies": [{"strategy": "split_filled_open", "series_labels": ["a ads", "b des"],
                                   "bbox_px": [0, 0, 120, 80]}]}
    _, audit = region_strategies.apply([ads, des], spec, lab, {"slope": 1, "intercept": 0}, 5.0)
    assert [p[:2] for p in ads["pts"]] == [(30.0, 40.0)]
    assert [p[:2] for p in des["pts"]] == [(80.0, 40.0)]
    assert audit[0]["reassigned"] == 2


def test_ignore_region_drops_listed_series_only():
    lab = np.zeros((50, 50, 3), np.float32)
    co2 = _series(0, "CO2", "filled", [(10, 10), (40, 40)], [50, 0, 0])
    n2 = _series(1, "N2", "filled", [(12, 12)], [50, 0, 0])
    spec = {"region_strategies": [{"strategy": "ignore", "series_labels": ["CO2"], "bbox_px": [0, 0, 20, 20]}]}
    region_strategies.apply([co2, n2], spec, lab, {"slope": 1, "intercept": 0}, 5.0)
    assert [p[:2] for p in co2["pts"]] == [(40.0, 40.0)]
    assert len(n2["pts"]) == 1


def test_agent02_coaching_reaches_the_spec():
    spec = {"x": {}, "y": {}, "series": [{"label": "CO2 273K"}], "llm_spec": {"series": [{"label": "CO2 273K"}]},
            "panel_bbox": [0, 0, 1000, 500]}
    check = {
        "x_axis": {}, "y_axis": {}, "series": [], "legend_bbox_norm": None, "complex_regions": [],
        "extraction_settings": [{"label": "CO2 273K", "color_tolerance": None, "separate_from": [], "y_bands": [],
                                 "template_fill": "on", "reason": "", "sample_marker_norm": [0.5, 0.2],
                                 "expected_counts": [{"x_from": 0, "x_to": 100, "count": 12}],
                                 "curve_points": [{"x": 50, "y": 2}, {"x": 0, "y": 0}]}],
        "region_strategies": [{"bbox_norm": [0.1, 0.1, 0.3, 0.5], "series_labels": [], "strategy": "ignore", "reason": ""}],
        "shared_x_columns": [20, 10],
    }
    out = apply_axes_check(spec, check, 1000, 500)
    series = out["series"][0]
    assert series["sample_px"] == [500.0, 100.0]
    assert [p["x"] for p in series["curve_points"]] == [0.0, 50.0]
    assert out["region_strategies"][0]["bbox_px"] == [100, 50, 300, 250]
    assert out["shared_x_columns"] == [10.0, 20.0]


def test_agent02_retry_limit_records_unexecuted_settings():
    from src.workflow.steps.agent02_check_extraction import application_audit
    audit = application_audit(3, 3, {"extraction_settings": [{}], "region_strategies": [{}],
                                     "shared_x_columns": [0.1], "complex_regions": [{}]},
                              state="unexecuted_retry_limit", reason="bounded limit")
    assert audit["checked"] and audit["proposed"]["extraction_settings"] == 1
    assert not audit["applied"] and audit["unexecuted"]


def test_isotherm_curve_starts_at_the_origin_on_linear_axes():
    from src.workflow.steps.agent02_check_extraction import with_origin
    linear = {"x": {"scale": "linear", "ticks": [0, 40]}, "y": {"scale": "linear", "ticks": [0, 3]}}
    assert with_origin(linear, [{"x": 2.0, "y": 0.8}])[0] == {"x": 0.0, "y": 0.0}
    log_x = {"x": {"scale": "log", "ticks": [0.01, 1]}, "y": {"scale": "linear", "ticks": [0, 3]}}
    assert with_origin(log_x, [{"x": 0.02, "y": 0.8}]) == [{"x": 0.02, "y": 0.8}]
    offset_y = {"x": {"scale": "linear", "ticks": [0, 40]}, "y": {"scale": "linear", "ticks": [1, 3]}}
    assert len(with_origin(offset_y, [{"x": 2.0, "y": 1.2}])) == 1


def test_sample_color_reads_the_face_not_the_dark_outline():
    image = np.full((40, 40, 3), 255, np.uint8)
    cv2.circle(image, (20, 20), 7, (40, 40, 214), -1)   # red face
    cv2.circle(image, (20, 20), 7, (0, 0, 0), 2)        # black outline
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    face = lab[20, 20]
    measured = region_strategies.sample_color(lab, (20, 20), np.float32([255, 128, 128]))
    assert np.linalg.norm(measured - face) < 8


def test_split_uses_the_glyph_size_not_the_detector_radius():
    image = np.full((80, 120, 3), 255, np.uint8)
    cv2.circle(image, (30, 40), 6, (40, 40, 220), 2)      # open ring, radius 6
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    red = lab[34, 30]
    ads = _series(0, "a ads", "filled", [(30, 40)], red)
    des = _series(1, "b des", "open", [], red)
    ads["r"] = 14.0  # a detector radius that drifted on a fused blob would see the ring as a filled centre
    spec = {"region_strategies": [{"strategy": "split_filled_open", "series_labels": ["a ads", "b des"],
                                   "bbox_px": [0, 0, 120, 80]}]}
    region_strategies.apply([ads, des], spec, lab, {"slope": 1, "intercept": 0}, 5.0,
                            glyphs=[{"size_px": 12}, {"size_px": 12}])
    assert [p[:2] for p in des["pts"]] == [(30.0, 40.0)] and not ads["pts"]


def _row(pid, series, x, y, evidence="visible_marker"):
    return {"point_id": pid, "series_name": series, "x": str(x), "y": str(y), "source_pixel_x": "0",
            "source_pixel_y": "0", "evidence_type": evidence}


def test_suspects_flag_off_curve_points_with_the_better_series():
    from src.tools import suspects
    rows = [_row(f"a{i}", "A", x, 4.0) for i, x in enumerate(range(0, 100, 10))]
    rows += [_row(f"b{i}", "B", x, 2.0) for i, x in enumerate(range(0, 100, 10))]
    rows.append(_row("stray", "A", 55, 2.05))            # an A row sitting on B's curve
    rows.append(_row("tf", "B", 35, 2.0, "template_fill"))
    spec = {"y": {"ticks": [0, 5]}, "series": []}
    found = {row["point_id"]: row["why"] for row in suspects.find(rows, spec)}
    assert "fits B better" in found["stray"]
    assert "inferred" in found["tf"]
    assert "a3" not in found


def test_agent04_tick_reading_is_cross_checked():
    from src.workflow.steps.agent04_final_check import printed_range_mismatch
    spec = {"x": {"ticks": [0, 40, 80, 120]}, "y": {"ticks": [0, 1, 2, 3]}}
    same = {"axis_check": {"x": {"printed_min": 0, "printed_max": 120}, "y": {"printed_min": 0, "printed_max": 3}}}
    other = {"axis_check": {"x": {"printed_min": 0, "printed_max": 100}, "y": {"printed_min": 0, "printed_max": 3}}}
    assert printed_range_mismatch(spec, same) == ""
    assert "x:" in printed_range_mismatch(spec, other)

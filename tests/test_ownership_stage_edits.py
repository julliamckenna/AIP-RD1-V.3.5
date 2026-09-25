import json

from src.tools.extract import (
    global_spatial_ownership,
    prune_expected_count_excess,
    spatial_ownership_guard,
)
from src.tools.stage_edits import apply_agent_edits


def _native_series(index, label, fill, points, radius=6):
    return {
        "idx": index,
        "r": radius,
        "spec": {"label": label},
        "pts": list(points),
    }


def test_crowded_filled_adsorption_series_deduplicate_one_native_marker():
    spec = {
        "llm_spec": {"branches": "adsorption_only"},
        "series": [
            {"label": "CO2 273K", "marker_fill": "filled"},
            {"label": "CO2 283K", "marker_fill": "filled"},
        ],
    }
    guard = spatial_ownership_guard(spec, ambiguity_groups=[{"group_id": 0}])
    assert guard["enabled"]
    series = [
        _native_series(0, "CO2 273K", "filled", [(20, 30, 0.92, False, True)]),
        _native_series(1, "CO2 283K", "filled", [(20.4, 30.1, 0.71, False, True)]),
    ]
    series, audit = global_spatial_ownership(series, spec=spec, enabled=True)
    assert audit["deduplicated"] == 1
    assert audit["unresolved"] == 0
    assert [len(item["pts"]) for item in series] == [1, 0]


def test_single_series_open_rings_are_not_spatially_deduplicated():
    spec = {
        "llm_spec": {"branches": "adsorption_only"},
        "series": [{"label": "CO2 195K", "marker_fill": "open"}],
    }
    series = [_native_series(0, "CO2 195K", "open", [(20, 30, 0.8, False, True)])]
    series, audit = global_spatial_ownership(series, spec=spec, enabled=None)
    assert not audit["enabled"]
    assert audit["overlap_components"] == 0
    assert series[0]["pts"] == [(20, 30, 0.8, False, True)]


def test_expected_count_soft_cap_prunes_weak_excess_with_audit():
    spec = {
        "x": {"ticks": [0, 100]},
        "y": {"ticks": [0, 100]},
        "series": [{
            "label": "CO2 273K",
            "expected_counts": [{"x_from": 0, "x_to": 100, "count": 2}],
            "curve_points": [{"x": 0, "y": 0}, {"x": 100, "y": 100}],
        }],
    }
    series = [_native_series(0, "CO2 273K", "filled", [
        (10, 10, 0.95, False, True),
        (20, 20, 0.95, False, True),
        (20.5, 31, 0.4, True, False),
        (50, 50, 0.95, False, True),
        (75, 75, 0.95, False, True),
        (90, 90, 0.35, True, False),
    ])]
    series, audit = prune_expected_count_excess(series, spec, lambda value: value, lambda value: value, enabled=True)
    # expected=2 with a minimum-two tolerance gives a soft cap of four.
    assert len(series[0]["pts"]) == 4
    assert audit["removed"] == 2
    assert all(item["reason"] == "expected_count_soft_cap_excess" for item in audit["series"][0]["removed"])
    assert any(item["px"] == [20.5, 31] for item in audit["series"][0]["removed"])


def test_expected_count_pruning_is_disabled_for_open_single_series():
    spec = {
        "x": {"ticks": [0, 100]},
        "y": {"ticks": [0, 100]},
        "series": [{"label": "CO2 195K", "expected_counts": [{"x_from": 0, "x_to": 100, "count": 1}]}],
    }
    series = [_native_series(0, "CO2 195K", "open", [(10, 10, 0.8, False, True)] * 4)]
    series, audit = prune_expected_count_excess(series, spec, lambda value: value, lambda value: value, enabled=False)
    assert len(series[0]["pts"]) == 4
    assert audit["removed"] == 0


def _staged_extraction():
    return {
        "calibration": {
            "frame_px": [0, 0, 100, 100],
            "axis_models": {
                "x": {"scale": "linear", "slope": 1, "intercept": 0},
                "y": {"scale": "linear", "slope": 1, "intercept": 0},
            },
        },
        "series": [
            {
                "label": "CO2 273K",
                "extract": True,
                "gas": "CO2",
                "marker_radius_px": 6,
                "points": [
                    {"point_id": "native-1", "px": [40, 40], "x": 40, "y": 40, "confidence": 0.92},
                    {"point_id": "native-2", "px": [60, 60], "x": 60, "y": 60, "confidence": 0.9},
                ],
            }
        ],
    }


def test_add_move_reassign_collisions_preserve_existing_point():
    spec = {"panel_bbox": [0, 0, 100, 100]}
    extraction = _staged_extraction()
    response = {
        "operations": [
            {"action": "add", "series_label": "CO2 273K", "x_norm": 0.41, "y_norm": 0.4,
             "evidence_kind": "native_visible", "uncertainty_px": 1.0, "evidence_ref": "panel.png", "source_evidence": "marker"},
            {"action": "move", "point_id": "native-1", "x_norm": 0.6, "y_norm": 0.6,
             "evidence_kind": "native_visible", "uncertainty_px": 1.0, "evidence_ref": "panel.png", "source_evidence": "marker"},
            {"action": "reassign", "point_id": "native-1", "series_label": "CO2 273K", "x_norm": 0.6, "y_norm": 0.605,
             "evidence_kind": "native_visible", "uncertainty_px": 1.0, "evidence_ref": "panel.png", "source_evidence": "marker"},
        ]
    }
    result, audit = apply_agent_edits(extraction, spec, response, "agent03")
    assert len(result["series"][0]["points"]) == 2
    assert all(item["status"] == "not_applied" for item in audit["operations"])
    assert all(item["detail"] == "same_series_marker_collision" for item in audit["operations"])
    assert all(item["collision_prevented"] for item in audit["operations"])


def test_edits_into_masked_regions_are_rejected_before_artifact_rebuild():
    extraction = _staged_extraction()
    spec = {"panel_bbox": [0, 0, 100, 100], "mask_bboxes": [[10, 10, 30, 30]]}
    response = {"operations": [
        {"action": "add", "series_label": "CO2 273K", "x_norm": 0.2, "y_norm": 0.2,
         "evidence_kind": "native_visible", "uncertainty_px": 1.0, "evidence_ref": "panel.png", "source_evidence": "marker"},
        {"action": "move", "point_id": "native-1", "x_norm": 0.2, "y_norm": 0.2,
         "evidence_kind": "native_visible", "uncertainty_px": 1.0, "evidence_ref": "panel.png", "source_evidence": "marker"},
        {"action": "reassign", "point_id": "native-2", "series_label": "CO2 273K", "x_norm": 0.2, "y_norm": 0.2,
         "evidence_kind": "native_visible", "uncertainty_px": 1.0, "evidence_ref": "panel.png", "source_evidence": "marker"},
    ]}
    result, audit = apply_agent_edits(extraction, spec, response, "agent04")
    assert audit["requested"] == 3 and audit["applied"] == 0
    assert [point["px"] for point in result["series"][0]["points"]] == [[40, 40], [60, 60]]
    assert all(item["detail"] == "position is inside an excluded source region" for item in audit["operations"])


def test_partial_marker_move_preserves_citation_uncertainty_and_caps_old_confidence():
    extraction = _staged_extraction()
    response = {"operations": [{
        "action": "move", "point_id": "native-1", "x_norm": 0.45, "y_norm": 0.4,
        "evidence_kind": "partial_marker", "uncertainty_px": 2.0,
        "evidence_ref": "tile:partial-1", "source_evidence": "Visible arc at native panel pixel (45, 40).",
    }]}
    result, audit = apply_agent_edits(extraction, {"panel_bbox": [0, 0, 100, 100]}, response, "agent04")
    point = result["series"][0]["points"][0]
    assert audit["applied"] == 1
    assert point["px"] == [45, 40]
    assert point["evidence_ref"] == "tile:partial-1"
    assert point["uncertainty_px"] == 2.0
    assert point["confidence"] <= 0.64 < 0.92


def test_direct_positional_edits_without_finite_uncertainty_are_rejected():
    extraction = _staged_extraction()
    response = {"operations": [
        {"action": "add", "series_label": "CO2 273K", "x_norm": 0.2, "y_norm": 0.2},
        {"action": "move", "point_id": "native-1", "x_norm": 0.2, "y_norm": 0.2,
         "evidence_kind": "native_visible", "uncertainty_px": float("nan"),
         "evidence_ref": "panel.png", "source_evidence": "native marker"},
        {"action": "reassign", "point_id": "native-2", "series_label": "CO2 273K", "x_norm": 0.2, "y_norm": 0.2,
         "evidence_kind": "native_visible", "uncertainty_px": True,
         "evidence_ref": "panel.png", "source_evidence": "native marker"},
    ]}
    result, audit = apply_agent_edits(extraction, {"panel_bbox": [0, 0, 100, 100]}, response, "agent04")
    assert audit["applied"] == 0
    assert len(result["series"][0]["points"]) == 2
    assert all("evidence" in item["detail"] for item in audit["operations"])
    json.dumps(audit, allow_nan=False)

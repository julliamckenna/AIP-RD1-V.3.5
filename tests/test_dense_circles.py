import cv2
import numpy as np

from src.tools import dense_circles, series_gate


def _stacked_rings(spacing=6):
    image = np.zeros((180, 80), np.uint8)
    truth = [(40, 15 + spacing * index) for index in range(20)]
    for point in truth:
        cv2.circle(image, point, 6, 1, 2)
    return image, truth


def test_dense_open_rings_recover_single_series_vertical_stack():
    image, truth = _stacked_rings()
    found, audit = dense_circles.detect(image, [], [20, 5, 60, 150], 6)
    assert audit["validated_centers"] == len(truth)
    errors = [min(np.hypot(x - tx, y - ty) for x, y, _ in found) for tx, ty in truth]
    assert max(errors) < 0.2


def test_dense_ring_merge_is_two_dimensional_not_column_based():
    existing = [(40.0, 35.0, 0.9, False, True)]
    candidates = [(40.0, 35.2, 0.8), (40.0, 47.0, 0.8)]
    additions, matched = dense_circles.merge_2d(existing, candidates, 6)
    assert len(matched) == 1
    assert additions == [(40.0, 47.0, 0.8)]


def test_solid_dense_band_is_unresolved_not_open_rings():
    image = np.zeros((180, 80), np.uint8)
    cv2.rectangle(image, (34, 5), (46, 155), 1, -1)
    found, audit = dense_circles.detect(image, [], [20, 5, 60, 160], 6)
    assert not found
    assert audit["roi_ink_pixels"] > 0
    assert audit["rejected"]["solid_core"] > 0


def test_band_gate_keeps_marker_within_pixel_scale_boundary_margin():
    series = {"idx": 0, "r": 6.0, "spec": {"label": "CO2", "y_bands": [
        {"x_from": 0, "x_to": 1, "y_min": 65, "y_max": 120}], "color_hex": "#000000"},
        "pts": [(0.2, 36.4, 0.9, False, True)]}
    spec = {"x": {"ticks": [0, 1]}, "y": {"ticks": [0, 140]}, "series": [series["spec"]]}
    audit = series_gate.apply([series], spec, [np.float32([0, 0, 0])], lambda x: x, lambda y: 100 - y)
    assert series["pts"]
    assert audit[0]["reason"] == "within_pixel_derived_y_band_margin"

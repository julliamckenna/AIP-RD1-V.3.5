import pathlib
import copy
import json
import shutil
import uuid

import pytest


# The production image stack is optional in lightweight contract-test
# environments.  These focused checks run wherever the extraction runtime is
# installed, and skip cleanly otherwise.
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from src.workflow.steps.agent03_review_points import (  # noqa: E402
    MAX_IMAGES, MAX_REVIEW_ITEMS, _await_batch, _batch_groups, _batch_images, _batch_manifest,
    _load_batch_checkpoint, _merge_answer, _recovery_application_summary, _review_groups,
)
from src.workflow.resources import load_json, load_schema  # noqa: E402
from jsonschema import Draft202012Validator  # noqa: E402
from src.tools import adjudicate_tiles, dense_regions, extract, missing_strips  # noqa: E402


def test_review_batches_keep_every_item_under_image_cap():
    # The packer only needs existing paths; repository files avoid depending
    # on pytest's temporary-directory permissions on locked-down Windows.
    native = pathlib.Path(__file__)
    candidate = pathlib.Path(__file__).with_name("test_workflow_contracts.py")
    groups = [
        {"native": native, "candidates": [candidate], "items": [{"item_id": f"item-{i}"}]}
        for i in range(23)
    ]
    batches = _batch_groups(groups, evidence_slots=10)
    assert len(batches) > 1
    assert [item["item_id"] for batch in batches for group in batch for item in group["items"]] == [f"item-{i}" for i in range(23)]
    assert all(sum(1 + len(group["candidates"]) for group in batch) <= 10 for batch in batches)


def test_default_batches_combine_multiple_marker_tiles_without_losing_items():
    paths = sorted((pathlib.Path(__file__).parents[1] / "src/tools").glob("*.py"))[:20]
    assert len(paths) == 20
    groups = [
        {"native": path, "candidates": [], "items": [
            {"item_id": f"tile-{tile}-box-{box}", "candidate_row_ids": [f"point-{tile}-{box}"]}
            for box in range(9)
        ]}
        for tile, path in enumerate(paths)
    ]
    batches = _batch_groups(groups)
    assert len(batches) == 5
    expected = [item["item_id"] for group in groups for item in group["items"]]
    assert [item["item_id"] for batch in batches for group in batch for item in group["items"]] == expected
    assert all(sum(len(group["items"]) for group in batch) <= MAX_REVIEW_ITEMS for batch in batches)


def test_batch_images_use_overview_and_all_evidence_without_full_native_panel():
    root = pathlib.Path(__file__).parent
    native = root / "test_agent03_coverage.py"
    candidate = root / "test_workflow_contracts.py"
    second = root / "test_reviewer_edit_contract.py"
    overview = root / "review_python_overlay.png"
    batch = [
        {"native": native, "candidates": [candidate]},
        {"native": native, "candidates": [candidate, second]},
    ]
    images, attached = _batch_images(batch, overview)
    assert images == [overview, native, candidate, second]
    assert attached == [native.name, candidate.name, second.name]
    assert not any(path.name == "panel.png" for path in images)
    assert len(images) <= MAX_IMAGES
    manifest = _batch_manifest({"panel_size_px": [1473, 881]}, [], attached, 1, 1)
    assert manifest["panel_size_px"] == [1473, 881]


def test_merge_records_omitted_manifest_items_unresolved():
    result = _merge_answer(
        [{"reasoning": "ok", "verdict": "accept", "series": [], "operations": [], "uncertainties": [], "coverage": []}],
        [{"item_id": "box:marker_01:1", "item_type": "box", "series_label": "CO2 273K"}],
        {"series": [{"label": "CO2 273K", "extract": True, "points": []}]},
    )
    assert result["coverage"] == [{"item_id": "box:marker_01:1", "item_type": "box", "status": "unresolved", "detail": "not returned by the review batch"}]
    assert result["verdict"] == "review"


def test_merge_namespaces_same_local_operation_id_from_different_batches():
    operation = {
        "operation_id": "op-1", "action": "delete", "point_id": "pt-a",
        "series_label": None, "x_norm": None, "y_norm": None,
        "overlap": False, "reason": "false marker", "source_evidence": "native pixels",
        "evidence_kind": "native_absence", "uncertainty_px": None, "evidence_ref": "tile:box-1",
    }
    second = {**operation, "point_id": "pt-b"}
    base = {"reasoning": "ok", "verdict": "accept", "series": [],
            "uncertainties": [], "coverage": []}
    result = _merge_answer(
        [{**base, "operations": [operation]}, {**base, "operations": [second]}],
        [],
        {"series": []},
    )
    assert [item["point_id"] for item in result["operations"]] == ["pt-a", "pt-b"]
    assert [item["operation_id"] for item in result["operations"]] == [
        "batch-001-op-1", "batch-002-op-1",
    ]


def test_ambiguity_contact_sheets_preserve_pixels_and_map_every_crop():
    work = pathlib.Path.cwd() / f".test-contact-sheets-{uuid.uuid4().hex}"
    work.mkdir()
    try:
        names = []
        originals = {}
        for index in range(27):
            name = f"ambiguity_{index + 1:03d}_native.png"
            image = np.full((31, 29, 3), (index * 7) % 255, dtype=np.uint8)
            cv2.circle(image, (14, 15), 4, (0, 0, 255), -1)
            assert cv2.imwrite(str(work / name), image)
            names.append(name)
            originals[name] = image

        sheets = adjudicate_tiles.build_ambiguity_contact_sheets(work, names)

        assert len(sheets) == 2
        cells = [cell for sheet in sheets for cell in sheet["cells"]]
        assert [cell["source_native_tile"] for cell in cells] == names
        first_sheet = cv2.imread(str(work / sheets[0]["native_tile"]))
        assert first_sheet.shape[0] <= 1600 and first_sheet.shape[1] <= 1600
        first = cells[0]
        x = (first["column"] - 1) * (29 + adjudicate_tiles.CONTACT_SHEET_GAP)
        y = ((first["row"] - 1) *
             (adjudicate_tiles.CONTACT_SHEET_LABEL_HEIGHT + 31 + adjudicate_tiles.CONTACT_SHEET_GAP) +
             adjudicate_tiles.CONTACT_SHEET_LABEL_HEIGHT)
        assert np.array_equal(first_sheet[y:y + 31, x:x + 29], originals[names[0]])
    finally:
        shutil.rmtree(work)


def test_agent03_batch_has_application_level_hard_timeout():
    import asyncio

    async def never_returns():
        await asyncio.Event().wait()

    with pytest.raises(TimeoutError, match="hard timeout"):
        asyncio.run(_await_batch(never_returns(), 2, 4, 0.01))


def test_missing_strip_recovery_is_noop_without_explicit_gate(monkeypatch):
    called = False

    def confirm(*args, **kwargs):
        nonlocal called
        called = True
        return (10.0, 20.0, 0.9)

    monkeypatch.setattr(missing_strips.extract, "confirm_marker_near", confirm)
    operations, audit = missing_strips.recover_confirmed_slots(
        {"unresolved_slots": {"slots": [{"series_label": "CO2 273K", "column_px": 10, "predicted_y_px": 20}]}, "series": []},
        {"panel_bbox": [0, 0, 100, 100]},
        {"satisfied": True, "issues": []},
        enabled=True,
    )
    assert operations == []
    assert not called
    assert audit[0]["status"] == "gated"


def test_recovery_application_summary_separates_confirmed_from_applied():
    requested = [
        {"operation_id": "missing-strip-0001", "action": "add"},
        {"operation_id": "missing-strip-0002", "action": "add"},
    ]
    application_audit = {"operations": [
        {"operation_id": "missing-strip-0001", "status": "applied"},
        {"operation_id": "missing-strip-0002", "status": "not_applied", "detail": "same_series_marker_collision"},
    ]}

    result = _recovery_application_summary(requested, application_audit)

    assert result["requested_count"] == 2
    assert result["applied_count"] == 1
    assert result["rejected_count"] == 1
    assert result["applied_operations"][0]["operation_id"] == "missing-strip-0001"
    assert result["rejected_operations"][0]["detail"] == "same_series_marker_collision"


def test_missing_strip_shape_rejects_thick_line_but_accepts_marker_on_line(tmp_path):
    source = np.full((90, 100, 3), 255, dtype=np.uint8)
    color = (0, 0, 255)
    cv2.line(source, (8, 45), (92, 45), color, 6)
    thick_line_path = tmp_path / "thick_line.png"
    assert cv2.imwrite(str(thick_line_path), source)
    line_spec = {"image": str(thick_line_path), "panel_bbox": [0, 0, 100, 90]}
    color_lab = extract.to_lab(np.uint8([[color]])).reshape(-1).tolist()

    assert missing_strips._native_shape_support(line_spec, color_lab, 6, (50, 45)) is None

    cv2.line(source, (8, 45), (92, 45), color, 2)
    cv2.circle(source, (50, 45), 6, color, -1)
    marker_on_line_path = tmp_path / "marker_on_line.png"
    assert cv2.imwrite(str(marker_on_line_path), source)
    marker_on_line_spec = {"image": str(marker_on_line_path), "panel_bbox": [0, 0, 100, 90]}

    support = missing_strips._native_shape_support(marker_on_line_spec, color_lab, 6, (50, 45))
    assert support is not None
    assert support["evidence_kind"] == "native_visible"


def test_batch_checkpoint_requires_matching_request_and_current_schema(tmp_path):
    path = tmp_path / "agent03_review_points_batch_001.json"
    answer = copy.deepcopy(load_json("examples/agent03_review_points.example.json"))
    path.write_text(json.dumps(answer), encoding="utf-8")
    validator = Draft202012Validator(load_schema("schemas/agent03_review_points.schema.json"))
    assert _load_batch_checkpoint(path, {}, "new", validator) is None
    checkpoint = {"batch_fingerprints": {path.name: "new"}}
    assert _load_batch_checkpoint(path, checkpoint, "changed-prompt", validator) is None
    assert _load_batch_checkpoint(path, checkpoint, "new", validator) == answer
    answer.pop("coverage")
    path.write_text(json.dumps(answer), encoding="utf-8")
    assert _load_batch_checkpoint(path, checkpoint, "new", validator) is None


def test_dense_diagnostics_are_attached_without_claiming_native_pixels(tmp_path):
    source = np.full((70, 80, 3), 255, dtype=np.uint8)
    source[4, 3] = (0, 0, 0)
    panel = tmp_path / "panel.png"
    assert cv2.imwrite(str(panel), source)
    for name in ("dense_roi_native.png", "dense_roi_line_suppressed.png"):
        assert cv2.imwrite(str(tmp_path / name), source)
    extraction = {"series": [{"label": "CO2 273K", "marker_radius_px": 3,
                               "points": [{"px": [3, 4], "x": 3, "y": 4, "confidence": 0.1}]}]}
    spec = {"image": str(panel), "panel_bbox": [0, 0, 80, 70], "legend_bbox": [60, 50, 79, 69]}
    _, manifest = adjudicate_tiles.build_review_tiles(extraction, spec, tmp_path)
    diagnostic = next(item for item in manifest["review_items"]
                      if item.get("diagnostic_tile") == "dense_roi_line_suppressed.png")
    assert diagnostic["native_tile"] is None
    assert diagnostic["evidence_kind"] == "derived_diagnostic"
    groups = _review_groups(manifest, tmp_path)
    assert any(group["native"].name == diagnostic["diagnostic_tile"] for group in groups)
    batch = _batch_manifest(manifest, [diagnostic], [diagnostic["diagnostic_tile"]], 1, 1)
    assert batch["dense_region_evidence"] == [diagnostic["diagnostic_tile"]]
    # An edge crop preserves geometry and pixel values, without stretching the
    # clipped crop to a full cell or introducing cubic-interpolation colors.
    tile = cv2.imread(str(tmp_path / "tile_01_native.png"))
    crop = source[:37, :36]
    expected = np.repeat(np.repeat(crop, adjudicate_tiles.ZOOM, axis=0), adjudicate_tiles.ZOOM, axis=1)
    assert np.array_equal(tile[22:22 + expected.shape[0], :expected.shape[1]], expected)


def test_dense_native_evidence_replicates_source_pixels(tmp_path):
    source = np.full((70, 80, 3), 255, dtype=np.uint8)
    source[15:20, 20:25] = (0, 0, 0)
    extraction = {"series": [], "calibration": {
        "dense_region_resolver": {"roi_px": [10, 10, 30, 30]},
        "series_assignment_audit": {"candidates": [{"action": "unresolved", "px": [20, 20]}]},
    }}
    dense_regions.write_evidence(source, extraction, {}, tmp_path)
    for filename, crop in [("dense_roi_native.png", source[10:30, 10:30]),
                           ("ambiguity_001_native.png", source[:53, :53])]:
        expected = np.repeat(np.repeat(crop, 3, axis=0), 3, axis=1)
        assert np.array_equal(cv2.imread(str(tmp_path / filename)), expected)

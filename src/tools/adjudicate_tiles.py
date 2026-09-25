"""Evidence tiles for agent03_review_points.

The extractor proposes marker locations and ownership. For disputed proposals,
this module builds paired 3x crops using nearest-neighbor pixel replication.
The unannotated crop retains source pixel values; its matching annotated crop
merely identifies the Python proposal under review.
"""

import json
import math
import pathlib
import re

import cv2
import numpy as np

from src.settings import CFG
from src.tools import dense_regions

ZOOM = 3
CROP = 32  # half-size of the crop around the marker (in original pixels) before zoom
PER_TILE = 8  # keep each visual packet and its point mapping small
CONTACT_SHEET_CELLS = 25
CONTACT_SHEET_COLUMNS = 5
CONTACT_SHEET_LABEL_HEIGHT = 24
CONTACT_SHEET_GAP = 6
STRIP_CROPS = 3
STRIP_OVERLAP_PX = 16
MAX_CONTEXT_ROWS = 4
CONF_OK = CFG["extract"]["confidence_ok"]


def lab_to_hex(lab):
    b, g, r = cv2.cvtColor(np.uint8([[[round(v) for v in lab]]]), cv2.COLOR_LAB2BGR)[
        0, 0
    ]
    return f"#{r:02x}{g:02x}{b:02x}"


def contested_points(extraction, scope=None):
    """(series index, point index) of every point worth asking about. scope 'all' = every measured marker."""
    scope = scope or "contested"
    out = []
    for si, s in enumerate(extraction["series"]):
        for pi, p in enumerate(s["points"]):
            if (
                p.get("source") in ("curve", "grid")
                or not p.get("px")
                or p["px"][0] is None
            ):
                continue
            if (
                scope == "all"
                or p["confidence"] < CONF_OK
                or p.get("overlap_flag")
                or p.get("assigned_by") in ("continuity", "curve")
            ):
                out.append((si, pi))
    return out


def build_tiles(extraction, spec, out_dir):
    """Write paired marker tiles and return their candidate-point references."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(spec["image"])
    x0, y0, x1, y1 = spec["panel_bbox"]
    panel = img[y0:y1, x0:x1]
    lx0, ly0, lx1, ly1 = spec["legend_bbox"]
    legend = panel[max(0, ly0 - 4) : ly1 + 4, max(0, lx0 - 4) : lx1 + 4]
    if legend.size == 0:
        legend = panel
    legend = cv2.resize(legend, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(out_dir / "legend.png"), legend)

    items = contested_points(extraction)
    tiles, box_no = [], 0
    for t in range(0, len(items), PER_TILE):
        chunk = items[t : t + PER_TILE]
        cols = 4
        rows = math.ceil(len(chunk) / cols)
        cell = (2 * CROP + 1) * ZOOM
        native_canvas = np.full(
            (rows * (cell + 26), cols * (cell + 10), 3), 255, np.uint8
        )
        candidate_canvas = native_canvas.copy()
        boxes = []
        for k, (si, pi) in enumerate(chunk):
            p = extraction["series"][si]["points"][pi]
            cx, cy = int(round(p["px"][0])), int(round(p["px"][1]))
            ya, yb = max(0, cy - CROP), min(panel.shape[0], cy + CROP + 1)
            xa, xb = max(0, cx - CROP), min(panel.shape[1], cx + CROP + 1)
            raw_crop = panel[ya:yb, xa:xb]
            raw_crop = cv2.resize(raw_crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
            crop = raw_crop.copy()
            rx, ry = (
                (cx - xa) * ZOOM + ZOOM // 2,
                (cy - ya) * ZOOM + ZOOM // 2,
            )
            cv2.circle(
                crop,
                (rx, ry),
                int(ZOOM * max(4, extraction["series"][si]["marker_radius_px"]) * 1.15),
                (255, 0, 255),
                2,
            )
            r_, c_ = divmod(k, cols)
            oy, ox = r_ * (cell + 26), c_ * (cell + 10)
            crop_height, crop_width = raw_crop.shape[:2]
            native_canvas[oy + 22 : oy + 22 + crop_height, ox : ox + crop_width] = raw_crop
            candidate_canvas[oy + 22 : oy + 22 + crop_height, ox : ox + crop_width] = crop
            box_no += 1
            cv2.putText(
                native_canvas,
                f"box {box_no}",
                (ox + 2, oy + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )
            cv2.putText(
                candidate_canvas,
                f"box {box_no}   x={p['x']:.3g} y={p['y']:.3g}",
                (ox + 2, oy + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )
            boxes.append((box_no, si, pi))
        tile_number = len(tiles) + 1
        native_path = out_dir / f"tile_{tile_number:02d}_native.png"
        candidate_path = out_dir / f"tile_{tile_number:02d}_candidate.png"
        cv2.imwrite(str(native_path), native_canvas)
        cv2.imwrite(str(candidate_path), candidate_canvas)
        tiles.append(
            {
                "tile": str(candidate_path),
                "native_tile": str(native_path),
                "candidate_tile": str(candidate_path),
                "boxes": boxes,
            }
        )
    return tiles, str(out_dir / "legend.png")


def build_ambiguity_contact_sheets(out_dir, source_names, max_cells=CONTACT_SHEET_CELLS):
    """Pack small native ambiguity crops without shrinking their pixels.

    Each source crop remains a distinct labeled cell and the returned mapping
    keeps its original filename, sheet filename, row and column auditable.
    Large region/dense views intentionally bypass this helper.
    """
    out_dir = pathlib.Path(out_dir)
    readable = []
    for name in source_names:
        image = cv2.imread(str(out_dir / name))
        if image is not None and image.size:
            readable.append((name, image))
    if not readable:
        return []

    sheets = []
    for offset in range(0, len(readable), max_cells):
        chunk = readable[offset : offset + max_cells]
        columns = min(CONTACT_SHEET_COLUMNS, len(chunk))
        rows = math.ceil(len(chunk) / columns)
        cell_width = max(image.shape[1] for _, image in chunk)
        cell_height = max(image.shape[0] for _, image in chunk)
        stride_x = cell_width + CONTACT_SHEET_GAP
        stride_y = CONTACT_SHEET_LABEL_HEIGHT + cell_height + CONTACT_SHEET_GAP
        canvas = np.full((rows * stride_y, columns * stride_x, 3), 255, np.uint8)
        cells = []
        for index, (source_name, image) in enumerate(chunk):
            row, column = divmod(index, columns)
            x = column * stride_x
            y = row * stride_y
            label = pathlib.Path(source_name).stem.replace("_native", "")
            cv2.putText(
                canvas,
                label,
                (x + 2, y + 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            height, width = image.shape[:2]
            image_x = x + (cell_width - width) // 2
            image_y = y + CONTACT_SHEET_LABEL_HEIGHT + (cell_height - height) // 2
            canvas[image_y : image_y + height, image_x : image_x + width] = image
            cells.append({
                "source_native_tile": source_name,
                "label": label,
                "row": row + 1,
                "column": column + 1,
            })
        sheet_name = f"ambiguity_contact_sheet_{len(sheets) + 1:02d}_native.png"
        cv2.imwrite(str(out_dir / sheet_name), canvas)
        sheets.append({"native_tile": sheet_name, "cells": cells})
    return sheets


def _series_context(extraction, spec):
    """Small legend context used to interpret nearby native crowded crops."""
    sources = (spec.get("llm_spec") or {}).get("series") or spec.get("series") or extraction.get("series", [])
    out = []
    for series in sources:
        label = str(series.get("label") or "")
        if not label:
            continue
        out.append({key: series.get(key) for key in (
            "label", "gas", "extract", "marker", "marker_fill", "line_style", "color_hex",
        ) if series.get(key) is not None})
    return out


def _candidate_rows_near(extraction, bbox_px, *, limit=MAX_CONTEXT_ROWS, center_px=None):
    """Return compact candidate rows whose native centers are near a source crop."""
    if not bbox_px or len(bbox_px) != 4:
        return []
    x0, y0, x1, y1 = map(float, bbox_px)
    cx, cy = center_px or ((x0 + x1) / 2, (y0 + y1) / 2)
    matches = []
    for series in extraction.get("series", []):
        label = str(series.get("label") or "")
        for point in series.get("points", []):
            px = point.get("px") or []
            if len(px) != 2 or px[0] is None or px[1] is None:
                continue
            px0, py0 = float(px[0]), float(px[1])
            if not (x0 <= px0 <= x1 and y0 <= py0 <= y1):
                continue
            item = {
                "point_id": point.get("point_id"), "series_label": label,
                "x": point.get("x"), "y": point.get("y"),
                "source_pixel": [round(px0, 2), round(py0, 2)],
                "evidence_type": point.get("evidence_type") or point.get("source"),
            }
            matches.append((math.hypot(px0 - cx, py0 - cy), item))
    matches.sort(key=lambda entry: entry[0])
    return [item for _, item in matches[:limit]]


def _native_strip_crops(panel, out_dir, bbox_px):
    """Write a few full-height, source-pixel slices of the calibrated low-x strip."""
    if bbox_px is None or len(bbox_px) != 4:
        return []
    height, width = panel.shape[:2]
    x0, y0, x1, y1 = map(float, bbox_px)
    xa, xb = max(0, int(math.floor(x0))), min(width, int(math.ceil(x1)))
    ya, yb = max(0, int(math.floor(y0))), min(height, int(math.ceil(y1)))
    if xa >= xb or ya >= yb:
        return []
    # Use exact 3x replication and overlapping y bands so each image stays
    # focused while the set covers the entire plot-height strip.
    max_band_height = max(1, 1500 // ZOOM)
    band_count = min(STRIP_CROPS, max(1, math.ceil((yb - ya) / max_band_height)))
    records = []
    for index in range(band_count):
        base_y0 = ya + round((yb - ya) * index / band_count)
        base_y1 = ya + round((yb - ya) * (index + 1) / band_count)
        sy0 = ya if index == 0 else max(ya, base_y0 - STRIP_OVERLAP_PX)
        sy1 = yb if index == band_count - 1 else min(yb, base_y1 + STRIP_OVERLAP_PX)
        native = panel[sy0:sy1, xa:xb]
        native = cv2.resize(native, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
        name = f"crowded_strip_native_{index + 1:02d}.png"
        cv2.imwrite(str(out_dir / name), native)
        records.append({"item_id": f"crowded_strip:{index + 1:02d}", "image": name,
                        "source_bbox_px": [xa, sy0, xb, sy1], "native_scale": ZOOM,
                        "kind": "crowded_strip"})
    return records


def _slot_search_crop(panel, out_dir, slot_x, frame_px, index, half_width=32):
    """Make a narrow full-height source crop for a slot without a y guess."""
    height, width = panel.shape[:2]
    if frame_px and len(frame_px) == 4:
        _, y0, _, y1 = map(float, frame_px)
    else:
        y0, y1 = 0, height
    ya, yb = max(0, int(math.floor(y0))), min(height, int(math.ceil(y1)))
    xa = max(0, int(math.floor(float(slot_x) - half_width)))
    xb = min(width, int(math.ceil(float(slot_x) + half_width + 1)))
    if xa >= xb or ya >= yb:
        return None
    scale = 2 if (yb - ya) * 2 <= 1500 and (xb - xa) * 2 <= 1500 else 1
    crop = panel[ya:yb, xa:xb]
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    name = f"slot_search_{index:04d}_native.png"
    cv2.imwrite(str(out_dir / name), crop)
    return {"image": name, "source_bbox_px": [xa, ya, xb, yb], "native_scale": scale}


def _low_pressure_bbox(extraction, spec, panel_shape):
    """Read the recorded strip or reconstruct it from the calibrated axes."""
    info = (extraction.get("calibration") or {}).get("dense_region_resolver") or {}
    recorded = info.get("low_pressure_strip_px")
    if recorded and len(recorded) == 4:
        return list(map(float, recorded))
    calibration = extraction.get("calibration") or {}
    frame = calibration.get("frame_px")
    model = (calibration.get("axis_models") or {}).get("x") or {}
    ticks = (spec.get("x") or {}).get("ticks") or []
    if not frame or len(frame) != 4 or len(ticks) < 2 or model.get("slope") in (None, 0):
        old_rois = info.get("roi_px") or []
        return old_rois[0] if old_rois else None
    low = (spec.get("x") or {}).get("minimum")
    high = (spec.get("x") or {}).get("maximum")
    limits = (float(low if low is not None else min(ticks)), float(high if high is not None else max(ticks)))
    return dense_regions.infer_low_pressure_roi(
        frame, [], fraction=float(CFG["extract"].get("dense_low_pressure_fraction", 0.20)),
        axis_scale=(spec.get("x") or {}).get("scale", "linear"), axis_limits=limits,
        slope=float(model["slope"]), intercept=float(model.get("intercept") or 0.0),
    )


def build_review_tiles(extraction, spec, out_dir, complex_regions=(), *, candidate_overlay=None):
    """Create agent 03's paired-tile evidence bundle and its auditable manifest."""
    out_dir = pathlib.Path(out_dir)
    tiles, legend_png = build_tiles(extraction, spec, out_dir)
    manifest_tiles = []
    for index, tile in enumerate(tiles, start=1):
        boxes = []
        for box_no, series_index, point_index in tile["boxes"]:
            series = extraction["series"][series_index]
            point = series["points"][point_index]
            px, py = map(float, point.get("px") or [0, 0])
            crop_bbox = [max(0, int(px - CROP)), max(0, int(py - CROP)),
                         min(int(spec["panel_bbox"][2] - spec["panel_bbox"][0]), int(px + CROP + 1)),
                         min(int(spec["panel_bbox"][3] - spec["panel_bbox"][1]), int(py + CROP + 1))]
            boxes.append(
                {
                    "box": box_no,
                    "point_id": point.get("point_id"),
                    "series_label": series.get("label"),
                    "source_pixel": point.get("px"),
                    "source_bbox_px": crop_bbox,
                    "native_scale": ZOOM,
                    "nearby_candidate_rows": _candidate_rows_near(extraction, crop_bbox, limit=4, center_px=(px, py)),
                    "reason": "low_confidence_or_overlap",
                }
            )
        manifest_tiles.append(
            {
                "tile_id": f"marker_{index:02d}",
                "native_tile": pathlib.Path(tile["native_tile"]).name,
                "candidate_tile": pathlib.Path(tile["candidate_tile"]).name,
                "boxes": boxes,
            }
        )

    panel = cv2.imread(spec["image"])
    if panel is None:
        raise OSError(f"could not load review source panel: {spec.get('image')}")
    overlay_path = pathlib.Path(candidate_overlay) if candidate_overlay else pathlib.Path(spec["image"]).with_name("python_overlay.png")
    overlay = cv2.imread(str(overlay_path)) if overlay_path.is_file() else None
    height, width = panel.shape[:2]
    for index, region in enumerate(complex_regions or (), start=1):
        if str(region.get("kind") or "").lower() == "inset":
            continue  # excluded source region: never make it review evidence
        bbox = region.get("bbox_norm") or []
        if len(bbox) != 4:
            continue
        x0, y0, x1, y1 = bbox
        xa, xb = sorted((max(0, int(x0 * width)), min(width, int(x1 * width))))
        ya, yb = sorted((max(0, int(y0 * height)), min(height, int(y1 * height))))
        if xa >= xb or ya >= yb:
            continue
        native = cv2.resize(
            panel[ya:yb, xa:xb], None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST
        )
        candidate_source = overlay if overlay is not None else panel
        candidate = cv2.resize(
            candidate_source[ya:yb, xa:xb],
            None,
            fx=ZOOM,
            fy=ZOOM,
            interpolation=cv2.INTER_CUBIC,
        )
        native_path = out_dir / f"region_{index:02d}_native.png"
        candidate_path = out_dir / f"region_{index:02d}_candidate.png"
        cv2.imwrite(str(native_path), native)
        cv2.imwrite(str(candidate_path), candidate)
        manifest_tiles.append(
            {
                "tile_id": f"region_{index:02d}",
                "native_tile": native_path.name,
                "candidate_tile": candidate_path.name,
                "boxes": [],
                "source_bbox_px": [xa, ya, xb, yb],
                "native_scale": ZOOM,
                "reason": str(region.get("reason") or "Agent 02 complex-region search"),
            }
        )

    strip_bbox = _low_pressure_bbox(extraction, spec, panel.shape)
    crowded_strip_crops = _native_strip_crops(panel, out_dir, strip_bbox)

    # The deterministic resolver writes these calibrated views before Agent 03
    # runs.  They are evidence, not candidate coordinates.
    dense_names = (
        "dense_roi_native.png",
        "dense_roi_line_suppressed.png",
        "dense_roi_series_likelihood.png",
        "dense_roi_proposed_assignments.png",
        "ambiguity_assignment_scores.png",
        "assignment_diagnostics.json",
    )
    # Dense ROI native pixels are covered by the mapped overlapping slices
    # above; avoid re-attaching one full-height image that transport would
    # downscale. Keep the compact derived diagnostics as navigation aids.
    present = [name for name in dense_names if name != "dense_roi_native.png" and (out_dir / name).is_file()]
    series_context = _series_context(extraction, spec)
    series_by_index = {
        index: str(item.get("label") or "")
        for index, item in enumerate((spec.get("llm_spec") or {}).get("series") or spec.get("series") or [])
    }
    try:
        diagnostics = json.loads((out_dir / "assignment_diagnostics.json").read_text(encoding="utf-8"))
        disputed = [item for item in diagnostics.get("candidates", [])
                    if item.get("action") in {"unresolved", "reassigned"}][:80]
    except (OSError, ValueError, TypeError):
        disputed = []
    native_ambiguity_tiles = sorted(
        path.name
        for path in out_dir.glob("ambiguity_*_native.png")
        if re.fullmatch(r"ambiguity_\d{3}_native\.png", path.name)
    )
    ambiguity_contact_sheets = build_ambiguity_contact_sheets(
        out_dir, native_ambiguity_tiles
    )
    ambiguity_sheet_by_source = {
        cell["source_native_tile"]: {
            "native_tile": sheet["native_tile"],
            "label": cell["label"],
            "row": cell["row"],
            "column": cell["column"],
        }
        for sheet in ambiguity_contact_sheets
        for cell in sheet["cells"]
    }

    # Keep a flat, stable review ledger in addition to the historical tile
    # shape.  Agent 03 is called in bounded batches, so this list is the
    # source of truth for deciding whether a tile/box/slot was actually
    # presented to a reviewer.  In particular, ambiguity crops must not be
    # silently omitted just because they are produced by the dense resolver.
    review_items = []
    for crop in crowded_strip_crops:
        bbox = crop["source_bbox_px"]
        nearby = _candidate_rows_near(extraction, bbox, limit=12)
        labels = list(dict.fromkeys(row["series_label"] for row in nearby if row.get("series_label")))
        crop.update({
            "item_type": "evidence", "native_tile": crop["image"],
            "evidence_kind": "native_source", "candidate_tile": None,
            "candidate_row_ids": [row.get("point_id") for row in nearby if row.get("point_id")],
            "nearby_candidate_rows": nearby,
            "competing_series": labels,
            "series_context": [item for item in series_context if item.get("label") in labels],
            "detail": "unchanged source crop across the full plot-height low-pressure strip",
        })
        review_items.append(dict(crop))

    for tile in manifest_tiles:
        tile_id = tile["tile_id"]
        native = tile.get("native_tile")
        candidate = tile.get("candidate_tile")
        if tile.get("boxes"):
            tile_rows = []
        else:
            tile_rows = _candidate_rows_near(extraction, tile.get("source_bbox_px"), limit=8)
        review_items.append({
            "item_id": f"tile:{tile_id}",
            "item_type": "tile",
            "tile_id": tile_id,
            "native_tile": native,
            "candidate_tile": candidate,
            "source_bbox_px": tile.get("source_bbox_px"),
            "source_regions_px": [box.get("source_bbox_px") for box in tile.get("boxes", [])],
            "native_scale": ZOOM,
            "candidate_row_ids": [row.get("point_id") for row in tile_rows if row.get("point_id")],
            "nearby_candidate_rows": tile_rows,
            "series_context": [item for item in series_context if item.get("label") in {
                row.get("series_label") for row in tile_rows
            }],
            "detail": tile.get("reason", ""),
        })
        for box in tile.get("boxes", []):
            review_items.append({
                "item_id": f"box:{tile_id}:{box['box']}",
                "item_type": "box",
                "tile_id": tile_id,
                "box": box["box"],
                "point_id": box.get("point_id"),
                "series_label": box.get("series_label"),
                "source_pixel": box.get("source_pixel"),
                "source_bbox_px": box.get("source_bbox_px"),
                "native_scale": box.get("native_scale"),
                "candidate_row_ids": [box.get("point_id")] if box.get("point_id") else [],
                "nearby_candidate_rows": box.get("nearby_candidate_rows", []),
                "series_context": [item for item in series_context if item.get("label") == box.get("series_label")],
                "competing_series": list(dict.fromkeys(row.get("series_label") for row in box.get("nearby_candidate_rows", [])
                                                       if row.get("series_label") and row.get("series_label") != box.get("series_label"))),
                "native_tile": native,
                "candidate_tile": candidate,
                "detail": box.get("reason", ""),
            })
    for name in present:
        is_native = name == "dense_roi_native.png"
        review_items.append({
            "item_id": f"evidence:{name}",
            "item_type": "evidence",
            "native_tile": name if is_native else None,
            "diagnostic_tile": None if is_native else name,
            "evidence_kind": "native_source" if is_native else "derived_diagnostic",
            "source_bbox_px": strip_bbox if is_native else None,
            "native_scale": ZOOM if is_native else None,
            "candidate_row_ids": [],
            "nearby_candidate_rows": _candidate_rows_near(extraction, strip_bbox, limit=MAX_CONTEXT_ROWS) if is_native and strip_bbox else [],
            "series_context": series_context if is_native else [],
            "competing_series": [],
            "candidate_tile": None,
            "detail": "unchanged source crop" if is_native else "Python-derived diagnostic; not source pixels or proof of a marker",
        })
    for ambiguity_index, name in enumerate(native_ambiguity_tiles):
        sheet_cell = ambiguity_sheet_by_source.get(name, {})
        candidate = disputed[ambiguity_index] if ambiguity_index < len(disputed) else {}
        center = candidate.get("px") or []
        if len(center) == 2:
            cx, cy = map(float, center)
            bbox = [max(0, int(cx - CROP)), max(0, int(cy - CROP)),
                    min(width, int(cx + CROP + 1)), min(height, int(cy + CROP + 1))]
        else:
            bbox = None
        nearby = _candidate_rows_near(extraction, bbox, limit=6, center_px=center if len(center) == 2 else None)
        competitors = list(dict.fromkeys(str(label) for label in (
            (candidate.get("winner") or {}).get("series_label"),
            (candidate.get("runner_up") or {}).get("series_label"),
            *(series_by_index.get(index, "") for index in candidate.get("source_series_indices", [])),
        ) if label))
        review_items.append({
            "item_id": f"ambiguity:{name}",
            "item_type": "ambiguity",
            # Attach the individual crop directly. The contact sheet remains
            # an optional index but is too ambiguous as the sole image input.
            "native_tile": name,
            "source_native_tile": name,
            "sheet_cell": {
                "label": sheet_cell.get("label", pathlib.Path(name).stem),
                "row": sheet_cell.get("row"),
                "column": sheet_cell.get("column"),
            },
            "source_bbox_px": bbox,
            "native_scale": ZOOM,
            "candidate_row_ids": [row.get("point_id") for row in nearby if row.get("point_id")],
            "nearby_candidate_rows": nearby,
            "competing_series": competitors,
            "series_context": [item for item in series_context if item.get("label") in competitors],
            "candidate_tile": None,
            "detail": "native crop for a disputed series assignment",
        })
    for index, slot in enumerate(
        (extraction.get("unresolved_slots") or {}).get("slots", []), start=1
    ):
        # A slot is an explicit unresolved disposition even when it has no
        # usable predicted pixel.  It is attached to dense native evidence
        # when available, but never promoted to a coordinate by this ledger.
        slot_x = float(slot.get("column_px") or 0)
        slot_y = slot.get("predicted_y_px")
        if isinstance(slot_y, (int, float)):
            slot_bbox = [max(0, slot_x - 32), max(0, float(slot_y) - 32),
                         min(width, slot_x + 33), min(height, float(slot_y) + 33)]
            slot_center = (slot_x, float(slot_y))
        elif strip_bbox:
            slot_bbox = [max(0, slot_x - 32), float(strip_bbox[1]),
                         min(width, slot_x + 33), float(strip_bbox[3])]
            slot_center = (slot_x, (slot_bbox[1] + slot_bbox[3]) / 2)
        else:
            slot_bbox = [max(0, slot_x - 32), 0, min(width, slot_x + 33), height]
            slot_center = (slot_x, height / 2)
        nearby = _candidate_rows_near(extraction, slot_bbox, limit=MAX_CONTEXT_ROWS, center_px=slot_center)
        slot_crop = next((crop for crop in crowded_strip_crops
                          if isinstance(slot_y, (int, float))
                          and crop["source_bbox_px"][1] <= slot_y <= crop["source_bbox_px"][3]
                          and crop["source_bbox_px"][0] <= slot_x <= crop["source_bbox_px"][2]), None)
        slot_native = slot_crop
        if slot_native is None and isinstance(slot_y, (int, float)):
            xa, ya, xb, yb = [int(round(value)) for value in slot_bbox]
            if xa < xb and ya < yb:
                name = f"slot_native_{index:04d}.png"
                crop = panel[ya:yb, xa:xb]
                crop = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(out_dir / name), crop)
                slot_native = {"image": name, "source_bbox_px": [xa, ya, xb, yb], "native_scale": ZOOM}
        elif slot_native is None:
            frame = (extraction.get("calibration") or {}).get("frame_px") or [0, 0, width, height]
            slot_native = _slot_search_crop(panel, out_dir, slot_x, frame, index)
        if slot_native:
            slot_bbox = slot_native["source_bbox_px"]
            nearby = _candidate_rows_near(
                extraction, slot_bbox, limit=MAX_CONTEXT_ROWS,
                center_px=(slot_x, (slot_bbox[1] + slot_bbox[3]) / 2)
            )
        competing = list(dict.fromkeys(row["series_label"] for row in nearby
                                      if row.get("series_label") and row.get("series_label") != slot.get("series_label")))
        review_items.append({
            "item_id": f"slot:{index:04d}",
            "item_type": "slot",
            "native_tile": slot_native["image"] if slot_native else "panel.png",
            "candidate_tile": None,
            "series_label": slot.get("series_label"),
            "source_pixel": [slot.get("column_px"), slot.get("predicted_y_px")],
            "source_bbox_px": slot_bbox,
            "native_scale": slot_native["native_scale"] if slot_native else 1,
            "candidate_row_ids": [row.get("point_id") for row in nearby if row.get("point_id")],
            "nearby_candidate_rows": nearby,
            "competing_series": competing,
            "series_context": [item for item in series_context if item.get("label") in {slot.get("series_label"), *competing}],
            "detail": slot.get("reason") or slot.get("state") or "unresolved slot",
        })

    manifest = {
        "schema_version": "1.2",
        "authority": "supporting_evidence_only",
        "native_image": pathlib.Path(spec["image"]).name,
        "legend": pathlib.Path(legend_png).name,
        "tiles": manifest_tiles,
        "dense_region_evidence": present,
        "ambiguity_native_evidence_tiles": native_ambiguity_tiles,
        "ambiguity_contact_sheets": ambiguity_contact_sheets,
        "crowded_native_evidence": crowded_strip_crops,
        "review_items": review_items,
        "instructions": (
            "Native tiles preserve source pixel values, with nearest-neighbor enlargement where needed. Candidate tiles identify Python proposals only. "
            "Diagnostic tiles (line suppression, likelihood, assignments and scores) are fallible derived views, not native evidence. "
            "Ambiguity contact sheets preserve each native crop as a labeled cell; use the manifest row and column to match item IDs. "
            "Each ambiguity crop is also attached individually with its source_bbox_px, nearby candidate row IDs and competing series. "
            "Crowded strip crops are unchanged source pixels, split horizontally to cover the full calibrated low-pressure strip from top to bottom. "
            "Search every complex-region native tile for missed CO2 markers; never use an annotated ring or count as coordinate evidence."
        ),
    }
    manifest_path = out_dir / "review_tiles.json"
    _write_json_atomic(manifest_path, manifest)
    return manifest_path, manifest


def _write_json_atomic(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=1), encoding="utf-8")
    temporary.replace(path)

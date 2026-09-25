"""Canonical python / agent03 / agent04 stage artifacts for the staged extraction workflow.

This module gives every stage stable row identities and writes the CSV
contract in schemas/points_csv.contract.json.  python_extract_points writes the python stage,
agent03_review_points the agent03 stage and agent04_final_check the agent04 stage.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import pathlib
import re

import cv2
import numpy as np

from src.models.naming import STAGES
from src.tools.extract import pixel_to_axis
from src.workflow.resources import load_json


STAGE_COLUMNS = list(load_json("schemas/points_csv.contract.json")["core_columns"])
AGGREGATION_COLUMNS = ["source_pdf", *STAGE_COLUMNS, "review_html"]


def _digest(value, length=16):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:length]


def _slug(value):
    text = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return text[:40] or "series"


def _write_json(path, value):
    path = pathlib.Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, default=float), encoding="utf-8")
    tmp.replace(path)


def _write_csv(path, rows, columns=STAGE_COLUMNS):
    path = pathlib.Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _color_hex(lab):
    if not lab:
        return ""
    px = np.uint8([[[round(float(v)) for v in lab]]])
    blue, green, red = cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0]
    return f"#{red:02x}{green:02x}{blue:02x}"


def _confidence(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "low"
    if value >= 0.85:
        return "high"
    if value >= 0.65:
        return "medium"
    return "low"


def demote_deleted_markers_to_curve_traces(
    candidate_rows, final_rows, candidate_extraction, spec, image_path, calibration,
    color_tolerance=45.0,
):
    """Keep source-supported Agent 04 deletions as line samples, never markers.

    Only deleted candidate clusters with native color support are carried into
    the final JSON's separate ``curve_traces`` layer. They do not enter the
    final CSV or aggregate point table.
    """
    final_ids = {str(row.get("point_id") or "") for row in final_rows}
    deleted = [
        row for row in candidate_rows
        if row.get("point_id") and str(row.get("point_id")) not in final_ids
        and str(row.get("export_eligible", "")).lower() == "true"
        and "".join(ch for ch in str(row.get("species", "")).upper() if ch.isalnum()) == "CO2"
    ]
    if not deleted:
        return [], {"demoted_candidate_count": 0, "trace_sample_count": 0, "skipped": "no deleted CO2 candidate rows"}

    image_path = pathlib.Path(image_path)
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return [], {"demoted_candidate_count": 0, "trace_sample_count": 0, "skipped": "source panel image unavailable"}
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    height, width = lab.shape[:2]
    excluded_regions = []
    boxes = list(spec.get("mask_bboxes") or [])
    for region in spec.get("region_strategies") or []:
        labels = list(region.get("series_labels") or [])
        if region.get("strategy") == "ignore" and not labels:
            box = region.get("bbox_px")
            if box:
                boxes.append(box)
                excluded_regions.append({"bbox_px": list(box), "reason": region.get("reason", "ignore region")})
    excluded_regions.extend(
        {"bbox_px": list(box), "reason": "spec.mask_bboxes"} for box in (spec.get("mask_bboxes") or [])
    )
    allowed = np.ones((height, width), dtype=bool)
    frame = calibration.get("frame_px") or [0, 0, width, height]
    left, top, right, bottom = map(int, frame)
    frame_mask = np.zeros((height, width), dtype=bool)
    frame_mask[max(0, top):min(height, bottom), max(0, left):min(width, right)] = True
    allowed &= frame_mask
    for box in boxes:
        x0, y0, x1, y1 = map(int, box)
        allowed[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = False

    series_radius = {
        str(series.get("label")): max(3.0, float(series.get("marker_radius_px") or 6.0))
        for series in candidate_extraction.get("series", [])
    }
    groups = {}
    for row in deleted:
        try:
            px = float(row["source_pixel_x"])
            py = float(row["source_pixel_y"])
        except (KeyError, TypeError, ValueError):
            continue
        label = str(row.get("series_name") or "")
        groups.setdefault((label, str(row.get("color_hex") or "")), []).append((px, py, row))

    traces = []
    demoted_ids = []
    for (label, color_hex), points in groups.items():
        try:
            color = color_hex.lstrip("#")
            red, green, blue = (int(color[offset:offset + 2], 16) for offset in (0, 2, 4))
            bgr_color = np.uint8([[[blue, green, red]]])
            target_lab = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
        except (ValueError, IndexError):
            continue
        color_mask = (np.linalg.norm(lab - target_lab, axis=2) <= float(color_tolerance)) & allowed
        radius = series_radius.get(label, 6.0)

        # Cluster nearby deletions so unrelated gaps in one chart do not get
        # bridged. A 3r neighborhood covers connected marker bands while
        # requiring native ink support at every emitted sample.
        remaining = list(points)
        clusters = []
        while remaining:
            cluster = [remaining.pop(0)]
            changed = True
            while changed:
                changed = False
                for item in remaining[:]:
                    if any(np.hypot(item[0] - member[0], item[1] - member[1]) <= 3.0 * radius for member in cluster):
                        cluster.append(item)
                        remaining.remove(item)
                        changed = True
            clusters.append(cluster)

        for cluster in clusters:
            if len(cluster) < 3:
                continue
            xs = np.asarray([point[0] for point in cluster], dtype=float)
            ys = np.asarray([point[1] for point in cluster], dtype=float)
            vertical = float(ys.max() - ys.min()) >= 2.0 * max(1.0, float(xs.max() - xs.min()))
            support_count = 0
            for px, py, _ in cluster:
                x0, x1 = max(0, int(px - radius)), min(width, int(px + radius + 1))
                y0, y1 = max(0, int(py - radius)), min(height, int(py + radius + 1))
                support_count += bool(color_mask[y0:y1, x0:x1].any())
            if support_count < max(3, int(np.ceil(0.75 * len(cluster)))):
                continue

            # Extend a trace to an immediately adjacent retained marker only
            # when that marker is close to the deleted cluster and the same
            # color mask supports the full sweep. This keeps the curve joined
            # while still refusing to draw through unsupported gaps.
            sweep_x_min, sweep_x_max = float(xs.min()), float(xs.max())
            sweep_y_min, sweep_y_max = float(ys.min()), float(ys.max())
            retained_neighbors = {"low": [], "high": []}
            for row in final_rows:
                if str(row.get("series_name") or "") != label:
                    continue
                row_color = str(row.get("color_hex") or "").lower()
                if row_color and row_color != color_hex.lower():
                    continue
                try:
                    px = float(row["source_pixel_x"])
                    py = float(row["source_pixel_y"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not np.isfinite([px, py]).all():
                    continue

                if vertical:
                    if py < sweep_y_min:
                        gap = float(np.hypot(max(sweep_x_min - px, 0.0, px - sweep_x_max), sweep_y_min - py))
                        side = "low"
                    elif py > sweep_y_max:
                        gap = float(np.hypot(max(sweep_x_min - px, 0.0, px - sweep_x_max), py - sweep_y_max))
                        side = "high"
                    else:
                        continue
                    if gap <= 3.0 * radius:
                        retained_neighbors[side].append((gap, px, py))
                else:
                    if px < sweep_x_min:
                        gap = float(np.hypot(sweep_x_min - px, max(sweep_y_min - py, 0.0, py - sweep_y_max)))
                        side = "low"
                    elif px > sweep_x_max:
                        gap = float(np.hypot(px - sweep_x_max, max(sweep_y_min - py, 0.0, py - sweep_y_max)))
                        side = "high"
                    else:
                        continue
                    if gap <= 3.0 * radius:
                        retained_neighbors[side].append((gap, px, py))

            for side in ("low", "high"):
                if not retained_neighbors[side]:
                    continue
                _, px, py = min(retained_neighbors[side], key=lambda item: item[0])
                sweep_x_min, sweep_x_max = min(sweep_x_min, px), max(sweep_x_max, px)
                sweep_y_min, sweep_y_max = min(sweep_y_min, py), max(sweep_y_max, py)

            samples = []
            if vertical:
                x0 = max(0, int(np.floor(sweep_x_min - radius)))
                x1 = min(width, int(np.ceil(sweep_x_max + radius)) + 1)
                for py in range(int(np.ceil(sweep_y_max)), int(np.floor(sweep_y_min)) - 1, -1):
                    ink_x = np.flatnonzero(color_mask[py, x0:x1]) + x0
                    if not len(ink_x):
                        continue
                    px = float(np.median(ink_x))
                    uncertainty = max(2.0, min(12.0, (float(ink_x.max() - ink_x.min()) + 1) / 2))
                    samples.append((px, float(py), uncertainty))
            else:
                y0 = max(0, int(np.floor(sweep_y_min - radius)))
                y1 = min(height, int(np.ceil(sweep_y_max + radius)) + 1)
                for px in range(int(np.floor(sweep_x_min)), int(np.ceil(sweep_x_max)) + 1):
                    ink_y = np.flatnonzero(color_mask[y0:y1, px]) + y0
                    if not len(ink_y):
                        continue
                    py = float(np.median(ink_y))
                    uncertainty = max(2.0, min(12.0, (float(ink_y.max() - ink_y.min()) + 1) / 2))
                    samples.append((float(px), py, uncertainty))
            if len(samples) < 2:
                continue

            # Break across unsupported source gaps instead of drawing an
            # interpolated bridge; each segment uses existing line_sample semantics.
            segments, segment = [], [samples[0]]
            for sample in samples[1:]:
                gap = abs(sample[1] - segment[-1][1]) if vertical else abs(sample[0] - segment[-1][0])
                if gap > 2.0:
                    if len(segment) >= 2:
                        segments.append(segment)
                    segment = [sample]
                else:
                    segment.append(sample)
            if len(segment) >= 2:
                segments.append(segment)

            for segment_index, segment in enumerate(segments):
                trace_points = []
                for px, py, uncertainty in segment:
                    trace_points.append({
                        "x": round(float(pixel_to_axis(calibration, "x", px)), 6),
                        "y": round(float(pixel_to_axis(calibration, "y", py)), 6),
                        "px": [round(px, 3), round(py, 3)],
                        "confidence": 0.5,
                        "overlap_flag": True,
                        "source": "curve",
                        "assigned_by": "agent04_deleted_marker_demoted_to_line_sample",
                        "uncertainty_px": round(uncertainty, 2),
                    })
                traces.append({
                    "series_name": label,
                    "series_id": cluster[0][2].get("series_id"),
                    "color_hex": color_hex,
                    "line_style": str(cluster[0][2].get("line_style") or "solid"),
                    "source": "curve",
                    "assigned_by": "agent04_deleted_marker_demoted_to_line_sample",
                    "evidence_type": "line_sample",
                    "marker_shape": "none",
                    "coordinate_space": "source_panel_pixels",
                    "orientation": "vertical" if vertical else "x_ordered",
                    "segment_index": segment_index,
                    "points": trace_points,
                    "provenance": {
                        "source_image": image_path.name,
                        "source_image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                        "source_candidate_ids": [point[2].get("point_id") for point in cluster],
                        "method": "Agent 04 deleted marker candidates were retained only as source='curve' line samples after native Lab color-mask support; coordinates are raster-supported band medians.",
                        "color_space": "CIELAB",
                        "target_color_hex": color_hex,
                        "color_distance_tolerance": float(color_tolerance),
                        "excluded_regions": excluded_regions,
                        "other_series_excluded_by": "Only the deleted row's CO2 color mask is sampled; the N2 curve uses a different color.",
                        "uncertainty": "Per-sample uncertainty_px reflects half the supported ink span; fused ring/line ink leaves centerline ambiguity.",
                    },
                })
            demoted_ids.extend(point[2].get("point_id") for point in cluster)

    return traces, {
        "demoted_candidate_count": len(set(demoted_ids)),
        "trace_sample_count": sum(len(trace["points"]) for trace in traces),
        "source_image": image_path.name,
        "source_image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "trace_count": len(traces),
    }


def _axis_uncertainty(calibration, axis, pixel, value, radius_px):
    """Convert a native-pixel radius to a conservative axis-unit uncertainty."""
    try:
        pixel = float(pixel)
        value = float(value)
        radius_px = float(radius_px)
        if radius_px < 0 or not np.isfinite([pixel, value, radius_px]).all():
            return ""
        anchor = (value, pixel)
        center = float(pixel_to_axis(calibration, axis, pixel, anchor=anchor))
        bounds = [
            float(pixel_to_axis(calibration, axis, pixel - radius_px, anchor=anchor)),
            float(pixel_to_axis(calibration, axis, pixel + radius_px, anchor=anchor)),
        ]
        if not np.isfinite([center, *bounds]).all():
            return ""
        return round(max(abs(center - bound) for bound in bounds), 4)
    except (KeyError, TypeError, ValueError, OverflowError):
        return ""


def _marker(value, glyph=None):
    """Marker shape for the CSV: the confidently measured legend glyph wins over the declared name."""
    if glyph and glyph.get("confidence", 0) >= 0.75 and glyph.get("shape") not in (None, "none"):
        return glyph["shape"]
    value = str(value or "circle").lower().replace("-", "_")
    return {"triangle": "triangle_up", "x": "cross"}.get(value, value)


def _series_specs(spec):
    return {
        str(item.get("label")): item
        for item in (spec.get("llm_spec") or {}).get("series", [])
    }


def _extract_specs(spec):
    return {str(item.get("label")): item for item in spec.get("series", [])}


def _series_value(style, key, default=""):
    value = style.get(key)
    return default if value in (None, "") else value


def _is_co2(value):
    """Return whether a declared species is explicitly CO2."""
    normalized = re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
    return normalized == "CO2"


def _is_eligible_series(series, extraction_style):
    """Apply the CO2-only coordinate-row contract at every artifact boundary."""
    species = _series_value(
        series, "gas", _series_value(extraction_style, "gas", ""),
    )
    return bool(extraction_style.get("extract", series.get("extract", False))) and _is_co2(species)


def _is_masked_point(point, mask_bboxes):
    """Return whether a source-pixel point falls inside an excluded region."""
    px = point.get("px") or []
    if len(px) != 2 or px[0] is None or px[1] is None:
        return False
    try:
        x, y = float(px[0]), float(px[1])
    except (TypeError, ValueError):
        return False
    return any(
        float(left) <= x <= float(right) and float(top) <= y <= float(bottom)
        for left, top, right, bottom in mask_bboxes
    )


def _axis_metadata(axis_id, axis, calibration):
    model = (calibration.get("axis_models") or {}).get(axis_id, {})
    ticks = list(axis.get("ticks") or [])
    if axis_id == "y":
        # Pixel rows increase downward, while ordinary y-axis values increase
        # upward.  The extractor stores y tick pixels from top to bottom.
        ticks = sorted(ticks, reverse=True)
    pixels = list(calibration.get(f"{axis_id}_ticks_px") or [])
    anchors = [
        {"pixel": float(pixel), "value": float(value)}
        for pixel, value in zip(pixels, ticks)
    ]
    return {
        "axis_id": axis_id,
        "scale": axis.get("scale", "linear"),
        "unit": axis.get("unit", ""),
        "quantity_type": axis.get("title") or ("pressure" if axis_id == "x" else "co2_loading"),
        "calibration_status": "valid" if len(anchors) >= 2 else "unresolved",
        "a": model.get("slope"),
        "b": model.get("intercept"),
        "anchors": anchors,
        "equation": "value=a*pixel+b" if axis.get("scale", "linear") == "linear" else "value=10**(a*pixel+b)",
    }


def build_candidate(extraction, spec, figure_id, panel_id, source_pdf=""):
    """Return canonical candidate JSON, CSV rows, and metadata."""
    candidate = copy.deepcopy(extraction)
    series_specs = _series_specs(spec)
    extract_specs = _extract_specs(spec)
    rows = []
    instances = []
    calibration = candidate.get("calibration") or {}
    metadata_series = []
    duplicate_counts = {}
    mask_bboxes = list(spec.get("mask_bboxes") or [])
    rejected_masked_points = 0

    eligible_series = []
    for series in candidate.get("series", []):
        label = str(series.get("label", "series"))
        style = series_specs.get(label, {})
        extraction_style = extract_specs.get(label, {})
        if not _is_eligible_series(series, extraction_style):
            continue
        retained_points = []
        for point in series.get("points", []):
            if _is_masked_point(point, mask_bboxes):
                rejected_masked_points += 1
                continue
            retained_points.append(point)
        series["points"] = retained_points
        series["n_points"] = len(retained_points)
        series_id = f"{figure_id}:{panel_id}:{_slug(label)}-{_digest(label, 8)}"
        series["series_id"] = series_id
        eligible_series.append(series)
        marker_shape = _marker(style.get("marker") or series.get("marker"), series.get("marker_glyph"))
        line_style = str(_series_value(style, "line_style", "solid"))
        color_hex = _color_hex(series.get("color_lab")) or str(_series_value(style, "color_hex"))
        species = str(_series_value(style, "gas", series.get("gas") or extraction_style.get("gas") or ""))
        export_eligible = bool(extraction_style.get("extract", series.get("extract", True)))
        species_evidence = str(_series_value(style, "species_evidence", ""))
        quantity_type = str(_series_value(style, "quantity_type", spec.get("y", {}).get("title") or "co2_loading"))
        loading_basis = str(_series_value(style, "loading_basis", "unknown"))
        metadata_series.append({
            "series_id": series_id,
            "series_name": label,
            "species": species,
            "export_eligible": export_eligible,
            "species_evidence": species_evidence,
            "marker_shape": marker_shape,
            "marker_fill": str(((series.get("marker_glyph") or {}).get("confidence", 0) >= 0.75
                                and series["marker_glyph"].get("fill")) or _series_value(style, "marker_fill", "filled")),
            "marker_glyph": series.get("marker_glyph"),
            "color_hex": color_hex,
            "line_style": line_style,
            "axis_x_id": "x",
            "axis_y_id": "y",
            "quantity_type": quantity_type,
            "loading_basis": loading_basis,
        })
        for point in series.get("points", []):
            px = point.get("px") or [None, None]
            source = str(point.get("source") or "auto")
            fingerprint = "|".join([
                str(figure_id), str(panel_id), label,
                "" if px[0] is None else f"{float(px[0]):.3f}",
                "" if px[1] is None else f"{float(px[1]):.3f}",
                source, str(point.get("branch") or "single"),
            ])
            occurrence = duplicate_counts.get(fingerprint, 0)
            duplicate_counts[fingerprint] = occurrence + 1
            source_instance_id = point.get("source_instance_id") or "src-" + _digest(f"{fingerprint}|{occurrence}")
            point_id = point.get("point_id") or "pt-" + _digest(source_instance_id)
            point["point_id"] = point_id
            point["source_instance_id"] = source_instance_id
            inferred = source in {"curve", "grid", "template_fill"} or px[0] is None or px[1] is None
            evidence_kind = point.get("evidence_kind")
            evidence_type = (
                "line_sample" if source in {"curve", "grid"}
                else source if source in {"template_fit", "template_fill"}
                else "partially_visible_marker" if evidence_kind == "partial_marker"
                else "visible_marker" if evidence_kind == "native_visible"
                else "partially_visible_marker" if point.get("overlap_flag") or source == "ai_proposed"
                else "visible_marker"
            )
            notes = []
            if point.get("branch"):
                notes.append(f"branch={point['branch']}")
            if point.get("overlap_flag"):
                notes.append("overlap_flag=true")
            if point.get("assigned_by"):
                notes.append(f"assigned_by={point['assigned_by']}")
            if source:
                notes.append(f"source={source}")
            if evidence_kind:
                notes.append(f"evidence_kind={evidence_kind}")
            uncertainty_px = point.get("uncertainty_px")
            if uncertainty_px is not None:
                notes.append(f"uncertainty_px={uncertainty_px}")
            evidence_ref = point.get("evidence_ref")
            if evidence_ref:
                notes.append(f"evidence_ref={evidence_ref}")
            source_evidence = str(point.get("source_evidence") or "").replace("; ", ", ")
            if source_evidence:
                notes.append(f"source_evidence={source_evidence}")
            uncertainty_x = _axis_uncertainty(calibration, "x", px[0], point.get("x"), uncertainty_px) if px[0] is not None and uncertainty_px is not None else ""
            uncertainty_y = _axis_uncertainty(calibration, "y", px[1], point.get("y"), uncertainty_px) if px[1] is not None and uncertainty_px is not None else ""
            row = {
                "figure_id": figure_id,
                "panel_id": panel_id,
                "series_id": series_id,
                "series_name": label,
                "x": point.get("x", ""),
                "x_unit": spec.get("x", {}).get("unit", ""),
                "y": point.get("y", ""),
                "y_unit": spec.get("y", {}).get("unit", ""),
                "marker_shape": "none" if evidence_type == "line_sample" else marker_shape,
                "line_style": line_style,
                "color_hex": color_hex,
                "is_inferred": str(bool(inferred)).lower(),
                "confidence": _confidence(point.get("confidence")),
                "source_pixel_x": "" if px[0] is None else px[0],
                "source_pixel_y": "" if px[1] is None else px[1],
                "notes": "; ".join(notes),
                "point_id": point_id,
                "source_instance_id": source_instance_id,
                "species": species,
                "export_eligible": str(export_eligible).lower(),
                "species_evidence": species_evidence,
                "quantity_type": quantity_type,
                "loading_basis": loading_basis,
                "evidence_type": evidence_type,
                "normalized_x": "", "normalized_x_unit": "",
                "normalized_y": "", "normalized_y_unit": "",
                "conversion_id": "", "uncertainty_x": uncertainty_x, "uncertainty_y": uncertainty_y,
            }
            rows.append(row)
            instances.append({
                "source_instance_id": source_instance_id,
                "point_id": point_id,
                "series_id": series_id,
                "state": "resolved",
                "source_pixel_x": None if px[0] is None else px[0],
                "source_pixel_y": None if px[1] is None else px[1],
                "evidence_type": evidence_type,
                "evidence_kind": evidence_kind,
                "uncertainty_px": uncertainty_px,
                "evidence_ref": point.get("evidence_ref"),
                "evidence": point.get("source_evidence") or "Extractor proposal retained for staged QA.",
                "review_only": inferred,
            })

    candidate["series"] = eligible_series

    image_path = pathlib.Path(spec.get("image", ""))
    image_sha = ""
    if image_path.is_file():
        image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    unresolved = candidate.get("unresolved_slots") or {"total": 0, "by_series": {}, "slots": []}
    slots = list(unresolved.get("slots") or [])
    slot_counts = {}
    for slot in slots:
        label = str(slot.get("series_label") or "")
        slot_counts[label] = slot_counts.get(label, 0) + 1
    unresolved_valid = (
        int(unresolved.get("total", 0)) == len(slots)
        and dict(unresolved.get("by_series") or {}) == dict(sorted(slot_counts.items()))
    )
    frame = calibration.get("frame_px") or spec.get("panel_bbox") or []
    metadata = {
        "schema_version": "4.0",
        "stage": "python",
        "figure_id": figure_id,
        "panel_id": panel_id,
        "source_pdf": str(source_pdf or ""),
        "image_sha256": image_sha,
        "status": "partial_review_required",
        "plot_bbox": frame,
        "series": metadata_series,
        "axes": [
            _axis_metadata("x", spec.get("x", {}), calibration),
            _axis_metadata("y", spec.get("y", {}), calibration),
        ],
        "instances": instances,
        "coverage": [{"bbox": frame, "inspected": False}] if frame else [],
        "limitations": [
            "This stage is replaced, not promoted, by the next extraction agent.",
        ],
        "rejected_masked_points": rejected_masked_points,
        "unresolved_slots": {"total": len(slots), "by_series": dict(sorted(slot_counts.items())), "slots": slots},
        "unresolved_accounting_valid": unresolved_valid,
        "row_count": len(rows),
    }
    return candidate, rows, metadata


def write_stage(
    out_dir, stage, extraction, spec, figure_id, panel_id, source_pdf="",
    agent_report=None, edit_audit=None, status=None,
):
    """Write one stage - python, agent03 or agent04 - as <stage>_points.csv/json and <stage>_metadata.json."""
    if stage not in STAGES:
        raise ValueError(f"unsupported extraction stage: {stage}")
    out_dir = pathlib.Path(out_dir)
    value, rows, metadata = build_candidate(
        extraction, spec, figure_id, panel_id, source_pdf=source_pdf,
    )
    verdict = str((agent_report or {}).get("verdict") or "").lower()
    if status is None:
        status = (
            "python_proposal" if stage == "python"
            else "agent03_adjusted" if stage == "agent03"
            else "complete_against_reviewed_evidence" if verdict == "accept"
            else "blocked" if verdict == "reject"
            else "partial_review_required"
        )
    metadata.update({
        "stage": stage,
        "status": status,
        "row_count": len(rows),
        "agent_report": agent_report or {},
        "edit_audit": edit_audit or {},
    })
    metadata["limitations"] = [] if status == "complete_against_reviewed_evidence" else [
        f"{stage} remains subject to downstream or human review."
    ]
    _write_json(out_dir / f"{stage}_points.json", value)
    _write_csv(out_dir / f"{stage}_points.csv", rows)
    _write_json(out_dir / f"{stage}_metadata.json", metadata)
    if edit_audit is not None:
        _write_json(out_dir / f"{stage}_edit_audit.json", edit_audit)
    return value, rows, metadata

"""Validate and load an Agent 04 authoritative replacement points CSV.

Unlike the operation based editor, this boundary treats the authored CSV as the
complete final point set.  It validates every row, rebuilds the normal
extraction shape used by plots and hard checks, and keeps the original row cells
on each point for downstream export.
"""

from __future__ import annotations

import copy
import csv
import math
import pathlib
import re
from collections import Counter

import cv2
from src.tools import stage_artifacts
from src.tools.extract import pixel_to_axis


_CONFIDENCE = {"high": 0.90, "medium": 0.75, "low": 0.40}
_EVIDENCE_TYPES = {
    "visible_marker", "partially_visible_marker", "template_fit", "template_fill",
    "line_sample", "reported_numeric", "manual",
}
_STATIC_STYLE_COLUMNS = (
    "series_id", "series_name", "x_unit", "y_unit", "line_style", "color_hex",
    "species", "export_eligible", "species_evidence", "quantity_type", "loading_basis",
)
_BOOLEAN_COLUMNS = ("is_inferred", "export_eligible")


class FinalCSVError(ValueError):
    """The final CSV is malformed or contradicts its source contract."""


def _fail(message: str) -> None:
    raise FinalCSVError(message)


def _finite(value, label: str, *, positive=False, nonnegative=False) -> float:
    if isinstance(value, bool):
        _fail(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        _fail(f"{label} must be a finite number")
    if not math.isfinite(number):
        _fail(f"{label} must be a finite number")
    if positive and number <= 0:
        _fail(f"{label} must be positive")
    if nonnegative and number < 0:
        _fail(f"{label} must be nonnegative")
    return number


def _numeric_text(value, label: str, *, optional=False, positive=False, nonnegative=False):
    if value == "" and optional:
        return None
    return _finite(value, label, positive=positive, nonnegative=nonnegative)


def _notes(row):
    result = {}
    for part in str(row.get("notes", "")).split(";"):
        key, sep, value = part.strip().partition("=")
        if sep and key.strip():
            result[key.strip()] = value.strip()
    return result


def _rounding_tolerance(text: str) -> float:
    """Half a unit in the final written decimal place (including exponent)."""
    value = str(text).strip().lower()
    match = re.fullmatch(r"[+-]?(?:\d+(?:\.(\d*))?|\.(\d+))(?:e([+-]?\d+))?", value)
    if not match:
        return 1e-8
    fraction = match.group(1) if match.group(1) is not None else match.group(2) or ""
    exponent = int(match.group(3) or 0)
    return min(5.1e-5, max(1e-8, 0.5000001 * 10.0 ** (exponent - len(fraction))))


def _validate_calibration(calibration, spec):
    if not isinstance(calibration, dict):
        _fail("report.calibration must contain the full calibration object")
    frame = calibration.get("frame_px")
    if not isinstance(frame, (list, tuple)) or len(frame) != 4:
        _fail("report.calibration.frame_px must contain [left, top, right, bottom]")
    frame = [_finite(value, "calibration frame coordinate") for value in frame]
    if frame[2] <= frame[0] or frame[3] <= frame[1]:
        _fail("calibration frame_px must have positive width and height")
    models = calibration.get("axis_models")
    if not isinstance(models, dict):
        _fail("report.calibration.axis_models is required")
    for axis in ("x", "y"):
        model = models.get(axis)
        if not isinstance(model, dict):
            _fail(f"report.calibration.axis_models.{axis} is required")
        slope = _finite(model.get("slope"), f"{axis} calibration slope")
        if abs(slope) <= 1e-15:
            _fail(f"{axis} calibration slope must be nonzero")
        _finite(model.get("intercept"), f"{axis} calibration intercept")
        if model.get("scale", "linear") not in {"linear", "log"}:
            _fail(f"{axis} calibration scale must be linear or log")
        expected_scale = str((spec.get(axis) or {}).get("scale") or "linear")
        if model.get("scale", "linear") != expected_scale:
            _fail(f"{axis} calibration scale conflicts with the chart specification")
    return frame


def _image_dimensions(spec):
    path = pathlib.Path(str(spec.get("image") or ""))
    if not path.is_file():
        _fail("spec.image must identify the source panel image")
    image = cv2.imread(str(path))
    if image is None:
        _fail("spec.image could not be read")
    return image.shape[1], image.shape[0]


def _validate_unresolved(unresolved):
    if not isinstance(unresolved, dict):
        _fail("report.unresolved_slots must be an object")
    if not {"total", "by_series", "slots"}.issubset(unresolved):
        _fail("report.unresolved_slots must preserve total, by_series, and slots")
    total = unresolved["total"]
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        _fail("report.unresolved_slots.total must be a nonnegative integer")
    if not isinstance(unresolved["by_series"], dict) or not isinstance(unresolved["slots"], list):
        _fail("report.unresolved_slots.by_series and slots have invalid types")
    if total != len(unresolved["slots"]):
        _fail("report.unresolved_slots.total must match its slots list")
    observed = Counter(str(slot.get("series_label") or "") for slot in unresolved["slots"]
                       if isinstance(slot, dict))
    expected = {key: value for key, value in sorted(observed.items())}
    if dict(unresolved["by_series"]) != expected:
        _fail("report.unresolved_slots.by_series must match its slots list")


def _series_maps(candidate, spec, figure_id, panel_id):
    raw = candidate.get("series")
    if not isinstance(raw, list):
        _fail("candidate.series must be a list")
    seen_point_ids, seen_source_ids = set(), set()
    for series in raw:
        for point in series.get("points", []):
            for key, seen, label in (("point_id", seen_point_ids, "point_id"),
                                     ("source_instance_id", seen_source_ids, "source_instance_id")):
                value = point.get(key)
                if value not in (None, ""):
                    value = str(value)
                    if value in seen:
                        _fail(f"candidate contains duplicate {label}: {value}")
                    seen.add(value)

    normalized, _, metadata = stage_artifacts.build_candidate(
        candidate, spec, figure_id, panel_id,
    )
    expected_by_series = {}
    series_by_id = {}
    series_by_label = {}
    candidate_by_id = {}
    for series in normalized.get("series", []):
        label = str(series.get("label", ""))
        sid = str(series.get("series_id", ""))
        if not sid or sid in series_by_id or label in series_by_label:
            _fail("candidate has missing or duplicate eligible series identities")
        series_by_id[sid] = series
        series_by_label[label] = series
        for point in series.get("points", []):
            pid = str(point.get("point_id", ""))
            source_id = str(point.get("source_instance_id", ""))
            if not pid or not source_id or pid in candidate_by_id:
                _fail("candidate has missing or duplicate point identities")
            candidate_by_id[pid] = (series, point)
    metadata_by_id = {str(item.get("series_id")): item for item in metadata.get("series", [])}
    for sid, series in series_by_id.items():
        item = metadata_by_id[sid]
        expected_by_series[sid] = {
            "series_id": sid,
            "series_name": str(series.get("label", "")),
            "x_unit": str((spec.get("x") or {}).get("unit", "")),
            "y_unit": str((spec.get("y") or {}).get("unit", "")),
            "marker_shape": item.get("marker_shape", "circle"),
            "line_style": item.get("line_style", "solid"),
            "color_hex": item.get("color_hex", ""),
            "species": item.get("species", ""),
            "export_eligible": str(bool(item.get("export_eligible"))).lower(),
            "species_evidence": item.get("species_evidence", ""),
            "quantity_type": item.get("quantity_type", ""),
            "loading_basis": item.get("loading_basis", ""),
        }
    return normalized, series_by_id, series_by_label, candidate_by_id, expected_by_series


def _parse_csv(path):
    try:
        with open(path, "r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames
            if header != stage_artifacts.STAGE_COLUMNS:
                _fail("CSV header must exactly match the canonical points column order")
            rows = []
            for index, row in enumerate(reader, 2):
                if None in row or any(value is None for value in row.values()):
                    _fail(f"CSV row {index} has a malformed number of cells")
                row = dict(row)
                # Python's csv writer serializes bool values as ``True``/``False``.
                # The canonical CSV contract uses lowercase JSON-style tokens, so
                # accept only those exact Python spellings and normalize them at
                # the file boundary. Other spellings remain invalid and are
                # rejected by _check_row.
                for column in _BOOLEAN_COLUMNS:
                    if row[column] in {"True", "False"}:
                        row[column] = row[column].lower()
                rows.append(row)
            return rows
    except UnicodeDecodeError as error:
        raise FinalCSVError("CSV must be UTF-8 encoded") from error
    except OSError as error:
        raise FinalCSVError(f"could not read final CSV: {error}") from error


def _check_row(row, index, spec, figure_id, panel_id, calibration, frame,
               image_size, series_by_id, series_by_label, candidate_by_id,
               expected_by_series, old_by_source):
    label = f"CSV row {index}"
    if row["figure_id"] != str(figure_id) or row["panel_id"] != str(panel_id):
        _fail(f"{label} has the wrong figure_id or panel_id")
    if not row["point_id"].strip() or not row["source_instance_id"].strip():
        _fail(f"{label} requires nonempty point_id and source_instance_id")
    if row["is_inferred"] not in {"true", "false"} or row["export_eligible"] != "true":
        _fail(f"{label} has invalid boolean fields or is not export eligible")
    if row["confidence"] not in _CONFIDENCE:
        _fail(f"{label} confidence must be high, medium, or low")
    if row["evidence_type"] not in _EVIDENCE_TYPES:
        _fail(f"{label} has an unsupported evidence_type")
    if row["evidence_type"] == "line_sample" and (row["marker_shape"] != "none" or row["is_inferred"] != "true"):
        _fail(f"{label} line_sample rows require marker_shape=none and is_inferred=true")
    if row["is_inferred"] == "true" and row["evidence_type"] not in {"line_sample", "template_fill"}:
        _fail(f"{label} inferred rows must be line_sample or template_fill")
    if row["species"].strip().upper() != "CO2" or not row["species_evidence"].strip():
        _fail(f"{label} must identify an eligible CO2 species with evidence")
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", row["color_hex"]):
        _fail(f"{label} color_hex must be a six-digit hex color")

    sid = row["series_id"]
    series = series_by_id.get(sid)
    if series is None or row["series_name"] != str(series.get("label", "")):
        _fail(f"{label} refers to a series that is not in the candidate")
    label_series = series_by_label.get(row["series_name"])
    if label_series is not series:
        _fail(f"{label} series identity is inconsistent")
    expected = expected_by_series.get(sid)
    if expected is None:
        _fail(f"{label} series is not an eligible CO2 series")
    for column in _STATIC_STYLE_COLUMNS:
        if row[column] != expected[column]:
            _fail(f"{label} {column} conflicts with the candidate series identity or style")
    expected_marker = "none" if row["evidence_type"] == "line_sample" else expected["marker_shape"]
    if row["marker_shape"] != expected_marker:
        _fail(f"{label} marker_shape conflicts with the candidate series style")
    if row["x_unit"] != str((spec.get("x") or {}).get("unit", "")):
        _fail(f"{label} x_unit conflicts with the chart specification")
    if row["y_unit"] != str((spec.get("y") or {}).get("unit", "")):
        _fail(f"{label} y_unit conflicts with the chart specification")

    x = _numeric_text(row["x"], f"{label} x")
    y = _numeric_text(row["y"], f"{label} y")
    px = _numeric_text(row["source_pixel_x"], f"{label} source_pixel_x")
    py = _numeric_text(row["source_pixel_y"], f"{label} source_pixel_y")
    if not (frame[0] <= px <= frame[2] and frame[1] <= py <= frame[3]):
        _fail(f"{label} native pixel lies outside the plot frame")
    if not (0 <= px < image_size[0] and 0 <= py < image_size[1]):
        _fail(f"{label} native pixel lies outside the panel image")
    for mask in spec.get("mask_bboxes") or []:
        if isinstance(mask, (list, tuple)) and len(mask) == 4:
            left, top, right, bottom = map(float, mask)
            if left <= px <= right and top <= py <= bottom:
                _fail(f"{label} native pixel lies in a masked region")

    try:
        expected_x = float(pixel_to_axis(calibration, "x", px))
        expected_y = float(pixel_to_axis(calibration, "y", py))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise FinalCSVError(f"{label} cannot be checked against the calibration") from error
    if not math.isfinite(expected_x) or not math.isfinite(expected_y):
        _fail(f"{label} calibration produces a non-finite value")
    if abs(x - expected_x) > _rounding_tolerance(row["x"]):
        _fail(f"{label} x disagrees with its native pixel and calibration")
    if abs(y - expected_y) > _rounding_tolerance(row["y"]):
        _fail(f"{label} y disagrees with its native pixel and calibration")

    for col in ("uncertainty_x", "uncertainty_y"):
        _numeric_text(row[col], f"{label} {col}", optional=True, nonnegative=True)
    for value_col, unit_col in (("normalized_x", "normalized_x_unit"),
                                ("normalized_y", "normalized_y_unit")):
        value = _numeric_text(row[value_col], f"{label} {value_col}", optional=True)
        if (value is None) != (row[unit_col] == ""):
            _fail(f"{label} {unit_col} must be present exactly when {value_col} is present")
    has_normalized = bool(row["normalized_x"] or row["normalized_y"])
    if has_normalized != bool(row["conversion_id"]):
        _fail(f"{label} conversion_id must accompany normalized values")

    notes = _notes(row)
    existing = candidate_by_id.get(row["point_id"])
    if existing:
        original_series, original_point = existing
        if row["source_instance_id"] != str(original_point.get("source_instance_id")):
            _fail(f"{label} changes source_instance_id for an existing point_id")
        moved = (float(original_point["px"][0]) != px or float(original_point["px"][1]) != py)
        reassigned = str(original_series.get("label")) != row["series_name"]
    else:
        moved, reassigned = False, False
        if row["point_id"] in candidate_by_id or row["source_instance_id"] in old_by_source:
            _fail(f"{label} reuses an existing point or source instance identity")
    if moved or existing is None:
        kind = notes.get("evidence_kind", "")
        support = notes.get("source_evidence", "")
        reference = notes.get("evidence_ref", "")
        uncertainty = _numeric_text(notes.get("uncertainty_px", ""),
                                    f"{label} notes uncertainty_px", optional=True, positive=True)
        if kind not in {"native_visible", "partial_marker"} or not support or not reference or uncertainty is None:
            _fail(f"{label} new or moved points require native evidence, reference, kind, and positive uncertainty_px notes")
    elif reassigned:
        if (notes.get("evidence_kind") not in {"native_visible", "partial_marker"}
                or not notes.get("source_evidence") or not notes.get("evidence_ref")):
            _fail(f"{label} reassignments require native evidence, reference, and evidence kind notes")

    evidence_kind = notes.get("evidence_kind") or {
        "visible_marker": "native_visible",
        "partially_visible_marker": "partial_marker",
    }.get(row["evidence_type"])
    source = notes.get("source") or {
        "line_sample": "curve", "template_fill": "template_fill", "template_fit": "template_fit",
    }.get(row["evidence_type"], "agent04_csv")
    return series, {"x": x, "y": y, "px": [px, py], "point_id": row["point_id"],
                    "source_instance_id": row["source_instance_id"],
                    "confidence": _CONFIDENCE[row["confidence"]],
                    "source": source,
                    "is_inferred": row["is_inferred"] == "true",
                    "evidence_type": row["evidence_type"],
                    "evidence_kind": evidence_kind,
                    "evidence_ref": notes.get("evidence_ref"),
                    "source_evidence": notes.get("source_evidence"),
                    "branch": notes.get("branch"),
                    "assigned_by": notes.get("assigned_by"),
                    "uncertainty_px": (_numeric_text(notes.get("uncertainty_px", ""),
                                                      f"{label} notes uncertainty_px", optional=True)),
                    "overlap_flag": notes.get("overlap_flag") == "true",
                    "csv_row": row}, moved, reassigned


def load_final_csv(csv_path, report, candidate, spec, figure_id, panel_id):
    """Load an Agent 04 full-replacement CSV into the normal extraction form.

    Returns ``(extraction, rows, audit)``.  ``rows`` and each point's
    ``csv_row`` retain all authored cells, including notes and optional fields.
    """
    if not isinstance(report, dict) or str(report.get("verdict", "")).lower() not in {"accept", "review", "reject"}:
        _fail("report must include a valid verdict")
    if not isinstance(report.get("series"), list):
        _fail("report.series must be the existing QA series report")
    if isinstance(report.get("row_count"), bool) or not isinstance(report.get("row_count"), int):
        _fail("report.row_count must be an integer")
    if not isinstance(report.get("changes"), list):
        _fail("report.changes must list the reason for every row-level change")
    calibration = copy.deepcopy(report.get("calibration"))
    frame = _validate_calibration(calibration, spec)
    image_size = _image_dimensions(spec)
    if not (0 <= frame[0] < frame[2] <= image_size[0]
            and 0 <= frame[1] < frame[3] <= image_size[1]):
        _fail("calibration frame_px lies outside the panel image")
    unresolved = copy.deepcopy(report.get("unresolved_slots"))
    _validate_unresolved(unresolved)

    (extraction, series_by_id, series_by_label, candidate_by_id,
     expected_by_series) = _series_maps(
        copy.deepcopy(candidate), spec, figure_id, panel_id,
    )
    old_by_source = {
        str(point.get("source_instance_id")): (series, point)
        for series, point in candidate_by_id.values()
    }
    rows = _parse_csv(csv_path)
    if report["row_count"] != len(rows):
        _fail("report.row_count must match the authoritative CSV row count")
    seen_points, seen_sources, seen_pixels = set(), set(), set()
    points_by_series = {label: [] for label in series_by_label}
    report_series = {}
    for item in report["series"]:
        if not isinstance(item, dict) or not isinstance(item.get("label"), str) or not item["label"].strip():
            _fail("each report.series entry requires a label")
        label = item["label"]
        if label in report_series:
            _fail(f"report.series contains duplicate label: {label}")
        if label not in series_by_label:
            _fail(f"report.series contains a noneligible or unknown label: {label}")
        report_series[label] = item
    if set(report_series) != set(series_by_label):
        _fail("report.series must include every eligible candidate series exactly once")
    audit = {
        "authoritative_csv": True,
        "added": [], "deleted": [], "moved": [], "reassigned": [],
        "candidate_count": len(candidate_by_id), "final_count": len(rows),
    }
    for index, row in enumerate(rows, 2):
        point_id, source_id = row["point_id"], row["source_instance_id"]
        if point_id in seen_points:
            _fail(f"duplicate point_id: {point_id}")
        if source_id in seen_sources:
            _fail(f"duplicate source_instance_id: {source_id}")
        seen_points.add(point_id)
        seen_sources.add(source_id)
        series, point, moved, reassigned = _check_row(
            row, index, spec, figure_id, panel_id, calibration, frame,
            image_size, series_by_id, series_by_label, candidate_by_id,
            expected_by_series, old_by_source,
        )
        pixel_key = (str(series.get("label")), point["px"][0], point["px"][1])
        if pixel_key in seen_pixels:
            _fail(f"duplicate native pixel in series {pixel_key[0]}: {pixel_key[1:]}")
        seen_pixels.add(pixel_key)
        points_by_series[str(series["label"])].append(point)
        prior = candidate_by_id.get(point_id)
        if prior is None:
            audit["added"].append(point_id)
        else:
            old_series, old_point = prior
            if moved:
                audit["moved"].append({"point_id": point_id, "from_px": list(old_point.get("px") or []),
                                       "to_px": list(point["px"])})
            if reassigned:
                audit["reassigned"].append({"point_id": point_id, "from": old_series.get("label"),
                                            "to": series.get("label")})
    audit["deleted"] = sorted(set(candidate_by_id) - seen_points)
    audit["series_counts"] = {
        label: {"candidate": len(series.get("points", [])), "final": len(points_by_series[label])}
        for label, series in series_by_label.items()
    }
    for label, item in report_series.items():
        final_count = item.get("final_count")
        if isinstance(final_count, bool) or not isinstance(final_count, int):
            _fail(f"report.series {label} requires an integer final_count")
        if final_count != len(points_by_series[label]):
            _fail(f"report.series {label} final_count does not match the authoritative CSV")

    inferred_changes = set()
    inferred_changes.update(("add", point_id) for point_id in audit["added"])
    inferred_changes.update(("delete", point_id) for point_id in audit["deleted"])
    inferred_changes.update(("move", item["point_id"]) for item in audit["moved"])
    inferred_changes.update(("reassign", item["point_id"]) for item in audit["reassigned"])
    reported_changes = {}
    for change in report["changes"]:
        if not isinstance(change, dict):
            _fail("each report.changes entry must be an object")
        action, point_id = change.get("action"), change.get("point_id")
        if action not in {"add", "move", "delete", "reassign"} or not isinstance(point_id, str) or not point_id.strip():
            _fail("report.changes entries require action and point_id")
        if not isinstance(change.get("reason"), str) or not change["reason"].strip():
            _fail(f"report change {action} {point_id} requires a reason")
        if not isinstance(change.get("source_evidence"), str) or not change["source_evidence"].strip():
            _fail(f"report change {action} {point_id} requires source_evidence")
        key = (action, point_id)
        if key in reported_changes:
            _fail(f"duplicate report change {action} for point {point_id}")
        reported_changes[key] = change
    if set(reported_changes) != inferred_changes:
        _fail("report.changes must match the additions, deletions, moves, and reassignments in the CSV")
    audit["reported_changes"] = copy.deepcopy(report["changes"])
    audit["requested"] = len(report["changes"])
    audit["applied"] = len(report["changes"])
    audit["operations"] = [
        {**copy.deepcopy(item), "status": "applied"}
        for item in report["changes"]
    ]
    audit["change_reasons"] = {
        f"{action}:{point_id}": {
            "reason": item["reason"], "source_evidence": item["source_evidence"],
        }
        for (action, point_id), item in reported_changes.items()
    }

    for series in extraction.get("series", []):
        label = str(series.get("label"))
        series["points"] = points_by_series[label]
        series["n_points"] = len(series["points"])
    extraction["calibration"] = calibration
    extraction["unresolved_slots"] = unresolved
    extraction["unresolved_total"] = unresolved["total"]
    extraction["authored_rows"] = rows
    extraction["agent04_report"] = copy.deepcopy(report)
    audit["contains_inferred"] = any(row["is_inferred"] == "true" for row in rows)
    audit["requires_review"] = audit["contains_inferred"]
    return extraction, rows, audit

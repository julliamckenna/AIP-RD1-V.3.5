"""Apply Agent 03/04 extraction edits to a staged extraction.

The agents choose which source instances to add, move, remove, or reassign.
This module performs only mechanical bookkeeping and calibrated pixel-to-axis
conversion so every model-authored change is retained in an audit trail.
"""

from __future__ import annotations

import copy
import math

from src.models import naming
from src.tools.extract import pixel_to_axis


def compact_rows(rows):
    """Return the coordinate/identity fields needed by an extraction agent.

    Per-series constants (marker shape, color, species, export flag) are left out - the agent
    has them from the structure - so a 17-series chart with hundreds of rows fits in a prompt.
    """
    keys = ("point_id", "series_name", "x", "y", "source_pixel_x", "source_pixel_y", "confidence", "evidence_type")
    compact = []
    for row in rows:
        item = {key: row.get(key) for key in keys}
        assigned = [part.split("=", 1)[1] for part in str(row.get("notes") or "").split("; ") if part.startswith("assigned_by=")]
        if assigned:
            item["assigned_by"] = assigned[0]
        compact.append(item)
    return compact


def rows_table(rows) -> str:
    """``compact_rows`` as CSV text for a prompt: one header line, one short line per point."""
    import csv
    import io

    compact = compact_rows(rows)
    columns = ["point_id", "series_name", "x", "y", "source_pixel_x", "source_pixel_y", "confidence", "evidence_type",
               "assigned_by"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(compact)
    return buffer.getvalue().rstrip("\n")


def _point_index(extraction):
    return {
        str(point.get("point_id")): (series, point)
        for series in extraction.get("series", [])
        for point in series.get("points", [])
        if point.get("point_id")
    }


def _series_index(extraction):
    return {naming.series_label(series.get("label")): series for series in extraction.get("series", [])}


def _eligible_series(series):
    gas = "".join(
        char for char in str(series.get("gas") or "").upper() if char.isalnum()
    )
    return bool(series.get("extract", False)) and gas == "CO2"


def _pixel(operation, width, height):
    x_norm, y_norm = operation.get("x_norm"), operation.get("y_norm")
    if x_norm is None or y_norm is None:
        return None
    if (isinstance(x_norm, bool) or isinstance(y_norm, bool)
            or not isinstance(x_norm, (int, float)) or not isinstance(y_norm, (int, float))):
        return None
    try:
        x_norm, y_norm = float(x_norm), float(y_norm)
    except (TypeError, ValueError):
        return None
    if (not math.isfinite(x_norm) or not math.isfinite(y_norm)
            or not 0 <= x_norm <= 1 or not 0 <= y_norm <= 1):
        return None
    return [round(x_norm * width, 2), round(y_norm * height, 2)]


def _inside_plot(px, extraction):
    """Return whether a panel-local pixel lies in the calibrated plot frame."""
    frame = (extraction.get("calibration") or {}).get("frame_px")
    if not frame or len(frame) != 4:
        return True
    left, top, right, bottom = (float(value) for value in frame)
    return left <= px[0] <= right and top <= px[1] <= bottom


def _position_problem(px, extraction, spec):
    if not _inside_plot(px, extraction):
        return "position is outside calibrated plot frame"
    if any(float(left) <= px[0] <= float(right) and float(top) <= px[1] <= float(bottom)
           for left, top, right, bottom in spec.get("mask_bboxes") or []):
        return "position is inside an excluded source region"
    return None


def _marker_radius(series, extraction):
    """Best available native marker radius in panel pixels.

    Stage edits operate in panel pixels, so the series' measured radius is a
    better collision scale than an axis-unit tolerance.  The conservative
    fallback keeps hand-authored/legacy extractions safe when the field is
    absent without making the check effectively global.
    """
    try:
        value = series.get("marker_radius_px")
        if value is None:
            value = (extraction.get("calibration") or {}).get("marker_radius_px")
        return max(1.0, float(value)) if value is not None else 4.0
    except (TypeError, ValueError):
        return 4.0


def _same_series_collision(series, px, extraction, *, exclude=None):
    """Find a preserved point too close to a model-authored target pixel.

    A point inside one marker radius is treated as the same physical marker.
    Existing points always win; callers can therefore reject an edit without
    changing the staged extraction.  ``exclude`` is used for move/reassign so
    a point is not considered to collide with itself.
    """
    target_radius = _marker_radius(series, extraction)
    for existing in series.get("points", []):
        if existing is exclude:
            continue
        existing_px = existing.get("px")
        if not existing_px or existing_px[0] is None or existing_px[1] is None:
            continue
        try:
            distance = math.hypot(float(px[0]) - float(existing_px[0]), float(px[1]) - float(existing_px[1]))
        except (TypeError, ValueError):
            continue
        existing_radius = _marker_radius(series, extraction)
        collision_radius = max(target_radius, existing_radius)
        if distance <= collision_radius:
            return {
                "point_id": existing.get("point_id"),
                "source_pixel": [round(float(existing_px[0]), 2), round(float(existing_px[1]), 2)],
                "distance_px": round(float(distance), 3),
                "collision_radius_px": round(float(collision_radius), 3),
                "rule": "same_series_within_marker_radius",
            }
    return None


def _collision_audit(record, collision):
    return {
        **record,
        "status": "not_applied",
        "detail": "same_series_marker_collision",
        "collision_prevented": True,
        "collision": collision,
    }


def _positional_support(operation):
    """Require native support and a positive uncertainty for a changed position."""
    kind = operation.get("evidence_kind")
    source = operation.get("source_evidence")
    reference = operation.get("evidence_ref")
    if not all(isinstance(value, str) for value in (kind, source, reference)):
        return None
    source, reference = source.strip(), reference.strip()
    try:
        raw_uncertainty = operation.get("uncertainty_px")
        if isinstance(raw_uncertainty, bool) or not isinstance(raw_uncertainty, (int, float)):
            return None
        uncertainty = float(raw_uncertainty)
    except (TypeError, ValueError):
        return None
    if (kind not in {"native_visible", "partial_marker"} or not source or not reference
            or not math.isfinite(uncertainty) or uncertainty <= 0):
        return None
    return kind, source, reference, uncertainty


def _nonpositional_support(operation, action):
    """Validate evidence citations for operations that do not change a position."""
    kind = operation.get("evidence_kind")
    source = operation.get("source_evidence")
    reference = operation.get("evidence_ref")
    if not all(isinstance(value, str) for value in (kind, source, reference)):
        return None
    source, reference = source.strip(), reference.strip()
    allowed = {"native_visible", "partial_marker", "native_absence"}
    if action == "reassign":
        allowed.discard("native_absence")
    if kind not in allowed or not source or not reference or operation.get("uncertainty_px") is not None:
        return None
    return kind, source, reference


def _audit_uncertainty(value):
    """Keep invalid direct-call values inspectable without writing non-JSON NaN."""
    if isinstance(value, bool) or value is None:
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return numeric if math.isfinite(numeric) else str(value)


def _operation_evidence(operation):
    """Copy operation evidence without erasing details from legacy point records."""
    values = {}
    for key in ("evidence_kind", "uncertainty_px", "evidence_ref", "source_evidence"):
        if key in operation and not (key == "uncertainty_px" and operation[key] is None):
            values[key] = operation[key]
    return values


def _apply_point_evidence(point, operation, series, extraction):
    point.update(_operation_evidence(operation))
    if operation.get("evidence_kind") == "partial_marker":
        score = _added_point_confidence("partial_marker", float(operation["uncertainty_px"]), series, extraction)
        try:
            point["confidence"] = min(float(point["confidence"]), score)
        except (KeyError, TypeError, ValueError):
            point["confidence"] = score


def _added_point_confidence(evidence_kind, uncertainty_px, series, extraction):
    """Conservative confidence from the cited positional uncertainty and marker scale.

    This is only the legacy display score used by downstream plots. The native
    evidence kind and raw uncertainty remain attached to the point and export.
    """
    marker_radius = _marker_radius(series, extraction)
    score = min(0.79, 1.0 / (1.0 + uncertainty_px / marker_radius))
    if evidence_kind == "partial_marker":
        score = min(score, 0.64)
    return round(max(0.25, score), 3)


def _recalculate(extraction):
    calibration = extraction.get("calibration") or {}
    for series in extraction.get("series", []):
        for point in series.get("points", []):
            px = point.get("px")
            if not px or px[0] is None or px[1] is None:
                continue
            point["x"] = round(float(pixel_to_axis(calibration, "x", float(px[0]))), 4)
            point["y"] = round(float(pixel_to_axis(calibration, "y", float(px[1]))), 4)
        series["points"].sort(
            key=lambda point: (
                math.inf if point.get("x") is None else float(point["x"]),
                math.inf if point.get("y") is None else float(point["y"]),
            )
        )
        series["n_points"] = len(series["points"])
        xs = [point["x"] for point in series["points"] if point.get("x") is not None]
        ys = [point["y"] for point in series["points"] if point.get("y") is not None]
        series["x_min"], series["x_max"] = (min(xs), max(xs)) if xs else (None, None)
        series["y_min"], series["y_max"] = (min(ys), max(ys)) if ys else (None, None)
    return extraction


def apply_agent_edits(extraction, spec, response, actor):
    """Apply schema-validated model operations and return extraction plus audit."""
    result = copy.deepcopy(extraction)
    bbox = spec.get("panel_bbox") or [0, 0, 0, 0]
    width, height = float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])
    audit = []

    for index, operation in enumerate((response or {}).get("operations") or []):
        action = str(operation.get("action") or "")
        point_id = str(operation.get("point_id") or "")
        target_label = naming.series_label(operation.get("series_label"))
        points = _point_index(result)
        series_by_label = _series_index(result)
        target_series = series_by_label.get(target_label)
        source = points.get(point_id)
        record = {
            "operation_index": index,
            "operation_id": operation.get("operation_id") or f"{actor}-{index + 1}",
            "action": action,
            "point_id": point_id or None,
            "series_label": target_label or None,
            "reason": operation.get("reason", ""),
            "source_evidence": operation.get("source_evidence", ""),
            "evidence_kind": operation.get("evidence_kind"),
            "uncertainty_px": _audit_uncertainty(operation.get("uncertainty_px")),
            "evidence_ref": operation.get("evidence_ref"),
            "actor": actor,
        }

        if target_series is not None and not _eligible_series(target_series):
            audit.append(
                {
                    **record,
                    "status": "not_applied",
                    "detail": "target series is excluded by CO2 policy",
                }
            )
            continue

        has_position = action in {"add", "move"} or (
            action == "reassign"
            and operation.get("x_norm") is not None
            and operation.get("y_norm") is not None
        )
        support = _positional_support(operation) if has_position else None
        if action in {"add", "move", "reassign"}:
            valid_support = support if has_position else _nonpositional_support(operation, action)
            if valid_support is None:
                audit.append({
                    **record,
                    "status": "not_applied",
                    "detail": "operation requires native evidence, an evidence reference, and valid positional uncertainty",
                })
                continue

        if action == "add":
            px = _pixel(operation, width, height)
            if target_series is None or px is None:
                audit.append(
                    {
                        **record,
                        "status": "not_applied",
                        "detail": "missing target series or position",
                    }
                )
                continue
            position_problem = _position_problem(px, result, spec)
            if position_problem:
                audit.append(
                    {
                        **record,
                        "status": "not_applied",
                        "detail": position_problem,
                    }
                )
                continue
            collision = _same_series_collision(target_series, px, result)
            if collision is not None:
                audit.append(_collision_audit(record, collision))
                continue
            evidence_kind, source_evidence, evidence_ref, uncertainty_px = support
            target_series.setdefault("points", []).append(
                {
                    "x": None,
                    "y": None,
                    "px": px,
                    "confidence": _added_point_confidence(evidence_kind, uncertainty_px, target_series, result),
                    "overlap_flag": bool(operation.get("overlap", False)),
                    "assigned_by": actor,
                    "source": actor,
                    "evidence_kind": evidence_kind,
                    "uncertainty_px": uncertainty_px,
                    "evidence_ref": evidence_ref,
                    "source_evidence": source_evidence,
                }
            )
            audit.append({**record, "status": "applied", "source_pixel": px})
            continue

        if source is None:
            audit.append(
                {**record, "status": "not_applied", "detail": "unknown point_id"}
            )
            continue
        source_series, point = source

        if action == "delete":
            source_series["points"].remove(point)
            audit.append({**record, "status": "applied"})
        elif action == "move":
            px = _pixel(operation, width, height)
            if px is None:
                audit.append(
                    {**record, "status": "not_applied", "detail": "missing position"}
                )
                continue
            position_problem = _position_problem(px, result, spec)
            if position_problem:
                audit.append(
                    {
                        **record,
                        "status": "not_applied",
                        "detail": position_problem,
                    }
                )
                continue
            collision = _same_series_collision(source_series, px, result, exclude=point)
            if collision is not None:
                audit.append(_collision_audit(record, collision))
                continue
            point["px"] = px
            point["assigned_by"] = actor
            point["source"] = actor
            _apply_point_evidence(point, operation, source_series, result)
            audit.append({**record, "status": "applied", "source_pixel": px})
        elif action == "reassign":
            if target_series is None:
                audit.append(
                    {
                        **record,
                        "status": "not_applied",
                        "detail": "unknown target series",
                    }
                )
                continue
            px = _pixel(operation, width, height)
            position_problem = _position_problem(px, result, spec) if px is not None else None
            if position_problem:
                audit.append(
                    {
                        **record,
                        "status": "not_applied",
                        "detail": position_problem,
                    }
                )
                continue
            target_px = px if px is not None else point.get("px")
            if target_px is not None:
                collision = _same_series_collision(target_series, target_px, result, exclude=point if source_series is target_series else None)
                if collision is not None:
                    audit.append(_collision_audit(record, collision))
                    continue
            if source_series is not target_series:
                source_series["points"].remove(point)
                target_series.setdefault("points", []).append(point)
            if px is not None:
                point["px"] = px
            if has_position:
                _apply_point_evidence(point, operation, target_series, result)
            else:
                point.update(_operation_evidence(operation))
            point["assigned_by"] = actor
            point["source"] = actor
            audit.append(
                {**record, "status": "applied", "source_pixel": point.get("px")}
            )
        else:
            audit.append(
                {**record, "status": "not_applied", "detail": "unsupported action"}
            )

    result = _recalculate(result)
    return result, {
        "schema_version": "1.0",
        "actor": actor,
        "requested": len((response or {}).get("operations") or []),
        "applied": sum(item["status"] == "applied" for item in audit),
        "operations": audit,
    }

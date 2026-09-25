"""Traceable source preservation and panel-crop generation for Agent 01."""

import hashlib
import pathlib
import shutil

import cv2


def source_integrity(spec):
    """Record the exact source image used by all extraction stages."""
    path = pathlib.Path(str(spec.get("image") or ""))
    if not path.is_file():
        return {"path": str(path), "exists": False, "sha256": "", "width": None, "height": None}
    image = cv2.imread(str(path))
    return {
        "path": str(path.resolve()), "exists": True,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "width": int(image.shape[1]) if image is not None else None,
        "height": int(image.shape[0]) if image is not None else None,
    }


def preserve_source(source_path, evidence_dir):
    source = pathlib.Path(source_path)
    evidence_dir = pathlib.Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    preserved = evidence_dir / f"source_original{source.suffix.lower()}"
    shutil.copy2(source, preserved)
    return {
        "input_path": str(source.resolve()),
        "preserved_path": str(preserved.resolve()),
        "sha256": hashlib.sha256(preserved.read_bytes()).hexdigest(),
    }


def crop_panel(source_path, bbox_norm, out_path, panel_id, pad=0.015):
    image = cv2.imread(str(source_path))
    if image is None:
        raise RuntimeError(f"could not read source image: {source_path}")
    height, width = image.shape[:2]
    x0, y0, x1, y1 = bbox_norm
    box = [
        max(0, int((x0 - pad) * width)),
        max(0, int((y0 - pad) * height)),
        min(width, int((x1 + pad) * width)),
        min(height, int((y1 + pad) * height)),
    ]
    px0, py0, px1, py1 = box
    crop = image[py0:py1, px0:px1]
    if crop.size == 0:
        raise RuntimeError(f"empty crop for {panel_id}: {box}")
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), crop):
        raise RuntimeError(f"could not write panel crop: {out_path}")
    return crop, {
        "panel_id": panel_id,
        "source_path": str(pathlib.Path(source_path).resolve()),
        "crop_path": str(out_path.resolve()),
        "source_size_px": [width, height],
        "bbox_norm": [float(v) for v in bbox_norm],
        "bbox_px": box,
        "crop_size_px": [crop.shape[1], crop.shape[0]],
        "padding_fraction": pad,
    }

"""Pack the complete panel evidence into a Code Interpreter input archive."""

import io
import csv
import json
import pathlib
import zipfile

from src.settings import ROOT
from src.tools.stage_artifacts import STAGE_COLUMNS


def csv_text(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=STAGE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def build_bundle(panel_dir, stage, *, schema, candidate=None, rows=None):
    """Include originals, every native tile and the full ledger, without image downsizing."""
    panel_dir = pathlib.Path(panel_dir)
    names = ["panel.png", "spec.json", "agent02_check_extraction.json", "python_summary.json",
             "python_points.csv", "python_points.json", "python_metadata.json", "python_hard_checks.json",
             "python_overlay.png", "python_recreated.png"]
    if stage == "agent04":
        names += ["agent03_points.csv", "agent03_points.json", "agent03_metadata.json",
                  "agent03_review_points.json", "agent03_edit_audit.json", "agent03_hard_checks.json",
                  "agent03_overlay.png", "agent03_recreated.png", "agent03_redraw_diff.png"]
    required = ["panel.png", "spec.json", "python_points.csv", "python_points.json"]
    if stage == "agent04":
        required += ["agent03_points.csv", "agent03_points.json", "agent03_review_points.json"]
    for name in required:
        if not (panel_dir / name).is_file():
            raise FileNotFoundError(f"Missing file review input: {name}")
    directory = panel_dir / f"{stage}_unstructured"
    directory.mkdir(exist_ok=True)
    path = directory / f"{stage}_inputs.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            file = panel_dir / name
            if file.is_file():
                archive.write(file, name)
        evidence = panel_dir / "adjudicate"
        if evidence.is_dir():
            for file in sorted(evidence.rglob("*")):
                if file.is_file() and file.suffix.lower() in {".png", ".json"}:
                    # Do not follow a link into another panel or outside the workspace.
                    file.resolve().relative_to(panel_dir.resolve())
                    archive.write(file, file.relative_to(panel_dir).as_posix())
        archive.writestr("response.schema.json", json.dumps(schema, indent=2))
        archive.write(ROOT / "schemas/points_csv.contract.json", "points_csv.contract.json")
        archive.write(ROOT / "examples/points.example.csv", "points.example.csv")
        if candidate is not None:
            archive.writestr("review_points.json", json.dumps(candidate, indent=1))
        if rows is not None:
            archive.writestr("review_points.csv", csv_text(rows))
    return path

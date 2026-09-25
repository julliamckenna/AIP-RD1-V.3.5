"""The naming convention (docs/naming.md) holds across the repository. No Azure access needed."""

import json
import os
import pathlib
import re
import pytest

from src.models import naming
from src.settings import ROOT
from src.workflow.resources import load_json, load_workflow

TEXT_SUFFIXES = (".py", ".md", ".yaml", ".yml", ".json", ".css", ".toml", ".txt", ".ps1", ".sh", ".csv")
ROOT_FILES = {".env.example", ".gitignore", "README.md", "config.yaml",
              "main.py", "pyproject.toml", "requirements.txt", "workflow.yaml"}
LOCAL_ONLY_FILES = {".env"}  # your own settings; allowed on disk, never part of the handed-over code
# folders that hold local environments, caches, run outputs or input papers - not project files
SKIP_DIRS = {".git", ".venv", "venv", ".azcli", "__pycache__", ".pytest_cache", "artifacts", "cases", ".idea"}


def project_files():
    """Every project file below the root (relative paths), without local environments, caches and run outputs."""
    files = []
    for folder, dirs, names in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        relative_folder = pathlib.Path(folder).relative_to(ROOT)
        for name in sorted(names):
            relative = (relative_folder / name).as_posix()
            if relative.startswith("data/") and name != ".keep":  # input papers
                continue
            files.append(relative)
    return files


def project_text():
    for path in project_files():
        if path.endswith(TEXT_SUFFIXES) and not path.startswith("evaluations/report/"):
            yield path, (ROOT / path).read_text(encoding="utf-8", errors="ignore")


BRITISH = "colo" + "ur"  # spelled in two parts so this file does not flag itself


def test_american_spelling_everywhere():
    offenders = [path for path, text in project_text() if re.search(BRITISH, text, re.I)]
    assert not offenders, f"use 'color' (docs/naming.md 6): {offenders}"


def test_stage_files_are_named_after_their_step():
    # old stage file names; <document>/final.csv is the deliberate user-facing name and stays
    old_names = re.compile(r"\bcandidate(\.csv|\.json|_[a-z_]+\.(json|png|csv))\b"
                           r"|\bfinal(\.json|_[a-z_]+\.(json|png|csv))\b|\bpython\.(csv|json)\b")
    offenders = [path for path, text in project_text()
                 if path.startswith(("src/", "scripts/", "docs/", "README.md")) and old_names.search(text)]
    assert not offenders, f"stage files are <stage>_<part> (docs/naming.md 2): {offenders}"
    assert naming.STAGES == ("python", "agent03", "agent04")
    assert naming.stage_file("agent04", "points.csv") == "agent04_points.csv"
    with pytest.raises(ValueError):
        naming.stage_file("final", "points.csv")


def test_every_step_id_follows_the_pattern():
    for step_id in load_workflow().steps:
        assert re.fullmatch(r"(agent\d\d|python)_[a-z]+_[a-z]+", step_id), step_id


def test_each_agent_step_has_pinned_prompt_agent_configuration():
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    for step in load_workflow().steps.values():
        if not step.agent:
            continue
        number = step.id[len("agent"):len("agent") + 2]
        assert step.agent.name.startswith(f"agent{number}-"), step.id
        assert step.agent.version, step.id
        assert step.agent.name_env == f"FOUNDRY_AGENT_{number}_NAME", step.id
        assert step.agent.version_env == f"FOUNDRY_AGENT_{number}_VERSION", step.id
        assert f"\n{step.agent.name_env}=" in example, f"{step.agent.name_env} missing from .env.example"
        assert f"\n{step.agent.version_env}=" in example, f"{step.agent.version_env} missing from .env.example"


def test_schema_keys_are_snake_case():
    def keys(node, path="$"):
        if isinstance(node, dict):
            for key, value in node.get("properties", {}).items():
                yield key, f"{path}.{key}"
                yield from keys(value, f"{path}.{key}")
            for key, value in node.items():
                if key != "properties":
                    yield from keys(value, path)
        elif isinstance(node, list):
            for item in node:
                yield from keys(item, path)
    for schema in sorted((ROOT / "schemas").glob("*.schema.json")):
        bad = [where for key, where in keys(json.loads(schema.read_text())) if not re.fullmatch(r"[a-z][a-z0-9_]*", key)]
        assert not bad, f"{schema.name}: {bad}"


def test_examples_use_only_the_shared_vocabulary():
    defs = load_json("schemas/shared.schema.json")["$defs"]
    allowed = {name: set(defs[name]["enum"]) for name in ("marker", "marker_fill", "line_style", "verdict")}

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in allowed and isinstance(value, str):
                    assert value in allowed[key], f"{key}={value!r} is not in the shared vocabulary"
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
    for example in sorted((ROOT / "examples").glob("*.example.json")):
        walk(json.loads(example.read_text()))


@pytest.mark.parametrize("raw, canonical", [
    ("CO2 273 K", "CO2 273K"), ("CO₂  273K", "CO2 273K"), ("CO2 273K ads", "CO2 273K ads"), ("N2 77.5 K", "N2 77.5K"),
])
def test_series_labels_normalise(raw, canonical):
    assert naming.series_label(raw) == canonical


def test_panel_ids_are_lowercase_letters():
    assert [naming.panel_id(value) for value in ("A", "b", "panel_C", "", None)] == ["a", "b", "c", "x", "x"]


def test_repository_root_holds_only_project_files():
    extra = {path for path in project_files() if "/" not in path} - ROOT_FILES - LOCAL_ONLY_FILES
    assert not extra, f"move these into a folder (docs/naming.md 8): {sorted(extra)}"

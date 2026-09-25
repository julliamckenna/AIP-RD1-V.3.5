"""Contract tests: workflow.yaml, schemas, examples and prompts stay in sync. No Azure access needed.

    python -m pytest -q
"""

import csv
import json
import pathlib

import pytest
from jsonschema import Draft202012Validator

from src.models import naming
from src.settings import ROOT
from src.workflow.build import STEP_CLASSES, build_workflow, mermaid
from src.workflow.resources import load_json, load_schema, load_workflow, prompt_placeholders
from src.workflow.steps.agent00_find_figures import panels_for_figure

CONFIG = load_workflow()
AGENT_STEPS = [step for step in CONFIG.steps.values() if step.agent]

# Values each step fills at runtime (besides $example, which agents.py always supplies).
RUNTIME_VALUES = {
    "agent00_find_figures": {"candidate_id", "document_context", "caption", "references", "co2_scope"},
    "agent01_read_chart": {"discovery_context", "co2_scope"},
    "agent02_check_extraction": {"round", "max_rounds", "image_list", "current_structure", "extraction_status", "python_result"},
    "agent03_review_points": {"figure_id", "panel_id", "runtime_files"},
    "agent04_final_check": {"figure_id", "panel_id", "runtime_files"},
}


def test_workflow_builds_and_matches_step_classes():
    assert set(CONFIG.steps) == set(STEP_CLASSES)
    build_workflow()
    assert "agent02_check_extraction -- adjust --> python_extract_points" in mermaid()
    assert "agent01_read_chart --> python_extract_points" in mermaid()


def test_workflow_allows_many_panels_per_document():
    # every panel passes ~10 workflow steps; the framework default of 100 stopped documents after ~8 panels
    assert CONFIG.max_workflow_steps >= 500
    assert build_workflow().max_iterations == CONFIG.max_workflow_steps


@pytest.mark.parametrize("step", AGENT_STEPS, ids=lambda step: step.id)
def test_agent_files_follow_the_step_naming_rule(step):
    agent = step.agent
    assert agent.prompt == f"prompts/{step.id}.md"
    assert agent.schema == f"schemas/{step.id}.schema.json"
    assert agent.example == f"examples/{step.id}.example.json"
    assert agent.name.startswith(step.id.split("_")[0] + "-")
    assert agent.version
    assert agent.name_env == f"FOUNDRY_AGENT_{step.id[5:7]}_NAME"
    assert agent.version_env == f"FOUNDRY_AGENT_{step.id[5:7]}_VERSION"


def _assert_strict(node, path="<root>"):
    """OpenAI strict structured outputs: closed objects, every property required, no leftover $ref."""
    if isinstance(node, dict):
        assert "$ref" not in node, f"unresolved $ref at {path}"
        if node.get("type") == "object" or "properties" in node:
            assert node.get("additionalProperties") is False, f"{path} must set additionalProperties: false"
            assert set(node.get("required", [])) == set(node.get("properties", {})), f"{path} must require every property"
        for key, value in node.items():
            _assert_strict(value, f"{path}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _assert_strict(value, f"{path}/{index}")


@pytest.mark.parametrize("step", AGENT_STEPS, ids=lambda step: step.id)
def test_schema_is_strict_and_example_is_valid(step):
    schema = load_schema(step.agent.schema)
    Draft202012Validator.check_schema(schema)
    if step.agent.response_mode == "json":
        _assert_strict(schema)
    errors = list(Draft202012Validator(schema).iter_errors(load_json(step.agent.example)))
    assert not errors, [error.message for error in errors]


@pytest.mark.parametrize("step", AGENT_STEPS, ids=lambda step: step.id)
def test_prompt_placeholders_are_all_supplied(step):
    # agents that name markers, series or verdicts get the shared vocabulary (docs/naming.md)
    names_things = step.id != "agent00_find_figures"
    example_keys = {"example"} if step.agent.response_mode == "json" else set()
    assert prompt_placeholders(step.agent.prompt) == RUNTIME_VALUES[step.id] | example_keys | ({"vocabulary"} if names_things else set())


def test_every_schema_prompt_and_example_file_is_used():
    used = {path for step in AGENT_STEPS for path in (step.agent.prompt, step.agent.schema, step.agent.example)}
    shared = {"schemas/shared.schema.json", "schemas/co2_scope.policy.json", "schemas/points_csv.contract.json",
              "examples/points.example.csv"}
    on_disk = {
        str(p.relative_to(ROOT)).replace("\\", "/")
        for folder in ("prompts", "schemas", "examples")
        for p in (ROOT / folder).iterdir()
    }
    assert on_disk == used | shared


def test_points_example_matches_csv_contract():
    with open(ROOT / "examples/points.example.csv", newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header == load_json("schemas/points_csv.contract.json")["core_columns"]


def test_figure_folders_are_named_by_figure_id():
    figure = {"page": 4, "figure_number": None, "file": "x.png"}
    discovery = json.loads((ROOT / "examples/agent00_find_figures.example.json").read_text())
    discovery["panels"] = [dict(discovery["panels"][0], panel=letter) for letter in ("a", "b")]
    used = set()
    first = panels_for_figure(figure, discovery, used)
    again = panels_for_figure(figure, dict(discovery, panels=discovery["panels"][:1]), used)
    assert [p.folder for p in first] == ["fig10a", "fig10b"]
    assert [p.folder for p in again] == ["fig10a_2"]
    assert naming.figure_id(None, 7, "x") == "figp7"
    assert naming.figure_id("2", 1, "x") == "fig2"
    assert pathlib.Path(first[0].folder).name == first[0].figure_id

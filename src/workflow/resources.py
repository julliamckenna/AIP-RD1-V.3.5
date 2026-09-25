"""Load the workflow's YAML/JSON resources: workflow.yaml, schemas, examples and prompt templates.

Every path is relative to the project root, exactly as written in workflow.yaml.
Response schemas may reference shared definitions with
``{"$ref": "shared.schema.json#/$defs/<name>"}``; ``load_schema`` inlines them so the
model receives one self-contained strict JSON schema.
"""

from __future__ import annotations

import copy
import json
import string
from dataclasses import dataclass, field
from functools import lru_cache

import yaml

from src.settings import ROOT

SCHEMA_DIR = ROOT / "schemas"


@lru_cache(maxsize=None)
def load_json(relative_path: str):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def load_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _resolve_refs(node, *, current_file: str):
    """Inline every ``$ref`` (local ``#/$defs/x`` or ``other.schema.json#/$defs/x``)."""
    if isinstance(node, list):
        return [_resolve_refs(item, current_file=current_file) for item in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        target_file, _, pointer = node["$ref"].partition("#")
        target_file = f"schemas/{target_file}" if target_file else current_file
        target = load_json(target_file)
        for part in filter(None, pointer.split("/")):
            target = target[part]
        resolved = _resolve_refs(copy.deepcopy(target), current_file=target_file)
        extra = {key: value for key, value in node.items() if key != "$ref"}
        return {**resolved, **_resolve_refs(extra, current_file=current_file)}
    return {
        key: _resolve_refs(value, current_file=current_file)
        for key, value in node.items()
        if key not in ("$schema", "$id", "$defs", "$comment")
    }


@lru_cache(maxsize=None)
def _load_schema_cached(relative_path: str) -> str:
    return json.dumps(_resolve_refs(load_json(relative_path), current_file=relative_path))


def load_schema(relative_path: str) -> dict:
    """Return a self-contained response schema (shared ``$ref``s inlined)."""
    return json.loads(_load_schema_cached(relative_path))


def _as_json(value) -> str:
    """JSON for a prompt: a list of flat records (point rows) is one compact record per line."""
    if isinstance(value, list) and value and all(
        isinstance(item, dict) and not any(isinstance(v, (dict, list)) for v in item.values()) for item in value
    ):
        lines = [json.dumps(item, ensure_ascii=False, default=float, separators=(",", ":")) for item in value]
        return "[\n" + ",\n".join(lines) + "\n]"
    return json.dumps(value, indent=1, ensure_ascii=False, default=float)


VOCABULARY_DEFS = ("marker", "marker_fill", "line_style", "axis_scale", "branches", "verdict")


def vocabulary() -> str:
    """The exact names agents must use, read from schemas/shared.schema.json (docs/naming.md)."""
    defs = load_json("schemas/shared.schema.json")["$defs"]
    lines = [f"- `{name}`: " + ", ".join(f"`{value}`" for value in defs[name]["enum"]) for name in VOCABULARY_DEFS]
    lines.append("- series labels: `<gas> <temperature>K[ <branch>]` exactly as the legend reads, e.g. `CO2 273K`, "
                 "`CO2 273K ads` - no space before K, `CO2` not `CO₂`")
    return "\n".join(lines)


def render_prompt(relative_path: str, values: dict) -> str:
    """Fill ``$placeholders`` in a prompt template. Non-string values are rendered as JSON."""
    rendered = {key: value if isinstance(value, str) else _as_json(value) for key, value in values.items()}
    return string.Template(load_text(relative_path)).substitute(rendered)


def prompt_placeholders(relative_path: str) -> set[str]:
    """Names used as ``$placeholder`` in a prompt template (used by the contract tests)."""
    template = string.Template(load_text(relative_path))
    return {
        match.group("named") or match.group("braced")
        for match in template.pattern.finditer(template.template)
        if match.group("named") or match.group("braced")
    }


#  workflow.yaml


@dataclass(frozen=True)
class AgentConfig:
    name: str
    instructions: str
    prompt: str
    schema: str
    example: str
    version: str = "1"
    name_env: str | None = None
    version_env: str | None = None
    timeout_seconds: int = 500
    response_mode: str = "json"


@dataclass(frozen=True)
class StepConfig:
    id: str
    title: str
    agent: AgentConfig | None = None


@dataclass(frozen=True)
class EdgeConfig:
    source: str
    target: str
    when: str | None = None


@dataclass(frozen=True)
class WorkflowConfig:
    name: str
    description: str
    start: str
    output: str
    steps: dict[str, StepConfig] = field(default_factory=dict)
    edges: list[EdgeConfig] = field(default_factory=list)
    max_image_side: int = 1600
    max_check_rounds: int = 3
    max_workflow_steps: int = 1000


@lru_cache(maxsize=1)
def load_workflow() -> WorkflowConfig:
    raw = yaml.safe_load(load_text("workflow.yaml"))
    agent_defaults = raw.get("agent_defaults") or {}
    steps = {}
    for step in raw["steps"]:
        agent = step.get("agent")
        steps[step["id"]] = StepConfig(
            id=step["id"],
            title=step["title"],
            agent=AgentConfig(**{**agent_defaults, **agent}) if agent else None,
        )
    edges = [EdgeConfig(source=e["from"], target=e["to"], when=e.get("when")) for e in raw["edges"]]
    return WorkflowConfig(
        name=raw["name"],
        description=raw["description"],
        start=raw["start"],
        output=raw["output"],
        steps=steps,
        edges=edges,
        max_image_side=int(raw.get("max_image_side", 1600)),
        max_check_rounds=int(raw.get("max_check_rounds", 3)),
        max_workflow_steps=int(raw.get("max_workflow_steps", 1000)),
    )

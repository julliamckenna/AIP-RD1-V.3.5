import json
import pathlib
import shutil
import uuid

import pytest

from src.workflow import agents
from src.settings import ROOT
from src.workflow.resources import load_json, load_schema, load_workflow
from src.workflow.steps.agent00_find_figures import (
    load_discovery_checkpoint,
    request_fingerprint,
    save_discovery_checkpoint,
)


def test_auth_mode_defaults_to_entra(monkeypatch):
    monkeypatch.delenv(agents.AUTH_MODE_ENV, raising=False)
    assert agents._auth_mode() == "entra"


def test_auth_mode_normalizes_supported_aliases(monkeypatch):
    monkeypatch.setenv(agents.AUTH_MODE_ENV, "entra-id")
    assert agents._auth_mode() == "entra"
    monkeypatch.setenv(agents.AUTH_MODE_ENV, "key")
    assert agents._auth_mode() == "api_key"


def test_auth_mode_rejects_silent_fallback(monkeypatch):
    monkeypatch.setenv(agents.AUTH_MODE_ENV, "unsupported")
    with pytest.raises(RuntimeError, match="use 'entra' or 'api_key'"):
        agents._auth_mode()


def test_local_prompt_contains_the_schema_contract():
    schema = load_schema("schemas/agent00_find_figures.schema.json")
    config = load_workflow().steps["agent00_find_figures"].agent
    prompt = agents.StepAgent(config).prompt({
        "candidate_id": "figure-1",
        "document_context": "Title: test",
        "caption": "Fig. 1: test",
        "references": "none found",
        "co2_scope": load_json("schemas/co2_scope.policy.json"),
    })

    assert "Return exactly one JSON object" in prompt
    assert json.dumps(schema, indent=2, ensure_ascii=False) in prompt


def test_agent00_timeout_is_shorter_than_other_prompt_agent_calls():
    config = load_workflow()
    assert config.steps["agent00_find_figures"].agent.timeout_seconds == 180
    assert config.steps["agent01_read_chart"].agent.timeout_seconds == 500


@pytest.fixture
def workspace_tmp_path():
    path = ROOT / "artifacts" / "test-tmp" / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def test_discovery_checkpoint_round_trip_is_atomic(workspace_tmp_path):
    path = workspace_tmp_path / "agent00_find_figures.json"
    entries = [{"candidate_id": "xref:1", "input_fingerprint": "abc", "answer": {"ok": True}}]

    save_discovery_checkpoint(path, entries)

    assert load_discovery_checkpoint(path) == entries
    assert not pathlib.Path(f"{path}.tmp").exists()


def test_discovery_request_fingerprint_covers_agent_prompt_and_images(workspace_tmp_path):
    image = workspace_tmp_path / "figure.png"
    image.write_bytes(b"first image")

    class FakeAgent:
        agent_name = "agent00-discovery"
        agent_version = "3"

        @staticmethod
        def prompt(values):
            return json.dumps(values, sort_keys=True)

    first = request_fingerprint(FakeAgent(), {"candidate_id": "xref:1"}, [image])
    image.write_bytes(b"changed image")
    changed_image = request_fingerprint(FakeAgent(), {"candidate_id": "xref:1"}, [image])
    changed_prompt = request_fingerprint(FakeAgent(), {"candidate_id": "xref:2"}, [image])

    assert first != changed_image
    assert changed_image != changed_prompt

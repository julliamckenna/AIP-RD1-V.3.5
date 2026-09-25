import asyncio
import json
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.workflow import agents
from src.workflow.resources import load_json, load_workflow


class FakeResponses:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response):
        self.responses = FakeResponses(response)


def agent00():
    return agents.StepAgent(load_workflow().steps["agent00_find_figures"].agent)


def agent00_values():
    return {
        "candidate_id": "figure-1",
        "document_context": "Title: test",
        "caption": "Fig. 1: test",
        "references": "none found",
        "co2_scope": load_json("schemas/co2_scope.policy.json"),
    }


def test_prompt_agent_request_preserves_contract_and_provenance(monkeypatch):
    example = load_json("examples/agent00_find_figures.example.json")
    response = SimpleNamespace(id="resp_123", usage={"total_tokens": 42}, output_text=json.dumps(example))
    client = FakeClient(response)
    monkeypatch.setattr(agents, "_openai_client", lambda: client)
    monkeypatch.setattr(agents, "_image_input", lambda path: {
        "type": "input_image", "image_url": str(path)
    })

    step = agent00()
    temp_path = Path.cwd() / f".test-prompt-agent-{uuid.uuid4().hex}"
    temp_path.mkdir()
    try:
        answer_path = temp_path / "agent00_answer.json"
        answer = asyncio.run(step.ask(
            agent00_values(),
            images=[temp_path / "first.png", temp_path / "second.png"],
            save_to=answer_path,
        ))

        provenance_path = temp_path / "agent00_answer.provenance.json"
        provenance = json.loads(provenance_path.read_text())
    finally:
        shutil.rmtree(temp_path)

    assert answer == example
    call = client.responses.calls[0]
    assert "model" not in call
    assert "conversation" not in call
    assert "previous_response_id" not in call
    assert "store" not in call
    assert "max_output_tokens" not in call
    assert "text" not in call
    assert call["timeout"] == step.config.timeout_seconds
    assert call["extra_body"] == {
        "agent_reference": {
            "name": "agent00-discovery",
                "version": "6",
            "type": "agent_reference",
        }
    }
    content = call["input"][0]["content"]
    assert content[0]["type"] == "input_text"
    assert step.config.instructions in content[0]["text"]
    assert [item["image_url"] for item in content[1:]] == [
        str(temp_path / "first.png"), str(temp_path / "second.png")
    ]
    assert provenance["agent_name"] == "agent00-discovery"
    assert provenance["agent_version"] == "6"
    assert provenance["response_id"] == "resp_123"


def test_prompt_agent_rounds_are_independent(monkeypatch):
    example = load_json("examples/agent00_find_figures.example.json")
    response = SimpleNamespace(output_text=json.dumps(example), usage={})
    client = FakeClient(response)
    monkeypatch.setattr(agents, "_openai_client", lambda: client)

    step = agent00()
    asyncio.run(step.ask(agent00_values()))
    asyncio.run(step.ask({**agent00_values(), "candidate_id": "figure-2"}))

    assert len(client.responses.calls) == 2
    assert all("conversation" not in call and "previous_response_id" not in call
               for call in client.responses.calls)
    assert [call["extra_body"]["agent_reference"]["version"] for call in client.responses.calls] == ["6", "6"]


def test_empty_or_filtered_prompt_agent_output_fails_cleanly(monkeypatch):
    response = SimpleNamespace(
        output_text="",
        status="incomplete",
        incomplete_details={"reason": "content_filter"},
        output=[],
    )
    client = FakeClient(response)
    monkeypatch.setattr(agents, "_openai_client", lambda: client)

    with pytest.raises(RuntimeError, match="no text output"):
        asyncio.run(agent00().ask(agent00_values()))


def test_non_json_and_schema_invalid_outputs_fail_after_transport(monkeypatch):
    non_json = FakeClient(SimpleNamespace(output_text="not json", usage={}))
    monkeypatch.setattr(agents, "_openai_client", lambda: non_json)
    with pytest.raises(RuntimeError, match="non-JSON"):
        asyncio.run(agent00().ask(agent00_values()))

    invalid = FakeClient(SimpleNamespace(output_text="{}", usage={}))
    monkeypatch.setattr(agents, "_openai_client", lambda: invalid)
    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(agent00().ask(agent00_values()))


def test_agent03_missing_coverage_gets_audited_safe_default(monkeypatch):
    answer_without_coverage = {
        "reasoning": "Reviewed the available evidence.",
        "verdict": "accept",
        "series": [],
        "operations": [],
        "uncertainties": [],
    }
    client = FakeClient(SimpleNamespace(
        id="resp_missing_coverage",
        output_text=json.dumps(answer_without_coverage),
        usage={},
    ))
    monkeypatch.setattr(agents, "_openai_client", lambda: client)
    step = agents.StepAgent(load_workflow().steps["agent03_review_points"].agent)
    values = {
        "figure_id": "fig3",
        "panel_id": "a",
        "runtime_files": [],
        "python_rows": "",
        "python_evidence": {},
        "agent02_check_extraction": {},
        "tile_manifest": {},
    }

    result = asyncio.run(step.ask(values, missing_defaults={"coverage": []}))

    assert result["coverage"] == []
    assert step.last_provenance["normalizations"] == [
        {"action": "missing_top_level_default", "key": "coverage"}
    ]


def test_missing_defaults_do_not_mask_other_schema_errors(monkeypatch):
    client = FakeClient(SimpleNamespace(output_text=json.dumps({"reasoning": "incomplete"}), usage={}))
    monkeypatch.setattr(agents, "_openai_client", lambda: client)
    step = agents.StepAgent(load_workflow().steps["agent03_review_points"].agent)
    values = {
        "figure_id": "fig3",
        "panel_id": "a",
        "runtime_files": [],
        "python_rows": "",
        "python_evidence": {},
        "agent02_check_extraction": {},
        "tile_manifest": {},
    }

    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(step.ask(values, missing_defaults={"coverage": []}))


def test_prompt_agent_request_is_cancelled_without_retry(monkeypatch):
    class BlockingResponses:
        def __init__(self):
            self.calls = 0
            self.started = asyncio.Event()

        async def create(self, **kwargs):
            self.calls += 1
            self.started.set()
            await asyncio.Event().wait()

    responses = BlockingResponses()
    client = SimpleNamespace(responses=responses)
    monkeypatch.setattr(agents, "_openai_client", lambda: client)

    async def cancel_request():
        task = asyncio.create_task(agent00().ask(agent00_values()))
        await responses.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_request())
    assert responses.calls == 1


def test_sdk_level_retries_are_disabled():
    assert agents.OPENAI_MAX_RETRIES == 0

"""File responses: native inputs, free-text transport, downloads and failure receipts."""

import asyncio
import copy
import json
import shutil
import uuid
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import httpx
from openai._legacy_response import HttpxBinaryResponseContent

from src.workflow.agents import StepAgent
from src.workflow.resources import load_workflow
from scripts.sync_prompt_agents import AgentSpec, definition_for


@pytest.fixture
def workspace():
    path = Path.cwd() / f".test-artifacts-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


class Response:
    def __init__(self, payload):
        self.payload = payload
        self.id = payload.get("id")
        self.usage = {"total_tokens": 42}

    def model_dump(self):
        return self.payload


class Client:
    def __init__(self, report, *, status="completed", cite=True):
        self.calls = []
        self.project = Project()
        self.uploads, self.deletes = [], []
        self.files = SimpleNamespace(create=self.upload, delete=self.delete)
        self.responses = SimpleNamespace(create=self.respond)
        self.containers = SimpleNamespace(files=SimpleNamespace(content=SimpleNamespace(retrieve=self.download)))
        annotations = [{"type": "container_file_citation", "filename": "/mnt/data/agent03_review_points.json",
                        "file_id": "out-file", "container_id": "container-1"}] if cite else []
        self.response = Response({"id": "resp-file", "status": status, "output": [
            {"type": "code_interpreter_call", "status": "completed"},
            {"type": "message", "content": [{"type": "output_text", "text": "Here is the report.",
                                                "annotations": annotations}]},
        ]})
        self.data = json.dumps(report).encode()

    async def upload(self, *, file, **kwargs):
        self.uploads.append(file.read())
        return SimpleNamespace(id="input-file")

    async def delete(self, file_id):
        self.deletes.append(file_id)

    async def respond(self, **kwargs):
        # Match Foundry's validation: agent_reference cannot override tools/text.
        assert "tools" not in kwargs and "text" not in kwargs
        self.calls.append(kwargs)
        return self.response

    async def download(self, **kwargs):
        assert kwargs["file_id"] == "out-file"
        assert kwargs["container_id"] == "container-1"

        # The installed async SDK returns this wrapper with a SYNCHRONOUS read().
        return HttpxBinaryResponseContent(httpx.Response(200, content=self.data))


class Project:
    def __init__(self):
        self.created, self.deleted = [], []
        self.agents = SimpleNamespace(get_version=self.get, create_version=self.create, delete=self.delete)

    async def get(self, **kwargs):
        return SimpleNamespace(definition={"kind": "prompt", "model": "pinned-deployment",
            "instructions": "Pinned scientific review instructions",
            "text": {"format": {"type": "text"}},
            "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}]})

    async def create(self, **kwargs):
        self.created.append(kwargs)
        return SimpleNamespace(version="1")

    async def delete(self, **kwargs):
        self.deleted.append(kwargs["agent_name"])


def invoke(workspace, monkeypatch, client=None):
    monkeypatch.delenv("CHART_EXTRACT_DRY_RUN", raising=False)
    step = StepAgent(load_workflow().steps["agent03_review_points"].agent)
    client = client or Client(step.example)
    step._client = client
    step._get_project_client = lambda: client.project
    bundle = workspace / "inputs.zip"
    bundle.write_bytes(b"lossless source and full candidate bundle")
    call = step.ask_artifacts(
        {"figure_id": "fig1", "panel_id": "a", "runtime_files": [bundle.name]},
        files=[bundle], output_dir=workspace / "outputs", expected_files=["agent03_review_points.json"],
        report_name="agent03_review_points.json",
    )
    return step, client, call


def test_unstructured_files_preserve_exact_request_and_validate_downloads(workspace, monkeypatch):
    step, client, call = invoke(workspace, monkeypatch)
    result = asyncio.run(call)
    request = client.calls[0]
    assert "text" not in request and "tools" not in request
    definition = client.project.created[0]["definition"].as_dict()
    assert definition["text"] == {"format": {"type": "text"}}
    assert definition["tools"][0]["container"]["file_ids"] == ["input-file"]
    assert definition["model"] == "pinned-deployment"
    assert request["extra_body"]["agent_reference"]["name"] == client.project.created[0]["agent_name"]
    assert client.project.deleted == [client.project.created[0]["agent_name"]]
    assert "json_schema" not in json.dumps(request)
    assert "Return exactly one JSON object" not in request["input"][0]["content"][0]["text"]
    assert result["report"] == step.example
    assert client.deletes == ["input-file"]
    assert (result["directory"] / "inputs.zip").read_bytes() == client.uploads[0]
    assert (result["directory"] / "response.json").is_file()
    saved = json.loads((result["directory"] / "request.json").read_text())
    assert saved["response_mode"] == "artifacts"
    assert saved["files"][0]["sha256"]


@pytest.mark.parametrize("status,cite,error", [
    ("incomplete", True, "did not complete"),
    ("completed", False, "did not cite required"),
])
def test_incomplete_or_missing_file_response_never_falls_back_to_prose(workspace, monkeypatch, status, cite, error):
    config = load_workflow().steps["agent03_review_points"].agent
    client = Client(StepAgent(config).example, status=status, cite=cite)
    _, _, call = invoke(workspace, monkeypatch, client)
    with pytest.raises(RuntimeError, match=error):
        asyncio.run(call)
    assert len(client.calls) == 1
    assert client.deletes == ["input-file"]
    assert len(list((workspace / "outputs").glob("*/failure.json"))) == 1


def test_malformed_artifact_is_preserved_but_rejected(workspace, monkeypatch):
    _, _, call = invoke(workspace, monkeypatch, Client({"reasoning": "missing properties"}))
    with pytest.raises(ValueError, match="file schema"):
        asyncio.run(call)
    assert len(list((workspace / "outputs").glob("*/agent03_review_points.json"))) == 1


def test_cancelled_artifact_call_does_not_retry(workspace, monkeypatch):
    client = Client({})
    started = asyncio.Event()

    async def wait_forever(**kwargs):
        client.calls.append(kwargs)
        started.set()
        await asyncio.Event().wait()
    client.responses.create = wait_forever
    _, _, call = invoke(workspace, monkeypatch, client)

    async def cancel():
        task = asyncio.create_task(call)
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(cancel())
    assert len(client.calls) == 1
    assert client.deletes == ["input-file"]


def test_published_agent_definitions_match_runtime_modes():
    for number, mode in [("00", "json"), ("03", "artifacts"), ("04", "artifacts")]:
        spec = AgentSpec(number, f"agent{number}", "deployment", "instructions", "report",
                         {"type": "object"}, mode)
        definition = definition_for(spec).as_dict()
        assert definition["text"]["format"]["type"] == ("text" if mode == "artifacts" else "json_schema")
        assert bool(definition.get("tools")) == (mode == "artifacts")


def test_client_cleanup_runs_after_failure_and_empties_caches(monkeypatch):
    from src.workflow import agents
    closed = []

    class Resource:
        def __init__(self, name):
            self.name = name

        async def close(self):
            closed.append(self.name)

    monkeypatch.setattr(agents, "_clients", {("entra", "project"): Resource("responses")})
    monkeypatch.setattr(agents, "_projects", {"project": Resource("project")})
    monkeypatch.setattr(agents, "_credentials", [Resource("credential")])

    async def failure():
        raise ValueError("test failure")
    with pytest.raises(ValueError, match="test failure"):
        asyncio.run(agents.with_client_cleanup(failure()))
    assert closed == ["responses", "project", "credential"]
    assert not agents._clients and not agents._projects and not agents._credentials


def test_explicit_recovery_reuses_identical_result_once(workspace, monkeypatch):
    from src.workflow.artifact_agent import consume_recovery
    step, client, call = invoke(workspace, monkeypatch)
    original = asyncio.run(call)
    directory = original["directory"]
    name = "agent03_review_points.json"
    (directory / "recovered.json").write_text(json.dumps({
        "files": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()},
    }))
    _, second_client, second = invoke(workspace, monkeypatch)
    recovered = asyncio.run(second)
    assert recovered["report"] == step.example
    assert not second_client.calls and not second_client.uploads
    consume_recovery(recovered)
    _, third_client, third = invoke(workspace, monkeypatch)
    asyncio.run(third)
    assert len(third_client.calls) == 1


def test_download_also_supports_async_read_wrapper(workspace):
    from src.workflow.artifact_agent import download_cited_files
    client = Client({"ok": True})

    async def retrieve(**kwargs):
        async def read():
            return b'{"ok": true}'
        return SimpleNamespace(read=read)
    client.containers.files.content.retrieve = retrieve
    asyncio.run(download_cited_files(client, client.response.payload, workspace, ["agent03_review_points.json"]))
    assert json.loads((workspace / "agent03_review_points.json").read_text()) == {"ok": True}

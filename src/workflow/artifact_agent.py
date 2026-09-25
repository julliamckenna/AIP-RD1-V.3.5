"""Unstructured Code Interpreter responses with explicit, validated file outputs.

No response JSON schema is sent to the model. The report inside a cited file
has a local contract; the final message may simply link the deliverables.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import logging
import pathlib
import shutil
import uuid
import zipfile

log = logging.getLogger("chart_extract")


def _save(path, value):
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False), encoding="utf-8")


def _citations(value):
    if isinstance(value, dict):
        if value.get("type") == "container_file_citation":
            yield value
        for child in value.values():
            yield from _citations(child)
    elif isinstance(value, list):
        for child in value:
            yield from _citations(child)


def _file_info(path):
    path = pathlib.Path(path)
    return {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _input_digest(path):
    """Ignore ZIP timestamps, but require identical names and bytes for every member."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = [(item.filename, hashlib.sha256(archive.read(item)).hexdigest())
                       for item in archive.infolist()]
        return hashlib.sha256(json.dumps(sorted(members)).encode()).hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recovered_attempt(directory, request):
    """Reuse only an explicitly recovered, unconsumed result for identical inputs."""
    fields = ("agent_name", "agent_version", "response_mode", "prompt", "images", "expected_files")
    for marker in sorted(directory.parent.glob("attempt-*/recovered.json"),
                         key=lambda path: path.stat().st_mtime, reverse=True):
        previous = marker.parent
        try:
            receipt = json.loads(marker.read_text())
            old = json.loads((previous / "request.json").read_text())
            if receipt.get("consumed") or any(old.get(key) != request.get(key) for key in fields):
                continue
            names = [item["name"] for item in request["files"]]
            if names != [item["name"] for item in old["files"]]:
                continue
            if any(not name or pathlib.Path(name).name != name for name in names):
                continue
            if any(_input_digest(directory / name) != _input_digest(previous / name) for name in names):
                continue
            if any(_file_info(previous / name)["sha256"] != receipt["files"].get(name)
                   for name in request["expected_files"]):
                continue
            return previous
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            continue
    return None


def consume_recovery(result):
    """A completed local stage consumes the one-time download recovery."""
    previous = result.get("recovery_source")
    if previous:
        marker = pathlib.Path(previous) / "recovered.json"
        receipt = json.loads(marker.read_text())
        receipt["consumed"] = True
        _save(marker, receipt)


async def _with_progress(call, name, timeout):
    task = asyncio.create_task(call)
    started = asyncio.get_running_loop().time()
    try:
        while True:
            elapsed = asyncio.get_running_loop().time() - started
            remaining = timeout - elapsed
            if remaining <= 0:
                raise TimeoutError(f"{name}: file review exceeded {timeout}s")
            done, _ = await asyncio.wait({task}, timeout=min(30, remaining))
            if done:
                return await task
            log.info("%s: file review still running (%ds elapsed)", name,
                     round(asyncio.get_running_loop().time() - started))
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


async def download_cited_files(client, payload, directory, expected, timeout=180):
    """Download completed-response citations without issuing another model call."""
    if not isinstance(payload, dict) or payload.get("status") != "completed":
        raise RuntimeError(f"Artifact response did not complete: {payload.get('status') if isinstance(payload, dict) else 'unknown'}")
    if not any(item.get("type") == "code_interpreter_call" for item in payload.get("output", [])):
        raise RuntimeError("Artifact response did not use Code Interpreter")
    citations = {}
    for citation in _citations(payload):
        name = pathlib.PurePosixPath(str(citation.get("filename", "")).replace("\\", "/")).name
        if name not in expected:
            continue
        if not citation.get("file_id") or not citation.get("container_id"):
            raise RuntimeError(f"Incomplete container citation: {name}")
        previous = citations.get(name)
        if previous and (previous["file_id"], previous["container_id"]) != (citation["file_id"], citation["container_id"]):
            raise RuntimeError(f"Ambiguous output citations for {name}")
        citations[name] = citation
    missing = sorted(set(expected) - set(citations))
    if missing:
        raise RuntimeError("Agent did not cite required output files: " + ", ".join(missing))
    for name in expected:
        citation = citations[name]
        content = await client.containers.files.content.retrieve(
            file_id=citation["file_id"], container_id=citation["container_id"],
            timeout=timeout,
        )
        data = content.read()
        if inspect.isawaitable(data):
            data = await data
        if not data:
            raise RuntimeError(f"Downloaded artifact is empty: {name}")
        (directory / name).write_bytes(data)


async def ask_artifacts(agent, values, *, files, images=(), output_dir,
                        expected_files, report_name, dry_run_files=None):
    from src.workflow.agents import _image_input, _jsonable, _provenance, dry_run

    expected = tuple(expected_files)
    if not expected or len(set(expected)) != len(expected) or report_name not in expected:
        raise ValueError("Expected unique output names including the report")
    if any(not name or name in {".", ".."} or "/" in name or "\\" in name for name in expected):
        raise ValueError("Artifact output names must be basenames")
    files, images = tuple(map(pathlib.Path, files)), tuple(map(pathlib.Path, images))
    if len({path.name for path in files}) != len(files):
        raise ValueError("Uploaded basenames must be unique")
    directory = pathlib.Path(output_dir) / f"attempt-{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    if any(path.name in expected or path.name in {"request.json", "response.json", "provenance.json", "failure.json"}
           for path in files):
        raise ValueError("Input and output artifact names must not overlap")
    # Preserve the exact input archive for this attempt, even after a future rerun.
    for path in files:
        shutil.copyfile(path, directory / path.name)
    files = tuple(directory / path.name for path in files)
    text = agent.prompt(values)
    request = {
        "agent_name": agent.agent_name, "agent_version": agent.agent_version,
        "response_mode": "artifacts", "prompt": text,
        "files": [_file_info(path) for path in files],
        "images": [_file_info(path) for path in images],
        "expected_files": list(expected), "timeout_seconds": agent.config.timeout_seconds,
    }
    request_hash = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    _save(directory / "request.json", request)
    previous = None if dry_run() else _recovered_attempt(directory, request)
    if previous:
        report = json.loads((previous / report_name).read_text(encoding="utf-8-sig"))
        agent.validator.validate(report)
        for name in (*expected, "response.json", "runtime_agent.json"):
            if (previous / name).is_file():
                shutil.copyfile(previous / name, directory / name)
        agent.last_provenance = json.loads((previous / "provenance.json").read_text())
        agent.last_provenance.update(recovered_from=str(previous), model_invoked=False,
                                     request_path=str(directory / "request.json"), request_sha256=request_hash)
        _save(directory / "provenance.json", agent.last_provenance)
        log.info("%s: using recovered completed response; no new model call", agent.config.name)
        return {"report": report, "artifacts": {name: directory / name for name in expected},
                "directory": directory, "provenance": agent.last_provenance, "recovery_source": previous}
    uploaded = []
    client = None
    project = None
    runtime_name = None

    async def invoke():
        nonlocal client, project, runtime_name
        from azure.ai.projects.models import PromptAgentDefinition

        client = agent._get_client()
        project = agent._get_project_client()
        base = await project.agents.get_version(agent_name=agent.agent_name, agent_version=agent.agent_version)
        definition = copy.deepcopy(base.definition.as_dict() if hasattr(base.definition, "as_dict") else dict(base.definition))
        if definition.get("kind") != "prompt" or (definition.get("text") or {}).get("format", {}).get("type") != "text":
            raise RuntimeError("The pinned reviewer must use free-text output; sync Agents 03/04 before rerunning")
        code_tools = [tool for tool in definition.get("tools", []) if tool.get("type") == "code_interpreter"]
        if len(code_tools) != 1:
            raise RuntimeError("The pinned reviewer must have one Code Interpreter tool; sync Agents 03/04")
        for path in files:
            with path.open("rb") as handle:
                result = await client.files.create(file=handle, purpose="assistants",
                                                  timeout=agent.config.timeout_seconds)
            uploaded.append(result.id)
        # Foundry forbids tools/text overrides on agent_reference requests.
        # Bind this panel's files in a temporary clone of the pinned definition,
        # exactly as the reference implementation does, then invoke that clone.
        code_tools[0]["container"] = {"type": "auto", "file_ids": uploaded, "memory_limit": "4g"}
        runtime_name = f"chart-review-{uuid.uuid4().hex[:20]}"
        runtime = await project.agents.create_version(
            agent_name=runtime_name, definition=PromptAgentDefinition(definition),
            description="Temporary chart review with this panel's input files",
            metadata={"source_agent": agent.agent_name, "source_version": agent.agent_version,
                      "request_sha256": request_hash},
        )
        runtime_version = str(runtime.version)
        _save(directory / "runtime_agent.json", {
            "name": runtime_name, "version": runtime_version,
            "source_name": agent.agent_name, "source_version": agent.agent_version,
            "definition": definition,
        })
        log.info("%s: invoking Prompt Agent %s version %s (Code Interpreter, complete panel)",
                 agent.config.name, agent.agent_name, agent.agent_version)
        response = await client.responses.create(
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": text},
                *(_image_input(path) for path in images),
            ]}],
            extra_body={"agent_reference": {
                "name": runtime_name, "version": runtime_version,
                "type": "agent_reference",
            }},
            timeout=agent.config.timeout_seconds,
        )
        payload = _jsonable(response)
        _save(directory / "response.json", payload)
        agent.last_provenance = _provenance(response, agent.agent_name, agent.agent_version)
        agent.last_provenance.update(response_mode="artifacts", request_sha256=request_hash,
                                     request_path=str(directory / "request.json"),
                                     runtime_agent_name=runtime_name, runtime_agent_version=runtime_version)
        _save(directory / "provenance.json", agent.last_provenance)
        await download_cited_files(client, payload, directory, expected, agent.config.timeout_seconds)

    try:
        if dry_run():
            if dry_run_files is None or set(dry_run_files) != set(expected):
                raise ValueError("Dry artifact run needs every expected fixture file")
            for name, data in dry_run_files.items():
                (directory / name).write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
            agent.last_provenance = {**request, "prompt": None, "dry_run": True,
                                     "response_id": None, "request_sha256": request_hash}
            _save(directory / "provenance.json", agent.last_provenance)
        else:
            await _with_progress(invoke(), agent.config.name, agent.config.timeout_seconds)
        report = json.loads((directory / report_name).read_text(encoding="utf-8-sig"))
        errors = list(agent.validator.iter_errors(report))
        if errors:
            raise ValueError("Artifact report does not match its file schema: " + "; ".join(
                f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors[:5]))
        return {"report": report, "artifacts": {name: directory / name for name in expected},
                "directory": directory, "provenance": agent.last_provenance}
    except BaseException as error:
        _save(directory / "failure.json", {"error_type": type(error).__name__, "error": str(error),
                                           "request_sha256": request_hash})
        raise
    finally:
        if runtime_name is not None:
            try:
                await asyncio.wait_for(project.agents.delete(agent_name=runtime_name), timeout=10)
            except Exception:
                log.warning("%s: could not remove temporary runtime agent %s", agent.config.name, runtime_name)
        # Uploaded inputs are temporary; saved local request/response/artifacts remain inspectable.
        for file_id in uploaded:
            try:
                await asyncio.wait_for(client.files.delete(file_id), timeout=5)
            except Exception:
                log.warning("%s: could not remove temporary uploaded input %s", agent.config.name, file_id)

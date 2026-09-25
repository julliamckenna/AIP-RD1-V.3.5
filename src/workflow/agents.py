"""The one place that talks to Foundry Prompt Agents.

Each agent step in workflow.yaml references a pinned Foundry Prompt Agent. A call sends one
local prompt (rendered from prompts/<step>.md), the step's ordered images, and local schema
instructions; the answer is validated against the same schema before
the workflow uses it.

Dry run: set ``CHART_EXTRACT_DRY_RUN=1`` to answer every call with the step's example
(examples/<step>.example.json) instead of calling a model.  Use it to wire up and preview
the workflow without Azure access or cost; the extracted data is then meaningless.
"""

from __future__ import annotations

import base64
import copy
import datetime
import json
import logging
import os
import pathlib

import cv2
from jsonschema import Draft202012Validator

from src.workflow.resources import AgentConfig, load_json, load_schema, load_workflow, render_prompt, vocabulary

log = logging.getLogger("chart_extract")

DRY_RUN_ENV = "CHART_EXTRACT_DRY_RUN"
AUTH_MODE_ENV = "FOUNDRY_AUTH_MODE"
PROJECT_API_KEY_ENV = "FOUNDRY_PROJECT_API_KEY"
OPENAI_MAX_RETRIES = 0

_clients: dict[tuple[str, str], object] = {}
_projects: dict[str, object] = {}
_credentials: list[object] = []
# A Prompt Agent call can be long-running.  Retrying inside the SDK is unsafe
# during Ctrl-C shutdown on Windows: the interrupted transport may otherwise
# start another paid request while asyncio is closing the event loop.  Surface
# transient failures to the workflow/user instead of retrying invisibly.

def dry_run() -> bool:
    return os.environ.get(DRY_RUN_ENV, "").strip().lower() in ("1", "true", "yes")


def _auth_mode() -> str:
    """Return the configured Foundry authentication mode."""
    mode = os.environ.get(AUTH_MODE_ENV, "entra").strip().lower().replace("-", "_")
    aliases = {"azure_ad": "entra", "entra_id": "entra", "key": "api_key"}
    mode = aliases.get(mode, mode)
    if mode not in {"entra", "api_key"}:
        raise RuntimeError(
            f"unsupported {AUTH_MODE_ENV}={mode!r}; use 'entra' or 'api_key'"
        )
    return mode


def _project_client():
    """Project CRUD client for binding temporary file-review agents."""
    endpoint = os.environ.get("FOUNDRY_PROJECT_ENDPOINT", "").strip().rstrip("/")
    if not endpoint:
        raise RuntimeError("set FOUNDRY_PROJECT_ENDPOINT to the Foundry project endpoint")
    if endpoint not in _projects:
        from azure.ai.projects.aio import AIProjectClient
        from azure.identity.aio import DefaultAzureCredential

        credential = DefaultAzureCredential()
        _credentials.append(credential)
        _projects[endpoint] = AIProjectClient(endpoint=endpoint, credential=credential, retry_total=0)
    return _projects[endpoint]


async def close_clients():
    """Close HTTP sessions and credentials before the CLI event loop exits."""
    resources = [*_clients.values(), *_projects.values(), *_credentials]
    _clients.clear()
    _projects.clear()
    _credentials.clear()
    seen = set()
    for resource in resources:
        if id(resource) in seen:
            continue
        seen.add(id(resource))
        try:
            await resource.close()
        except Exception as error:
            log.warning("Could not close Foundry client: %s", error)


async def with_client_cleanup(call):
    try:
        return await call
    finally:
        await close_clients()


def _openai_client():
    """Return the project OpenAI client used to invoke Prompt Agents."""
    endpoint = os.environ.get("FOUNDRY_PROJECT_ENDPOINT", "").strip().rstrip("/")
    if not endpoint:
        raise RuntimeError("set FOUNDRY_PROJECT_ENDPOINT to the Foundry project endpoint")
    mode = _auth_mode()
    cache_key = (mode, endpoint)
    if cache_key in _clients:
        return _clients[cache_key]

    if mode == "entra":
        client = _project_client().get_openai_client(max_retries=OPENAI_MAX_RETRIES)
    else:
        from openai import AsyncOpenAI

        api_key = os.environ.get(PROJECT_API_KEY_ENV) or os.environ.get("FOUNDRY_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"{AUTH_MODE_ENV}=api_key requires {PROJECT_API_KEY_ENV} or FOUNDRY_API_KEY"
            )
        client = AsyncOpenAI(
            base_url=f"{endpoint}/openai/v1",
            api_key=api_key,
            max_retries=OPENAI_MAX_RETRIES,
        )

    _clients[cache_key] = client
    return client


def _image_input(path) -> dict:
    """Return an ordered Responses API image item, resized to the workflow limit."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"cannot read image {path}")
    limit = load_workflow().max_image_side
    height, width = image.shape[:2]
    if max(height, width) > limit:
        scale = limit / max(height, width)
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, png = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"cannot encode image {path}")
    encoded = base64.b64encode(png.tobytes()).decode("ascii")
    return {
        "type": "input_image",
        "image_url": f"data:image/png;base64,{encoded}",
    }


def _jsonable(value):
    """Convert SDK response metadata into JSON-safe values for provenance files."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    for method in ("model_dump", "to_dict"):
        converter = getattr(value, method, None)
        if callable(converter):
            return _jsonable(converter())
    return repr(value)


def _response_text(response) -> str:
    """Return text output or raise a useful error for refusal/filter/incomplete output."""
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    details = []
    for attribute in ("status", "incomplete_details", "error"):
        value = getattr(response, attribute, None)
        if value is not None:
            details.append(f"{attribute}={_jsonable(value)!r}")
    for item in getattr(response, "output", None) or []:
        item_type = str(getattr(item, "type", ""))
        if "refusal" in item_type.lower():
            details.append(f"refusal={_jsonable(getattr(item, 'content', item))!r}")
    suffix = "; ".join(details) or "no response details"
    raise RuntimeError(f"Prompt Agent returned no text output ({suffix})")


def _provenance(response, agent_name: str, agent_version: str) -> dict:
    return {
        "agent_name": agent_name,
        "agent_version": agent_version,
        "response_id": getattr(response, "id", None),
        "usage": _jsonable(getattr(response, "usage", None)) or {},
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dry_run": False,
    }


class StepAgent:
    """A workflow step's Prompt Agent: prompt + ordered images + schema-valid answer."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.schema = load_schema(config.schema)
        self.example = load_json(config.example)
        self.validator = Draft202012Validator(self.schema)
        self._client = None
        self.last_provenance: dict = {}

    @property
    def agent_name(self) -> str:
        name = os.environ.get(self.config.name_env, self.config.name) if self.config.name_env else self.config.name
        name = name.strip()
        if not name:
            raise RuntimeError(f"set {self.config.name_env or 'the Prompt Agent name'}")
        return name

    @property
    def agent_version(self) -> str:
        version = os.environ.get(self.config.version_env, self.config.version) if self.config.version_env else self.config.version
        version = str(version).strip()
        if not version:
            raise RuntimeError(f"set {self.config.version_env or 'the Prompt Agent version'}")
        return version

    def _get_client(self):
        if self._client is None:
            self._client = _openai_client()
        return self._client

    def _get_project_client(self):
        return _project_client()

    def prompt(self, values: dict) -> str:
        """Render local instructions, step data, and the local JSON contract."""
        rendered = render_prompt(self.config.prompt, {"example": self.example, "vocabulary": vocabulary(), **values})
        instructions = self.config.instructions.strip()
        if self.config.response_mode == "artifacts":
            return "\n\n".join(part for part in (instructions, rendered) if part)
        schema_request = (
            "Return exactly one JSON object and no Markdown fences or explanatory text. "
            "The JSON must satisfy this schema:\n"
            + json.dumps(self.schema, indent=2, ensure_ascii=False)
        )
        return "\n\n".join(part for part in (instructions, rendered, schema_request) if part)

    async def ask_artifacts(self, values, *, files, images=(), output_dir,
                            expected_files, report_name, dry_run_files=None):
        """Use unstructured Code Interpreter output; validate the downloaded report."""
        from src.workflow.artifact_agent import ask_artifacts

        return await ask_artifacts(
            self, values, files=files, images=images, output_dir=output_dir,
            expected_files=expected_files, report_name=report_name,
            dry_run_files=dry_run_files,
        )

    async def ask(
        self,
        values: dict,
        images=(),
        save_to: pathlib.Path | None = None,
        missing_defaults: dict | None = None,
    ) -> dict:
        """Send one request and return the schema-valid JSON answer.

        ``missing_defaults`` is deliberately narrow: it may supply a safe value
        only when a top-level key is absent.  The completed answer still passes
        the configured schema, and the normalization is recorded in provenance.
        This lets a stage convert an omitted optional-to-the-workflow ledger into
        explicit unresolved records without weakening the shared JSON contract.
        """
        text = self.prompt(values)
        if dry_run():
            log.info("%s: dry run, answering with %s", self.config.name, self.config.example)
            answer = copy.deepcopy(self.example)
            self.last_provenance = {
                "agent_name": self.agent_name,
                "agent_version": self.agent_version,
                "response_id": None,
                "usage": {},
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "dry_run": True,
            }
        else:
            agent_name = self.agent_name
            agent_version = self.agent_version
            log.info(
                "%s: invoking Prompt Agent %s version %s",
                self.config.name,
                agent_name,
                agent_version,
            )
            response = await self._get_client().responses.create(
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": text},
                        *(_image_input(path) for path in images),
                    ],
                }],
                extra_body={
                    "agent_reference": {
                        "name": agent_name,
                        "version": agent_version,
                        "type": "agent_reference",
                    }
                },
                timeout=self.config.timeout_seconds,
            )
            self.last_provenance = _provenance(response, agent_name, agent_version)
            output_text = _response_text(response)
            try:
                answer = json.loads(output_text)
            except json.JSONDecodeError as error:
                preview = output_text[:500].replace("\n", " ")
                raise RuntimeError(f"Prompt Agent returned non-JSON output: {preview!r}") from error
        applied_defaults = []
        if missing_defaults and isinstance(answer, dict):
            answer = dict(answer)
            for key, value in missing_defaults.items():
                if key not in answer:
                    answer[key] = copy.deepcopy(value)
                    applied_defaults.append(key)
            if applied_defaults:
                log.warning(
                    "%s: response omitted %s; applied safe defaults before validation",
                    self.config.name,
                    ", ".join(applied_defaults),
                )
                self.last_provenance["normalizations"] = [
                    {"action": "missing_top_level_default", "key": key}
                    for key in applied_defaults
                ]
        if save_to is not None:
            provenance_path = pathlib.Path(save_to).with_name(
                f"{pathlib.Path(save_to).stem}.provenance.json"
            )
            provenance_path.write_text(
                json.dumps(self.last_provenance, indent=1, ensure_ascii=False), encoding="utf-8"
            )
        errors = sorted(self.validator.iter_errors(answer), key=lambda error: list(error.path))
        if errors:
            details = "; ".join(f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors[:5])
            raise ValueError(f"{self.config.name} answer does not match {self.config.schema}: {details}")
        if save_to is not None:
            pathlib.Path(save_to).write_text(json.dumps(answer, indent=1, ensure_ascii=False), encoding="utf-8")
        return answer

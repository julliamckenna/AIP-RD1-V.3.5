"""Create Foundry Prompt Agent versions from the model mappings in .env.

Preview the resolved configuration:
    python scripts/sync_prompt_agents.py

Create immutable Prompt Agent versions:
    python scripts/sync_prompt_agents.py --apply

The command prints the FOUNDRY_AGENT_<NN>_VERSION values to pin in .env.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from dataclasses import dataclass

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    AutoCodeInterpreterToolParam,
    CodeInterpreterTool,
    PromptAgentDefinition,
    PromptAgentDefinitionTextOptions,
    TextResponseFormatJsonSchema,
    TextResponseFormatText,
)
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv, set_key

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.settings import ROOT
from src.workflow.resources import load_schema, load_workflow


@dataclass(frozen=True)
class AgentSpec:
    number: str
    name: str
    model: str
    instructions: str
    response_format_name: str
    response_schema: dict
    response_mode: str = "json"

    @property
    def version_env(self) -> str:
        return f"FOUNDRY_AGENT_{self.number}_VERSION"


def required_setting(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"set {name} in {ROOT / '.env'}")
    return value


def load_specs(numbers=None) -> list[AgentSpec]:
    """Resolve every agent name and desired model deployment from .env."""
    specs = []
    for step in load_workflow().steps.values():
        if not step.agent:
            continue
        number = step.id[5:7]
        if numbers and number not in numbers:
            continue
        specs.append(
            AgentSpec(
                number=number,
                name=required_setting(f"FOUNDRY_AGENT_{number}_NAME"),
                model=required_setting(f"FOUNDRY_MODEL_DEPLOYMENT_NAME_AGENT_{number}"),
                instructions=step.agent.instructions,
                response_format_name=step.id,
                response_schema=load_schema(step.agent.schema),
                response_mode=step.agent.response_mode,
            )
        )
    return specs


def definition_for(spec):
    """Only Agents 03/04 use free text responses and a Python file runtime."""
    if spec.response_mode == "artifacts":
        return PromptAgentDefinition(
            model=spec.model, instructions=spec.instructions,
            text=PromptAgentDefinitionTextOptions(format=TextResponseFormatText()),
            tools=[CodeInterpreterTool(container=AutoCodeInterpreterToolParam(memory_limit="4g"))],
        )
    return PromptAgentDefinition(
        model=spec.model, instructions=spec.instructions,
        text=PromptAgentDefinitionTextOptions(format=TextResponseFormatJsonSchema(
            name=spec.response_format_name, schema=spec.response_schema, strict=True,
        )),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="create versions; without this flag, only print the resolved plan",
    )
    parser.add_argument("--agents", nargs="+", choices=["00", "01", "02", "03", "04"],
                        help="sync only these agent numbers")
    parser.add_argument("--write-env", action="store_true",
                        help="pin each successfully created version in the local .env (requires --apply)")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    endpoint = required_setting("FOUNDRY_PROJECT_ENDPOINT").rstrip("/")
    if args.write_env and not args.apply:
        parser.error("--write-env requires --apply")
    specs = load_specs(args.agents)

    print("Prompt Agent version plan:")
    for spec in specs:
        mode = "free text + Code Interpreter files" if spec.response_mode == "artifacts" else "structured JSON"
        print(f"  {spec.number}: {spec.name} <- {spec.model} ({mode})")
    if not args.apply:
        print("\nPreview only. Run again with --apply to create the versions.")
        return

    project_client = AIProjectClient(
        endpoint=endpoint,
        credential=DefaultAzureCredential(),
    )
    created = []
    for spec in specs:
        result = project_client.agents.create_version(
            agent_name=spec.name,
            definition=definition_for(spec),
            description="Chart extraction Prompt Agent; model selected from local .env",
            metadata={"managed_by": "chart-extraction-workflow", "response_mode": spec.response_mode},
        )
        version = str(result.version)
        created.append((spec, version))
        print(f"created {spec.name} version {version} ({spec.model})")
        if args.write_env:
            set_key(str(ROOT / ".env"), spec.version_env, version, quote_mode="never")

    print("\nPin these immutable versions in .env:")
    for spec, version in created:
        print(f"{spec.version_env}={version}")


if __name__ == "__main__":
    main()

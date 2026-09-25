"""One small, paid Code Interpreter file round trip; never runs chart extraction.

Usage: python scripts/check_artifact_transport.py --agent 03
"""

import argparse
import asyncio
import hashlib
import json
import pathlib
import sys
import zipfile
from dataclasses import replace

from dotenv import load_dotenv
from jsonschema import Draft202012Validator

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.settings import ROOT
from src.workflow.agents import StepAgent, with_client_cleanup
from src.workflow.console import configure_console_logging
from src.workflow.resources import load_workflow


async def check(number):
    config = next(step.agent for step in load_workflow().steps.values()
                  if step.id.startswith(f"agent{number}_"))
    agent = StepAgent(replace(config, timeout_seconds=180))
    folder = ROOT / "artifacts" / "analysis" / "artifact_transport" / number
    folder.mkdir(parents=True, exist_ok=True)
    source = b"chart-extraction file transport check\n"
    bundle = folder / "transport_input.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("transport_input.txt", source)
    expected_hash = hashlib.sha256(source).hexdigest()
    agent.validator = Draft202012Validator({
        "type": "object", "required": ["status", "input_sha256"], "additionalProperties": False,
        "properties": {"status": {"const": "ok"}, "input_sha256": {"const": expected_hash}},
    })
    agent.prompt = lambda values: (
        "This is a file transport check, not a chart extraction task. Use Code Interpreter to find the uploaded "
        "transport_input.zip, read transport_input.txt from it, compute its SHA256 with Python, and write "
        "/mnt/data/transport_check.json containing only status='ok' and input_sha256=<computed hex digest>. "
        "Reload the JSON to check it. Your final response must cite transport_check.json as a downloadable file."
    )
    result = await agent.ask_artifacts({}, files=[bundle], output_dir=folder,
                                       expected_files=["transport_check.json"], report_name="transport_check.json")
    print(json.dumps({"agent": number, "source_version": agent.agent_version,
                      "result": result["report"], "receipt": str(result["directory"])}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=["03", "04"], required=True)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=True)
    configure_console_logging()
    asyncio.run(with_client_cleanup(check(args.agent)))

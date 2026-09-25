"""Recover cited files from a completed response; never invoke a model."""

import argparse
import asyncio
import hashlib
import json
import pathlib
import sys

from dotenv import load_dotenv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.settings import ROOT
from src.workflow.agents import StepAgent, with_client_cleanup
from src.workflow.artifact_agent import download_cited_files
from src.workflow.resources import load_workflow


async def recover(attempt, number):
    directory = pathlib.Path(attempt).resolve()
    directory.relative_to((ROOT / "artifacts").resolve())
    config = next(step.agent for step in load_workflow().steps.values() if step.id.startswith(f"agent{number}_"))
    agent = StepAgent(config)
    request = json.loads((directory / "request.json").read_text())
    if request["agent_name"] != agent.agent_name:
        raise ValueError("Saved attempt belongs to a different agent")
    expected = request["expected_files"]
    if any(not name or name in {".", ".."} or "/" in name or "\\" in name for name in expected):
        raise ValueError("Expected artifact names must be basenames")
    response = json.loads((directory / "response.json").read_text())
    await download_cited_files(agent._get_client(), response, directory, expected)
    report_name = "agent03_review_points.json" if number == "03" else "agent04_final_check.json"
    report = json.loads((directory / report_name).read_text(encoding="utf-8-sig"))
    agent.validator.validate(report)
    receipt = {"response_id": response["id"], "model_invoked": False,
               "files": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in expected}}
    (directory / "recovered.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print(json.dumps({"recovered": expected, "directory": str(directory), "model_invoked": False}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--agent", choices=["03", "04"], required=True)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=True)
    asyncio.run(with_client_cleanup(recover(args.attempt, args.agent)))

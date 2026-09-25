"""Entry point: run the CO2 chart extraction workflow locally; only the model calls go to Foundry.

    python main.py --run paper_c.pdf    process one document (a file in data/input, a path or a URL)
    python main.py --graph              print the workflow graph (Mermaid)

Every file is written to your machine (artifacts/runs, or CHART_EXTRACT_OUTPUT_DIR). Nothing is deployed.
To check and correct points by hand afterwards: python scripts/review_server.py

Rerun an existing run from a persisted panel stage:
    python main.py --rerun --agent 03 --only
    python main.py --rerun --agent 03 --all
"""

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

# The repository's .env is the source of truth even when the parent shell contains
# stale FOUNDRY_* values from an earlier run.
load_dotenv(Path(__file__).resolve().with_name(".env"), override=True)

from src.workflow.build import build_workflow, mermaid  # noqa: E402
from src.workflow.console import configure_console_logging, log_context  # noqa: E402
from src.workflow.rerun import rerun, rerun_all_inputs  # noqa: E402
from src.workflow.agents import with_client_cleanup  # noqa: E402


async def run_once(document: str) -> str:
    with log_context(source=document, figure="-", step=""):
        response = await build_workflow().as_agent().run(document)
    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--run", metavar="DOCUMENT", help="process one PDF/image (file in data/input, path or URL) and exit")
    operation.add_argument("--rerun", action="store_true", help="rerun agents for all input files, or one --document")
    parser.add_argument("--graph", action="store_true", help="print the workflow graph as Mermaid and exit")
    parser.add_argument("--document", help="one run directory name/path for --rerun (default: every file in data/input)")
    parser.add_argument(
        "--figure", "--figure-id", dest="figure",
        help="rerun only this canonical figure/panel id, for example figp1a or fig3a",
    )
    parser.add_argument("--agent", metavar="NN", help="starting agent for --rerun: 01, 02, 03 or 04")
    rerun_mode = parser.add_mutually_exclusive_group()
    rerun_mode.add_argument("--only", action="store_true", help="rerun only the selected agent")
    rerun_mode.add_argument("--all", dest="all_downstream", action="store_true",
                             help="rerun the selected agent and all downstream stages")
    args = parser.parse_args()
    configure_console_logging()
    if args.graph:
        print(mermaid())
    elif args.run:
        print(asyncio.run(with_client_cleanup(run_once(args.run))))
    elif args.rerun:
        if not args.agent:
            parser.error("--rerun requires --agent 01, 02, 03 or 04")
        if not args.only and not args.all_downstream:
            parser.error("--rerun requires either --only or --all")
        try:
            operation = (
                rerun(args.document, args.agent, args.all_downstream, args.figure)
                if args.document
                else rerun_all_inputs(args.agent, args.all_downstream, args.figure)
            )
            print(asyncio.run(with_client_cleanup(operation)))
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        raise SystemExit(130)

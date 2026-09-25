# chart-extraction

Extract CO2‚ isotherm data from research-paper figures into CSV tables. The workflow runs locally; only Prompt Agent
requests go to your Foundry project. Nothing is deployed.

One run processes one document. Each model request handles one figure or panel.

## Quick start

Requirements: Windows, PowerShell, Python 3.11+, Azure CLI, and a Foundry project with five pinned Prompt Agents that
support images and structured output. Your account needs the **Azure AI User** role on the project.

```powershell
.\scripts\setup.ps1
```

Add these values to `.env`:

```text
FOUNDRY_PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>
FOUNDRY_AGENT_00_NAME=<agent-name>
FOUNDRY_AGENT_00_VERSION=<version>
FOUNDRY_AGENT_01_NAME=<agent-name>
FOUNDRY_AGENT_01_VERSION=<version>
FOUNDRY_AGENT_02_NAME=<agent-name>
FOUNDRY_AGENT_02_VERSION=<version>
FOUNDRY_AGENT_03_NAME=<agent-name>
FOUNDRY_AGENT_03_VERSION=<version>
FOUNDRY_AGENT_04_NAME=<agent-name>
FOUNDRY_AGENT_04_VERSION=<version>
```

Use `FOUNDRY_AUTH_MODE=api_key` and `FOUNDRY_PROJECT_API_KEY` only when you are not using Entra authentication.
For Entra authentication, run `az login`. Keep `.env` private.

Put a PDF or image in `data/input/`, then run:

```powershell
python main.py --run paper.pdf
```

Without Azure, test the wiring with example responses:

```powershell
$env:CHART_EXTRACT_DRY_RUN=1; python main.py --run paper.pdf; $env:CHART_EXTRACT_DRY_RUN=0
```

## Workflow

`workflow.yaml` is the source of truth. In brief:

1. Python inventories the document and its figures.
2. Agent 00 identifies CO2 isotherms and panels.
3. Agent 01 reads each panel's axes, legend, and series.
4. Python calibrates the axes and detects marker points.
5. Agent 02 checks the extraction and can request a bounded rebuild loop.
6. Agents 03 and 04 review, correct, and validate the points.
7. Python publishes CSV tables, reports, and the review dashboard.

## Common commands

```powershell
python main.py --graph                         # show the workflow
python -m pytest -q                            # run tests
python scripts/review_server.py                # review and correct points
python scripts/rebuild_table.py                # rebuild tables from saved edits
python main.py --rerun --agent 03 --all        # every supported file under data/input
python main.py --rerun --document RUN --agent 03 --all
python main.py --rerun --document RUN --agent 04 --only
python -m evaluations.telemetry_usage          # token and cost tables
python scripts/sync_prompt_agents.py --agents 03 04 --apply --write-env   # update and pin file-based reviewers
```

Without `--document`, rerun processes every supported file under `data/input`; files without the requested saved
starting stage are reported as skipped. Use `--only` to rerun one agent. Use `--all` to continue through downstream stages. For discovery or crop changes,
start a new run with `python main.py --run <input-file>`.

## Outputs

Results are written to `artifacts/runs/` or the folder set by `CHART_EXTRACT_OUTPUT_DIR`:

- `final.csv` â€” final rows for one document.
- `all_data_accepted.csv` â€” rows accepted by Agent 04.
- `all_data_review.csv` â€” rows that still need human review.
- `dashboard.html` and `review.html` â€” review pages.
- JSON, images, overlays, and audit files â€” intermediate evidence and provenance.

## Project rules

- Keep `workflow.yaml`, step code, prompts, schemas, and examples named with the same step ID.
- Keep model calls in `src/workflow/agents.py`; keep extraction tools in `src/tools/` deterministic.
- Update schemas, examples, prompts, and tests together. Run `python -m pytest -q` after changes.
- Never commit `.env`, secrets, or generated run output.
- Follow [docs/naming.md](docs/naming.md) and [docs/instructions.md](docs/instructions.md).

## Known limits

Vector figures are rendered as images. Log axes are supported; broken axes and double y-axes are not. Hidden or
overlapping markers may require manual review.

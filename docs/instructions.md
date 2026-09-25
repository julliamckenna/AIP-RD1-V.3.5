# Setup and development instructions

This page combines onboarding, normal operation, and change guidance. `workflow.yaml` is the source of truth for the
workflow and its Prompt Agent settings.

## Get started

Requirements:

- Windows, PowerShell, and Python 3.11+.
- Azure CLI (`az login`).
- A Foundry project endpoint and five pinned Prompt Agents: Agents 00â€“02 use structured output; Agents 03/04 use Code Interpreter file outputs.
- The **Azure AI User** role on the Foundry project.

Run setup from the project folder:

```powershell
.\scripts\setup.ps1
```

Set `FOUNDRY_PROJECT_ENDPOINT` and the `FOUNDRY_AGENT_<NN>_NAME` / `FOUNDRY_AGENT_<NN>_VERSION` values in `.env`.
Use Entra authentication by running `az login`, or set `FOUNDRY_AUTH_MODE=api_key` and
`FOUNDRY_PROJECT_API_KEY`. Keep `.env` private.

Put a document in `data/input/` and run:

```powershell
python main.py --run <file-name>
```

To test without Azure, use the example responses:

```powershell
$env:CHART_EXTRACT_DRY_RUN=1; python main.py --run <file-name>; $env:CHART_EXTRACT_DRY_RUN=0
```

Review points with `python scripts/review_server.py`. The server writes corrections to `artifacts/reviews/` and
rebuilds the tables. Use `python scripts/rebuild_table.py` to rebuild them without the server.

## Files to change

Each workflow step uses the same step ID across its files:

| Purpose | Location |
|---|---|
| Workflow, agent instructions, and graph | `workflow.yaml` |
| Step implementation | `src/workflow/steps/<step_id>.py` |
| Model task and runtime placeholders | `prompts/<step_id>.md` |
| Response contract (00â€“02) or report-file contract (03/04) | `schemas/<step_id>.schema.json` |
| Valid response example | `examples/<step_id>.example.json` |
| Shared schema definitions | `schemas/shared.schema.json` |
| Tests | `tests/` |

When changing a schema, update its example and prompt together. Agents 00â€“02 use strict response schemas: every
property is required and `additionalProperties` is `false`. Agents 03/04 have unstructured final messages that cite
files. Their schemas validate the downloaded JSON report; they are not API response constraints. QA additionally
creates a full CSV using `schemas/points_csv.contract.json`. If you add or rename a prompt placeholder, provide it
from the step and update the matching test contract.

Prompt wording and report-file schema changes are read locally on the next run. Changes to `agent.instructions`,
response mode, tools, or an Agents 00â€“02 response schema require a new immutable Foundry Prompt Agent version:

```powershell
python scripts/sync_prompt_agents.py --agents 03 04                       # preview
python scripts/sync_prompt_agents.py --agents 03 04 --apply --write-env    # create and pin versions
```

The sync utility configures Agents 03/04 with Code Interpreter and free-text responses. `--write-env` pins each
successfully created version in `.env`; without it, copy the reported versions yourself. Other agents are unchanged
when `--agents 03 04` is specified. Start a fresh process after synchronization to reload environment settings.

## Python tools and workflow steps

Keep helpers in `src/tools/` deterministic and free of model calls. Call them from the relevant step and add focused
tests. If a model needs a helper result, pass it through a named prompt placeholder and label it as supporting evidence.
For images, pass the file through the step's `images` input and identify whether it is source material or a derived
diagnostic.

For a new step:

1. Use an ID such as `agentNN_<verb>_<noun>` or `python_<verb>_<noun>`.
2. Add `src/workflow/steps/<step_id>.py` and register it in `src/workflow/build.py`.
3. Add the step and graph edges to `workflow.yaml`.
4. For an agent, also add its prompt, schema, and example.
5. Run `python main.py --graph` and the contract tests.

All model calls go through `src/workflow/agents.py` and its file transport in `src/workflow/artifact_agent.py`.
Agent 03 reviews the complete table in one Code Interpreter call and writes a JSON edit report. Local
`src/tools/stage_edits.py` applies it and rebuilds the Agent 03 candidate.

Agent 04 independently inspects the native panel, directly writes the complete replacement `agent04_points.csv`,
and writes `agent04_final_check.json`. It may add, remove, move, or reassign rows. `src/tools/qa_csv.py` validates
header/order, IDs, eligible series, native evidence for changes, calibration, and audit counts. Python preserves
all authored CSV cells and rebuilds canonical plots/metadata from those rows. There is no candidate-row restoration
or silent numerical correction. Invalid files fail the stage and remain available for inspection.

Both agents receive a lossless archive of the native panel, complete tables/reports, and every adjudication tile.
Foundry disallows tools/text overrides with agent_reference. The workflow fetches the pinned definition, binds
the uploaded file IDs in a temporary agent copy, invokes that copy, and removes it afterward. The pinned base
agent is unchanged. Project agent CRUD uses your Entra identity, including when Responses uses API-key mode.
Code Interpreter can create extra crops anywhere in the native image. To add a local Python tool, keep its helper
in `src/tools/`, call it from the workflow, and add its output to `src/workflow/review_bundle.py` when the remote
reviewer needs it. Bundling a helper result does not make that result authoritative. To add a different remote
tool, update the definition in `scripts/sync_prompt_agents.py`, the request transport if needed, and synchronize
new agent versions.

Source images are authoritative. Python detections, scores, overlays, and redraws are supporting evidence only.

## Timeouts, retries, and run receipts

Set `agent.timeout_seconds` in `workflow.yaml`. Agents 03/04 currently allow 900 seconds for the full file review;
progress is logged every 30 seconds. SDK retries remain zero. A timeout/cancellation stops the call without an
automatic paid replay; use rerun after inspecting the error. To change retry behavior, make it explicit in the
workflow and test cancellation before enabling it.

Each `agent03_unstructured/attempt-...` or `agent04_unstructured/attempt-...` folder contains the exact input archive,
`request.json` (rendered prompt, version, hashes), `response.json`, `provenance.json`, `runtime_agent.json` (bound definition), cited files, and `failure.json`
on failure. Uploaded inputs are temporary. Raw model files stay separate from reconciled canonical stage files.

## Rerun a saved document

```powershell
# Rebuild Agent 03, Agent 04, and published output
python main.py --rerun --document RUN --agent 03 --all

# Rerun only Agent 04; it still refreshes final tables
python main.py --rerun --document RUN --agent 04 --only

# Recreate the Agent 01 spec and extraction, then continue downstream
python main.py --rerun --document RUN --agent 01 --all
```

`--only` leaves later agent stages unchanged, except Agent 04 still refreshes final and master tables. Start a new run
for discovery, figure detection, or crop changes; rerunning Agent 01 cannot fix an incorrect saved crop.

File-based Agents 03/04 always make a fresh call on rerun; old Agent 03 batch checkpoints are ignored. Check
`agent03_edit_audit.json` for unapplied edits and `agent04_edit_audit.json` for the changes made by the final CSV.
To exercise both changed agents on the current run:

```powershell
python main.py --rerun --document calf20_02 --agent 03 --all
```

## Checks after changes

```powershell
python -m pytest -q
```

For focused workflow or reviewer changes:

```powershell
python -m pytest -q tests/test_workflow_contracts.py tests/test_reviewer_edit_contract.py tests/test_qa_csv.py tests/test_artifact_agent.py tests/test_ownership_stage_edits.py
```

After changing `src/tools/extract.py`, `dense_regions.py`, or extraction settings, also run:

```powershell
python -m evaluations.generate evaluations/cases
python -m evaluations.run evaluations/cases evaluations/report
```

Do not commit `.env`, secrets, or generated `artifacts/` output. Follow [naming.md](naming.md) for IDs, stage files,
series labels, and spelling.

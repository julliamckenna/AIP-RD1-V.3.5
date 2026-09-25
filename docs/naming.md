# Naming convention

One rule per kind of name, written once here and enforced by `tests/test_naming.py`. The workflow's Prompt Agent
calls follow it too: every agent prompt gets the exact vocabulary through `$vocabulary`.

## 1. Steps own their files

A step id is `agentNN_<verb>_<noun>` (a model call) or `python_<verb>_<noun>` (deterministic Python).
The step id names everything that belongs to the step:

| Thing | Name |
|---|---|
| code | `src/workflow/steps/<step_id>.py` |
| prompt | `prompts/<step_id>.md` |
| response schema | `schemas/<step_id>.schema.json` |
| example answer | `examples/<step_id>.example.json` |
| agent answer | `<figure_id>/<step_id>.json` (`agent02_check_extraction_round<N>.json` per loop round) |

## 2. Stages are named after the step that writes them

The extracted points exist in three stages. Each stage file is `<stage>_<part>`
(`src/models/naming.py`: `STAGES`, `stage_file()`):

| Stage | Written by | Files |
|---|---|---|
| `python` | `python_extract_points` | `python_points.csv/json`, `python_metadata.json`, `python_overlay.png`, `python_compare.png`, `python_redraw_diff.png`, `python_hard_checks.json`, `python_summary.json`, `python_audit.json` |
| `agent03` | `agent03_review_points` | `agent03_points.csv/json`, `agent03_metadata.json`, `agent03_edit_audit.json`, `agent03_overlay.png`, `agent03_compare.png`, `agent03_redraw_diff.png`, `agent03_hard_checks.json` |
| `agent04` | `agent04_final_check` | `agent04_points.csv/json`, `agent04_metadata.json`, `agent04_edit_audit.json`, `agent04_overlay.png`, `agent04_compare.png`, `agent04_hard_checks.json` |

User-facing files keep plain names: `<document>/final.csv` (every final row of a document),
`all_data*.csv`, `dashboard.html`, `review.html`, `panel.png`, `spec.json`.

## 3. Ids

| Id | Rule | Example |
|---|---|---|
| document | the input file name without extension (`config.yaml` `input.document_name`) | `combined_04` |
| figure / folder | `config.yaml` `naming.figure_id`; the folder name *is* the id | `fig2`, `fig3a`, `figp7` |
| panel | lowercase panel letter, `x` for a single-panel figure (`naming.panel_id`) | `a`, `x` |
| series | `<figure_id>:<panel_id>:<label slug>-<hash>` | `fig3a:a:co2-195k-d3fd5c48` |
| point | `pt-<hash>` | `pt-6a16ebd68e6ff815` |

## 4. Series labels

`<gas> <temperature>K[ <branch>]`, as the legend reads: `CO2 273K`, `CO2 273K ads`, `N2 77K`.
No space before `K`, `CO2` not `CO₂`. `naming.series_label()` normalises anything an agent writes
(`CO2 273 K` -> `CO2 273K`) before labels are compared; code never compares raw labels.

## 5. Vocabulary

Marker shapes, fills, line styles, axis scales, branches and verdicts come only from the enums in
`schemas/shared.schema.json` (`triangle_up`, never "up triangle"). Prompts show them through
`$vocabulary` (`src/workflow/resources.py`); nothing else defines them.

## 6. Spelling and keys

- American spelling in code, keys, config and docs: `color`, `color_hex`, `color_tolerance`.
- JSON/YAML keys and Python names are `snake_case`; CSV columns follow `schemas/points_csv.contract.json`.

## 7. Environment variables

| Prefix | For | Examples |
|---|---|---|
| `FOUNDRY_` | the Foundry project and Prompt Agents | `FOUNDRY_PROJECT_ENDPOINT`, `FOUNDRY_AGENT_00_NAME`, `FOUNDRY_AGENT_00_VERSION` |
| `CHART_EXTRACT_` | this project's own switches | `CHART_EXTRACT_OUTPUT_DIR`, `CHART_EXTRACT_REVIEWS_DIR`, `CHART_EXTRACT_DRY_RUN` |

The per-step Prompt Agent overrides are `FOUNDRY_AGENT_<NN>_NAME` and `FOUNDRY_AGENT_<NN>_VERSION`, where `<NN>` is
the step's agent number; `workflow.yaml` names them in `name_env` and `version_env`, and `.env.example` lists every one.

## 8. Where things live

| Folder | Holds |
|---|---|
| `src/workflow/` | the workflow graph, agents, state; `steps/` one module per step |
| `src/tools/` | deterministic Python (extraction, checks, artifacts); no model calls |
| `src/models/` | naming and shared helpers |
| `prompts/`, `schemas/`, `examples/` | one file per agent step (rule 1) plus `shared.schema.json`, the CSV contract and the CO2 scope policy |
| `scripts/` | command-line utilities (review server, rebuild tables, setup) |
| `evaluations/` | the benchmark |
| `docs/` | documentation |

Only project files live in the repository root (`README.md`, `workflow.yaml`, `config.yaml`,
`main.py`, packaging and environment templates).

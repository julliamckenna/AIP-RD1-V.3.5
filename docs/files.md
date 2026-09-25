# What each file does

Grouped by role. For the order in which things run, see `README.md` → *The workflow*; for names, see
`docs/naming.md`.

Everything runs on your machine; only Prompt Agent calls go to Foundry.

Analogy: `workflow.yaml` is the Power Automate flow definition, each `src/workflow/steps/*.py` is one action,
`prompts/` + `schemas/` + `examples/` are the AI Builder prompt and its output format, `src/tools/` is the
custom connector code that does the real pixel work, and `config.yaml` holds the environment variables of the
solution.

## Root - the project's control files

| File | What it does |
|---|---|
| `workflow.yaml` | **Single source of truth for the workflow**: the 8 steps, which are agents (name, version, overrides, local instructions), their local prompt / schema / example, the edges (including the agent 02 → Python `adjust` loop), `max_check_rounds`, image size limit |
| `config.yaml` | Folders, input filter, naming patterns, PDF rendering, every deterministic extraction setting (`extract:`) and the hard-check tolerances (`checks:`) |
| `main.py` | Entry point: `--run <pdf>` processes one document; `--graph` prints the workflow graph |
| `requirements.txt` | Python packages (OpenCV, PyMuPDF, Agent Framework, Azure AI Projects client, jsonschema …), each with the reason it is needed |
| `pyproject.toml` | pytest settings (import path) |
| `.env.example` | Template for `.env`: Foundry endpoint, Prompt Agent name/version per step, output folders, dry run |
| `.gitignore` | Only for version control: keeps `.env`, `.venv`, `artifacts/` and input PDFs out of it; not used by the code |
| `README.md` | Overview: workflow, output layout, setup, run, review, benchmark, rules for changing the code, limits |

## `src/workflow/` - the workflow engine (Microsoft Agent Framework)

| File | What it does |
|---|---|
| `build.py` | Turns `workflow.yaml` into an Agent Framework `WorkflowBuilder` graph; maps step ids to step classes (`STEP_CLASSES`); prints the Mermaid graph |
| `agents.py` | **The only place that calls a Prompt Agent.** Renders local instructions, prompt data and the JSON schema, attaches ordered images, validates the answer, saves it and a provenance sidecar; dry-run mode answers with the example |
| `resources.py` | Loads `workflow.yaml`, schemas (inlining shared `$ref`s), examples and prompt templates; renders prompts (rows as CSV / compact JSON) and the shared `$vocabulary` block |
| `state.py` | The message passed along the edges: `DocumentRun` (one document) and `Panel` (one chart panel: status, check rounds, row counts, verdict); route names |
| `steps/base.py` | Shared plumbing for steps: run one panel, catch errors per panel, JSON read/write |

## `src/workflow/steps/` - one file per step (in run order)

| File | Kind | What it does | Main outputs |
|---|---|---|---|
| `python_read_document.py` | Python | Takes the request (attached or named PDF/image), lists every figure candidate with page image, caption and mentions | `discovery/python_inventory.json` |
| `agent00_find_figures.py` | agent | Per figure: is it a CO2 isotherm chart, which panels, figure number; creates the panel list and folder names | `discovery/agent00_find_figures.json` |
| `agent01_read_chart.py` | agent | Per panel: crops `panel.png`, the agent reads axes, ticks, legend, series (label, marker, color, gas); builds `spec.json` for Python | `agent01_read_chart.json`, `spec.json` |
| `python_extract_points.py` | Python | Runs the extractor on `spec.json`, writes the python stage, runs the hard checks, builds the summary agent 02 sees; a calibration or axis-range failure is recorded for agent 02 | `python_points.csv/json`, `python_summary.json`, `python_hard_checks.json`, images |
| `agent02_check_extraction.py` | agent | Judges Python's result (numbers first), corrects the structure and coaches Python (sample markers, curves, counts, region strategies); not satisfied → Python runs again (≤ `max_check_rounds`) | `agent02_check_extraction_round<N>.json` |
| `agent03_review_points.py` | agent | Reviews the Python points with zoomed evidence tiles; returns add / move / delete / reassign operations; applies them | `agent03_points.csv/json`, `agent03_hard_checks.json`, images |
| `agent04_final_check.py` | agent | Independent QA from a per-series summary and Python's suspect points; verdict + last operations; blocks values that fail the hard axis check; builds the review page | `agent04_points.csv/json`, `qa.json`, `review.html` |
| `python_publish_results.py` | Python | Writes the document's `final.csv` and run summary, rebuilds the master tables and dashboard, returns the Markdown report | `final.csv`, `all_data*.csv`, `dashboard.html` |

## `src/tools/` - deterministic Python (no model calls)

| File | What it does |
|---|---|
| `extract.py` | **The extractor**: finds the axis frame and ticks, calibrates pixels → values, reads legend colors/glyphs, detects every marker of every series, separates look-alike series, fills crowded strips; writes one extraction per panel |
| `legend_markers.py` | Turns each legend symbol into a vector glyph (shape incl. triangle direction, open/filled, colors, outline); used for detection, overlays and the legend check |
| `marker_assignment.py` | Decides which series owns a marker when colors overlap (shape, color, trajectory scores) |
| `series_gate.py` | Separates same-color series by curve position (agent 01 anchors, agent 02 `curve_points` / `y_bands`) |
| `dense_regions.py` | Crowded low-pressure regions: validates agent regions, resolves marker cores, lists unresolved slots |
| `template_fill.py` | Places a series' legend template on unresolved crowded slots (`template_fit` / inferred `template_fill`) |
| `region_strategies.py` | Applies agent 02's coaching inside the extraction: color sampling at a marker, `ignore`, `split_filled_open`, `sample_band_at_columns` |
| `hard_checks.py` | Checks the numbers: value range vs printed axes and frame (blocking), redraw recall/precision + diff image, physics flags |
| `checks.py` | Physics checks on isotherms (rising with pressure, colder above warmer, isosteric heat range) |
| `suspects.py` | Builds agent 04's input: per-series summary table and the suspect points (inferred rows, points off their own curve) |
| `stage_artifacts.py` | Writes a stage (`<stage>_points.csv/json`, `<stage>_metadata.json`) in the CSV contract, with stable point ids |
| `stage_edits.py` | Applies agent 03/04 operations mechanically through the calibration, with an audit; turns rows into prompt tables |
| `adjudicate_tiles.py` | Zoomed native evidence tiles for agent 03 (uncertain markers, complex regions) |
| `recreate.py` | Re-plots a stage from its numbers and puts it next to the source (`*_compare.png`) - the main visual check |
| `evidence.py` | Preserves the source figure and crops panels traceably for agent 01 |
| `pdf_reader.py` | Reads a PDF: raster and vector figures as PNG, captions, sentences that mention each figure |
| `table.py` | Builds `all_data.csv` / `_accepted` / `_review` and the dashboard from every panel on disk, applying saved human reviews |
| `review_page.py` + `review.css` | The standalone HTML review page (drag / add / delete points, export corrections) |

## `src/` - shared

| File | What it does |
|---|---|
| `settings.py` | Loads `config.yaml` (`CFG`), the project root (`ROOT`) and folder paths with `CHART_EXTRACT_*_DIR` overrides |
| `models/naming.py` | Every name pattern in code: figure ids, run ids, review file names, stage file names, `panel_id()`, `series_label()` |

## `prompts/`, `schemas/`, `examples/` - what each agent is told and must answer

One file per agent step, named by the step id (`agent00_find_figures` … `agent04_final_check`):

| Folder | What it holds |
|---|---|
| `prompts/<step>.md` | The task text with `$placeholders` filled at run time (Python results, images list, `$vocabulary`, `$example`) |
| `schemas/<step>.schema.json` | The exact JSON the agent must return (strict: every field required, no extra fields) |
| `examples/<step>.example.json` | A valid answer: shown to the model, used for dry runs, checked by the tests |
| `schemas/shared.schema.json` | Shared pieces referenced by every schema: marker / fill / line style / verdict vocabularies, boxes, axes, point operations, extraction settings, region strategies, `axis_check` |
| `schemas/points_csv.contract.json` + `examples/points.example.csv` | The output CSV contract: columns, evidence types, allowed values |
| `schemas/co2_scope.policy.json` | Which charts and series count as CO2 uptake data (others are context only) |

## `scripts/` - command-line utilities

| File | What it does |
|---|---|
| `setup.ps1` | One-time install on Windows: `.venv`, packages, folders, `.env`, tests, graph |
| `setup_azcli.ps1` | Installs the Azure CLI into `.azcli` on Windows without admin rights |
| `review_server.py` | Local server for review pages and the dashboard; saves corrections and rebuilds the tables |
| `rebuild_table.py` | Rebuilds the master tables and dashboard from disk (no server) |

## `tests/` - run with `python -m pytest -q`

| File | What it checks |
|---|---|
| `test_workflow_contracts.py` | `workflow.yaml`, step classes, prompts, schemas and examples stay in sync; schemas are strict; examples are valid |
| `test_hard_checks.py` | Axis-range check (shifted / stretched / legitimate overhang), curve points, region strategies, color sampling, suspects, tick cross-check |
| `test_naming.py` | The naming convention: spelling, stage file names, step ids, snake_case keys, vocabulary, labels, panel ids, repository root |

## `evaluations/` - benchmark for the extractor

| File | What it does |
|---|---|
| `generate.py` | Creates 24 synthetic isotherm charts with exact ground truth (4-17 series, degraded like real figures) |
| `run.py` | Runs the extractor on every case and scores recall, precision, series ownership, centre error |
| `diagnose.py` | For one case: which extraction stage lost each missed marker |
| `report/` | The current benchmark result (`REPORT.md`, `results.csv`); rerun the benchmark to refresh it. Also holds sample journal PDFs from the original project |

## `docs/`

| File | What it covers |
|---|---|
| `implementation.md` | Step-by-step setup, first run, review, tuning, troubleshooting |
| `files.md` | This file |
| `naming.md` | The naming convention (enforced by `tests/test_naming.py`) |
| `architecture.md` | How the parts fit together |
| `instructions.md`, `operations.md`, `safety-and-usage.md` | Getting started, running it day to day, safe use |
| `domain/` | Reference notes (measurement rules, unit conversions); not loaded by the code |

## Tool configuration folders

| Folder | What it holds |
|---|---|
| `.vscode/` | Run configurations: run one document, dry run, graph, tests, review server, rebuild tables |
| `data/input/` | Put PDFs here |

## Output files (per run, on your machine)

```
artifacts/runs/<document>/
├─ discovery/                        python_inventory.json, agent00 answers, run_summary.json
├─ <figure_id>/                      one folder per panel, e.g. fig2a
│   ├─ panel.png, native.png         the panel crop sent to the agents / untouched source crop
│   ├─ spec.json                     what Python extracts from (agent 01 + agent 02 corrections)
│   ├─ agent01_read_chart.json … agent04_final_check.json    each agent's answer
│   ├─ python_points.csv/json        Python's proposal       (+ _metadata, _overlay, _compare, _redraw_diff, _hard_checks, _summary, _audit)
│   ├─ agent03_points.csv/json       after agent 03's edits  (+ _metadata, _edit_audit, _overlay, _compare, _redraw_diff, _hard_checks)
│   ├─ agent04_points.csv/json       after agent 04 = final  (+ _metadata, _edit_audit, _overlay, _compare, _hard_checks)
│   ├─ qa.json                       physics + hard-check summary, verdict, row counts
│   └─ review.html                   drag-and-drop review page
└─ final.csv                         every final row of this document
artifacts/runs/all_data.csv          all documents (accepted + review)
artifacts/runs/all_data_accepted.csv rows agent 04 accepted
artifacts/runs/all_data_review.csv   rows that still need a human
artifacts/runs/dashboard.html        one card per panel, links to the review pages
artifacts/reviews/                   corrections saved from the review pages
```

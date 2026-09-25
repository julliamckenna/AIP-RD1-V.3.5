# Architecture

```text
request (PDF / image / name / URL)
  -> main.py                       --run <document>
  -> src/workflow/build.py         workflow.yaml -> Agent Framework WorkflowBuilder graph
  -> src/workflow/steps/<step>.py  one executor per step; agent steps call src/workflow/agents.py -> Foundry Prompt Agents
  -> src/tools/                    deterministic library: PDF reading, extraction, stages, tables, review pages
  -> artifacts/runs/<document>/    discovery/, <figure_id>/, final.csv  (+ cross-document all_data*.csv)
```

## Pieces

| Area | Owner |
|---|---|
| `workflow.yaml` | Steps, agents (name/version overrides, local instructions, prompt, schema, example) and edges |
| `config.yaml` | Folders, naming patterns, PDF and extraction tuning |
| `src/workflow/state.py` | `DocumentRun` / `Panel` - the message passed along the edges, and the route names |
| `src/workflow/agents.py` | The only code that calls a Prompt Agent: local instructions + prompt + ordered images + strict schema, answer validated |
| `src/workflow/resources.py` | Loads YAML/JSON, inlines shared `$ref`s, fills prompt `$placeholders` |
| `src/workflow/steps/` | One module per step, named by step id |
| `src/tools/` | Deterministic extraction and publishing (unchanged algorithms) |
| `prompts/`, `schemas/`, `examples/` | Per-step prompt, response schema and example; shared definitions and data contracts |

## How the graph works

Every edge carries one `DocumentRun`. A step returns a new `DocumentRun` with a `route`; `workflow.yaml` edges
with `when: <route>` pick the next step, edges without `when` are always taken. Routes: `continue`,
`next_panel` (loop over panels), `adjust` (agent 02 corrected the structure: Python rebuilds), `publish`.

The panel steps (agent 01 → agent 04) work on `run.panel`. When a panel is finished early (no eligible CO2 series, failed calibration, an
exception) the remaining panel steps pass it through, and agent 04 moves on to the next panel - one bad panel never
stops the document. Large data never travels in the message: each step writes its files to the figure folder and
the next step reads them, so every stage is auditable on disk.

## Check loop and calibration rule

Python (the Python extraction) runs straight after the agent 01 reading. Agent 02 checks every Python result; when it is not satisfied
its corrected structure is merged into `spec.json` and the Python extraction runs again, up to `max_check_rounds`.

Tick candidates must satisfy the deterministic affine-fit residual (`max_tick_fit_residual_px`). Agent 02 anchors may
only choose among valid candidates, and the chosen mapping must pass `max_tick_anchor_residual_px`. When calibration
fails, the Python extraction draws `python_tick_check.png` with the exact detected candidates; agent 02 maps printed values to them,
nearby anchors are snapped (`agent02_tick_reconciliation_round<N>.json`) and the same strict checks run in the next
round. The loop never loosens a gate.

## Evidence rule

Only native-supported marker cores become rows. Unbracketed dense-region slots stay unresolved without a fabricated
coordinate. Color-ambiguous ownership needs native geometry and a score margin; excluded series (e.g. N2) may win
classification but never produce rows. Model answers are operations on point ids / normalized source positions;
`src/tools/stage_edits.py` converts them through the calibrated axes and records an audit entry per operation.

## Hard checks on the numbers

Agents judge images; `src/tools/hard_checks.py` judges numbers, on the python, agent03 and agent04 stages:

- **Axis range (blocking):** every value must lie inside the printed axis range plus a margin, and the plot frame,
  converted with the fitted calibration, may overhang the outer printed tick by at most one tick step
  (`config.yaml` → `checks:`). A shifted or stretched tick mapping fails here even when every marker sits on the right
  pixel. At the python stage a failure goes back to agent 02 as a calibration error; at the agent04 stage the panel
  is blocked and never published.
- **Redraw score:** per series, the share of the source's marker ink covered by a point (recall) and the share of
  points sitting on ink (precision), with a diff image (`<stage>_redraw_diff.png`).
- **Physics flags** from `src/tools/checks.py` (isotherms rise with pressure, colder above warmer).

Agents 02 and 04 fill an `axis_check` only for what needs the image: the tick labels they read and whether the
re-plot puts the curves at the same values as the source.

## Agent 02 coaches Python

Besides correcting the structure, agent 02 can give Python measurable guidance, applied by
`src/tools/region_strategies.py` and `src/tools/series_gate.py`:

| Field | Python uses it to |
|---|---|
| `sample_marker_norm` | measure the series color on one clean marker instead of the legend |
| `expected_counts` | report found vs expected markers per x range (`count_check` in the next round) |
| `curve_points` | assign crowded and same-color markers to the series whose curve explains them |
| `y_bands`, `separate_from`, `color_tolerance`, `template_fill` | separate look-alike series and tune detection per series |
| `region_strategies` + `shared_x_columns` | `ignore` a box, `split_filled_open`, or take line samples in a fused band |

## What agent 04 sees

Instead of every row, agent 04 gets a per-series summary, the hard-check numbers and only the points Python flags as
suspect (`src/tools/suspects.py`: inferred rows, and points off their own curve with the series that fits better).

# Implementation guide - from zero to a first local run

Follow the steps in order. Each step says what to do, the exact command, and what success looks like.
The project runs on **Windows**. Every command is for **PowerShell** (for example the VS Code terminal); run it from the
project folder (the one that contains `workflow.yaml`).

This is a **local Python project**: the whole workflow runs on your machine and writes every file to your disk; only
the Prompt Agent calls go to your Microsoft Foundry project. Nothing is deployed.

Think of it like a Power Automate desktop flow that calls an AI Builder model: the flow and its files live on your
PC, and only the AI model lives in the cloud.

---

## Step 0 - What you need before you start (ask for these up front)

| Need | Why | Who gives it |
|---|---|---|
| Azure subscription | Foundry project and Prompt Agents | your Azure admin |
| A **Microsoft Foundry project** | the agents' home; Prompt Agent calls go here | create it (Step 3) or reuse one |
| Role **Azure AI User** on the project | lets your login call the Prompt Agents | project owner |
| Five pinned **Prompt Agent versions** whose underlying models read images and return JSON schema output | every agent sends images and must answer in its step's schema | you, in the Foundry portal (Step 3) |
| The project folder (`chart-extract`) | the code | handed over with this guide |

Tools on your machine:

| Tool | Version | Check with |
|---|---|---|
| Python | 3.11 or newer | `python --version` |
| Azure CLI (`az`) | 2.60+ | `az version` (Windows without admin rights: `.\scripts\setup_azcli.ps1` installs it into `.azcli`) |
| VS Code | optional, recommended | run configurations for every command below (Run and Debug) |

---

## Step 1 - Get the code

Copy the `chart-extract` project folder to your machine (for example unzip it into `C:\projects\chart-extract`)
and open a terminal in it:

```powershell
cd C:\projects\chart-extract
```

**Success:** `workflow.yaml`, `config.yaml` and `src/` are in the folder.

---

## Step 2 - Install (once)

```powershell
.\scripts\setup.ps1
```

If Windows answers *"running scripts is disabled on this system"*, run it once this way instead (it changes nothing
permanently):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

The script creates `.venv` (a private Python environment for this project), installs `requirements.txt`, creates
the folders `data/input`, `artifacts/runs`, `artifacts/reviews`, copies `.env.example` to `.env`, runs the tests and
prints the workflow graph.

**Success:** the last lines show all tests passing and a Mermaid graph starting with `python_read_document`.

From now on use the project's Python:

```powershell
.\.venv\Scripts\Activate.ps1
```

(After activating, `python` means the project's Python and the prompt starts with `(.venv)`. If activation is blocked,
run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or use `.\.venv\Scripts\python.exe` instead of
`python` in every command below.)

---

## Step 3 - Prepare Microsoft Foundry (portal, once)

1. Open <https://ai.azure.com>, pick your subscription, **Create project** (or open an existing one).
2. Create or verify five **Prompt Agent** versions. Each underlying model must support **image input** and
   **structured outputs (JSON schema)**. Record each agent's exact name and pinned version.
3. Project **Overview**: copy the **project endpoint**, it looks like
   `https://<resource>.services.ai.azure.com/api/projects/<project>`.
**Success:** you have the project endpoint and five Prompt Agent name/version pairs.

---

## Step 4 - Fill in `.env`

Open `.env` (created in Step 2) and set:

```ini
FOUNDRY_PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>
FOUNDRY_AUTH_MODE=entra
FOUNDRY_AGENT_00_NAME=agent00-discovery
FOUNDRY_AGENT_00_VERSION=3
FOUNDRY_AGENT_01_NAME=agent01-structure
FOUNDRY_AGENT_01_VERSION=1
FOUNDRY_AGENT_02_NAME=agent02-calibration
FOUNDRY_AGENT_02_VERSION=1
FOUNDRY_AGENT_03_NAME=agent03-extraction
FOUNDRY_AGENT_03_VERSION=1
FOUNDRY_AGENT_04_NAME=agent04-qa
FOUNDRY_AGENT_04_VERSION=1
```

Use `FOUNDRY_AUTH_MODE=entra` for the local `DefaultAzureCredential` chain. To use the
OpenAI-compatible project endpoint with a key instead, set `FOUNDRY_AUTH_MODE=api_key` and
`FOUNDRY_PROJECT_API_KEY` in `.env`; never commit that key.

The values override the pinned defaults in `workflow.yaml`; keep versions explicit so old run artifacts remain
reproducible. Do not use `latest`.
The next run picks the change up - no code change needed.

Other optional lines (leave empty for the default):

| Variable | Use |
|---|---|
| `CHART_EXTRACT_OUTPUT_DIR` | write results elsewhere, e.g. `C:/Users/<you>/OneDrive/chart-runs` |
| `CHART_EXTRACT_REVIEWS_DIR` | where the review page saves corrections |
| `CHART_EXTRACT_DRY_RUN=1` | no Azure calls; every agent answers with its example (wiring test only) |

`.env` holds your project settings: keep it private and never hand it over with the code.

---

## Step 5 - Sign in to Azure

```powershell
az login
```

A browser opens; sign in with the account that has the roles from Step 0. Local runs use this login
(`DefaultAzureCredential`) when `FOUNDRY_AUTH_MODE=entra`.

**Success:** `az account show` prints the right subscription. Wrong one? `az account set --subscription "<name or id>"`.

---

## Step 6 - Smoke tests (no model calls)

```powershell
python -m pytest -q                 # 43 passed
python main.py --graph              # prints the workflow graph
```

Wiring test through every step with example answers (put any PDF in `data/input/` first):

```powershell
$env:CHART_EXTRACT_DRY_RUN=1; python main.py --run paper.pdf; $env:CHART_EXTRACT_DRY_RUN=0
```

**Success:** a Markdown report table is printed and `artifacts/runs/paper/` exists. The values are meaningless in a
dry run; this only proves the wiring.

---

## Step 7 - First real run on one PDF

1. Copy the paper into `data/input/`, e.g. `data/input/combined_04.pdf`.
2. Run it:

   ```powershell
   python main.py --run combined_04.pdf
   ```

   Or in VS Code: **Run and Debug -> "1. Run one document from data/input"**.

3. Wait. A figure takes about 1-5 minutes: agent 02 may loop up to 3 times. Expected cost is roughly $0.20-1.70 per
   figure depending on the model and on how many series the chart has.

**Success:** the printed report has one row per panel and the files are on your disk:

```
artifacts/runs/combined_04/
├─ discovery/                 inventory, agent 00 answers, run_summary.json
├─ fig2a/                     one folder per panel (see docs/files.md -> "Output files")
│   ├─ panel.png, spec.json
│   ├─ python_points.csv, agent03_points.csv, agent04_points.csv
│   ├─ *_compare.png          source next to the chart re-plotted from the data - look at this first
│   └─ review.html
└─ final.csv                  every final row of the document
artifacts/runs/all_data.csv, all_data_accepted.csv, all_data_review.csv, dashboard.html
```

Panel status in the report:

| Status | Meaning | What to do |
|---|---|---|
| `done` | agent 04 accepted, hard checks passed | use `all_data_accepted.csv` |
| `review` | usable, a human must check the listed items | Step 8 |
| `failed` | calibration never passed, or the numbers contradicted the axes | read `<panel>/python_error.txt` / `qa.json` |

---

## Step 8 - Review and correct points by hand

```powershell
python scripts/review_server.py
```

Open <http://127.0.0.1:8765/dashboard.html>, pick a panel, drag / add / delete points, **Save corrections**. The
server writes `<document>__<figure>__reviewed.json` into `artifacts/reviews/` and rebuilds `all_data*.csv`.
Without the server: `python scripts/rebuild_table.py` rebuilds the tables from disk.

---

## Step 9 - Tune (optional)

| Goal | Where | How |
|---|---|---|
| Different agent behavior | Foundry | create and pin a new Prompt Agent version, then update its `FOUNDRY_AGENT_<NN>_VERSION` override |
| Role/safety behavior | `workflow.yaml` -> step -> `instructions` | edit the local instruction block; it is sent with every request |
| Fewer check rounds | `workflow.yaml` -> `max_check_rounds` | 3 -> 2 |
| Extraction behaviour | `config.yaml` -> `extract:` | every setting is commented; run the benchmark after changing |
| Axis tolerance | `config.yaml` -> `checks:` | `axis_range_margin`, `axis_frame_max_overhang_steps` |
| Output folder / naming | `config.yaml` -> `folders:` / `naming:` | see `docs/naming.md` |

---

## Step 10 - Change the code safely

The rules are in `README.md` -> *Rules for changing the code*; names follow `docs/naming.md`. After every change:

```powershell
python -m pytest -q                                              # must pass
python -m evaluations.generate evaluations/cases                 # once, creates the 24 benchmark charts
python -m evaluations.run evaluations/cases evaluations/report   # after changing src/tools/extract.py or config extract:
```

Adding an agent step: add it to `workflow.yaml`, then create `src/workflow/steps/<id>.py` (class in
`STEP_CLASSES`, `src/workflow/build.py`), `prompts/<id>.md`, `schemas/<id>.schema.json`,
`examples/<id>.example.json`; the contract tests tell you what is missing.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `set FOUNDRY_AGENT_NN_NAME or FOUNDRY_AGENT_NN_VERSION` | `.env` not filled / not loaded | Step 4; run from the project folder |
| `401` / `403` on Prompt Agent calls | login lacks **Azure AI User** on the project, or wrong subscription | Step 0 / Step 5 |
| `… answer does not match schemas/…` | the Prompt Agent returned JSON outside the schema (e.g. `"keep"`, `"up triangle"`) | use a version configured for structured outputs (Step 3) |
| Panel `failed`: `Axis range check failed` | tick labels mapped to the wrong tick marks (values outside the printed axis) | agent 02 gets 3 rounds to fix it; otherwise check `python_tick_check.png` and correct `ticks` in `spec.json` by hand |
| `TickAnchorMismatchError` | agent 02's tick positions are more than 3 px off the detected ticks | it copies `pixel_norm` from `python_tick_candidates.json` in the next round |
| Review page shows no points for an old run | runs made before the naming change use old file names | run the document again |

# Operations

## Calls per document

Agent 00 makes one call per figure candidate. Each eligible panel then makes one agent 01 call, one agent 02 call per
check round (1 to `max_check_rounds`), and one call each for agents 03 and 04. Calls run one after another; there are no batch calls.

## What happened in a run

Everything is on disk, per step, in `artifacts/runs/<document>/`:

- `discovery/run_summary.json` - every panel with its status, message, check rounds and row counts per stage;
- `<figure>/agent0N_*.json` - each agent's answer (`agent02_check_extraction_round<N>.json` per check round);
- `<figure>/python_summary.json`, `*_hard_checks.json`, `qa.json` - Python's numbers, the hard checks and the final QA;
- `<figure>/*.provenance.json` - Prompt Agent name/version, response id, usage and timestamp for each saved answer;
- the console log - one line per step and panel while the run is going.

## Configuration

| File | Holds |
|---|---|
| `workflow.yaml` | Prompt Agent names/versions, local instructions, timeout, image size, edges |
| `config.yaml` | folders, naming, PDF and extraction tuning |
| `.env` | endpoint, Prompt Agent names/versions, and desired model deployments - keep it private |

## Failures

| Symptom | Where to look / what to do |
|---|---|
| 401 / 403 | `az login`; your login needs the **Azure AI User** role on the Foundry project |
| authentication mode errors | check `FOUNDRY_AUTH_MODE` (`entra` or `api_key`) and the matching credential variables in `.env` |
| `set FOUNDRY_AGENT_NN_NAME or FOUNDRY_AGENT_NN_VERSION` | set the Prompt Agent override in `.env` |
| change a Prompt Agent model | set `FOUNDRY_MODEL_DEPLOYMENT_NAME_AGENT_NN`, run `python scripts/sync_prompt_agents.py --apply`, then pin the reported version in `.env` |
| `answer does not match schemas/...` | the model returned invalid JSON for that step; the panel is marked failed, see `error.txt` |
| panel `failed` after the check rounds | `python_error.txt`, `python_tick_check.png`, `agent02_check_extraction_round<N>.json` |
| panel `review` with "not satisfied after N rounds" | read `issues` in the last `agent02_check_extraction_round<N>.json` |
| panel `review` after agent 01 | agent 01 found no eligible CO2 series; check `agent01_read_chart.json` |
| any panel `failed` | `artifacts/runs/<document>/<figure>/error.txt` has the traceback |

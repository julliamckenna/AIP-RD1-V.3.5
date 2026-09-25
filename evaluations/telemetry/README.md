# Model usage telemetry

Generate the tables from the repository root:

```powershell
python -m evaluations.telemetry_usage
```

- `usage_by_call.csv` contains one row per Prompt Agent response, including
  the run, figure, step, agent, response ID, usage counters and estimated cost.
- `usage_by_run_agent.csv` sums those calls and costs for each run and agent
  version.

`cached_tokens` are input tokens read from the prompt cache.
`cache_write_tokens` are input tokens written to the prompt cache.
`uncached_input_tokens` is derived as
`input_tokens - cached_tokens - cache_write_tokens` (clamped at zero).
`cache_hit_pct` is `cached_tokens / input_tokens * 100`.

## Estimated cost

`evaluations/telemetry_pricing.json` maps each pinned agent version to its
model and records the USD rates used by the generator. The default profile uses
public OpenAI Standard short-context list prices as of 2026-09-24:

| Model | Input / 1M | Cached input / 1M | Cache write / 1M | Output / 1M |
|---|---:|---:|---:|---:|
| `gpt-5.6-sol` | $4.00 | $0.40 | $5.00 | $20.00 |
| `gpt-5.6-luna` | $0.20 | $0.02 | $0.25 | $1.20 |

Each component is calculated separately, then summed into
`estimated_total_cost_usd`. Reasoning tokens are already included in
`output_tokens` and are not charged a second time.

These are planning estimates, not billing records. Microsoft Foundry/Azure
pricing, regional uplifts, negotiated rates, taxes, long-context rates and
non-token charges can differ. Update `evaluations/telemetry_pricing.json` when
an agent version or price changes. The price source is
<https://developers.openai.com/api/docs/pricing>.

The generator reads both provenance layouts used by this project:

- `artifacts/runs/<run>/<figure>/*.provenance.json`
- provenance embedded in
  `artifacts/runs/<run>/discovery/agent00_find_figures.json`

The CSV files are snapshots. Rerun the command after adding or replacing runs.

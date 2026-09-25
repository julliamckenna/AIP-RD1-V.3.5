"""Build CSV token-usage tables from workflow provenance artifacts.

The workflow stores most model-call provenance in ``*.provenance.json``
sidecars. Discovery (agent 00) is stored inline in
``discovery/agent00_find_figures.json``. This module reads both forms.

Run from the repository root::

    python -m evaluations.telemetry_usage
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import defaultdict
from typing import Any, Iterable


COST_COLUMNS = [
    "estimated_uncached_input_cost_usd",
    "estimated_cache_write_cost_usd",
    "estimated_cached_input_cost_usd",
    "estimated_output_cost_usd",
    "estimated_total_cost_usd",
]

DETAIL_COLUMNS = [
    "run",
    "figure",
    "step",
    "agent_name",
    "agent_version",
    "model",
    "pricing_profile",
    "created_at",
    "response_id",
    "dry_run",
    "input_tokens",
    "cache_write_tokens",
    "cached_tokens",
    "uncached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
    "cache_hit_pct",
    *COST_COLUMNS,
    "source_path",
]

SUMMARY_COLUMNS = [
    "run",
    "agent_name",
    "agent_version",
    "model",
    "pricing_profile",
    "calls",
    "figures",
    "input_tokens",
    "cache_write_tokens",
    "cached_tokens",
    "uncached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
    "cache_hit_pct",
    *COST_COLUMNS,
]


def _integer(value: Any) -> int:
    """Return an integer token count, treating absent SDK fields as zero."""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"expected an integer token count, got {value!r}") from error


def _read_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read provenance JSON {path}: {error}") from error


def _detail_row(
    provenance: dict[str, Any],
    *,
    run: str,
    figure: str,
    step: str,
    source_path: str,
) -> dict[str, Any]:
    usage = provenance.get("usage") or {}
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}

    input_tokens = _integer(usage.get("input_tokens"))
    cache_write_tokens = _integer(input_details.get("cache_write_tokens"))
    cached_tokens = _integer(input_details.get("cached_tokens"))
    output_tokens = _integer(usage.get("output_tokens"))
    reasoning_tokens = _integer(output_details.get("reasoning_tokens"))
    total_tokens = _integer(usage.get("total_tokens"))
    # Input not reported as either a cache read or write. A clamp makes the
    # derived value robust to older SDK accounting/rounding behavior.
    uncached_input_tokens = max(
        0, input_tokens - cache_write_tokens - cached_tokens
    )

    return {
        "run": run,
        "figure": figure,
        "step": step,
        "agent_name": provenance.get("agent_name") or "",
        "agent_version": str(provenance.get("agent_version") or ""),
        "created_at": provenance.get("created_at") or "",
        "response_id": provenance.get("response_id") or "",
        "dry_run": bool(provenance.get("dry_run", False)),
        "input_tokens": input_tokens,
        "cache_write_tokens": cache_write_tokens,
        "cached_tokens": cached_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "cache_hit_pct": round(100 * cached_tokens / input_tokens, 2)
        if input_tokens
        else 0.0,
        "source_path": source_path,
    }


def collect_usage(runs_root: pathlib.Path) -> list[dict[str, Any]]:
    """Return one normalized row per model call under ``runs_root``."""
    runs_root = pathlib.Path(runs_root)
    rows: list[dict[str, Any]] = []

    for path in sorted(runs_root.glob("*/*/*.provenance.json")):
        relative = path.relative_to(runs_root)
        rows.append(
            _detail_row(
                _read_json(path),
                run=relative.parts[0],
                figure=relative.parts[1],
                step=path.name.removesuffix(".provenance.json"),
                source_path=relative.as_posix(),
            )
        )

    for path in sorted(runs_root.glob("*/discovery/agent00_find_figures.json")):
        relative = path.relative_to(runs_root)
        payload = _read_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"expected a list in discovery file {path}")
        for index, item in enumerate(payload, start=1):
            provenance = item.get("provenance") if isinstance(item, dict) else None
            if not isinstance(provenance, dict):
                continue
            candidate_id = item.get("candidate_id") or f"candidate_{index:03d}"
            rows.append(
                _detail_row(
                    provenance,
                    run=relative.parts[0],
                    figure="discovery",
                    step=f"agent00_find_figures:{candidate_id}",
                    source_path=f"{relative.as_posix()}#{index}",
                )
            )

    return sorted(
        rows,
        key=lambda row: (
            row["run"],
            row["agent_name"],
            row["created_at"],
            row["figure"],
            row["step"],
        ),
    )


def load_pricing(path: pathlib.Path) -> dict[str, Any]:
    """Load and minimally validate an explicit model-pricing profile."""
    pricing = _read_json(pathlib.Path(path))
    if not isinstance(pricing, dict):
        raise ValueError(f"expected a pricing object in {path}")
    for key in ("profile", "unit_tokens", "agent_versions", "models"):
        if key not in pricing:
            raise ValueError(f"pricing config {path} is missing {key!r}")
    if _integer(pricing["unit_tokens"]) <= 0:
        raise ValueError("pricing unit_tokens must be positive")
    return pricing


def apply_pricing(
    rows: Iterable[dict[str, Any]], pricing: dict[str, Any]
) -> list[dict[str, Any]]:
    """Attach model and estimated USD costs to detailed usage rows."""
    unit_tokens = _integer(pricing["unit_tokens"])
    profile = str(pricing["profile"])
    agent_versions = pricing["agent_versions"]
    models = pricing["models"]
    priced_rows = []

    for original in rows:
        row = dict(original)
        agent_version_key = f"{row['agent_name']}@{row['agent_version']}"
        model = agent_versions.get(agent_version_key, "")
        rates = models.get(model) if model else None
        row["model"] = model
        row["pricing_profile"] = profile if rates else "unpriced"
        if rates:
            components = {
                "estimated_uncached_input_cost_usd": row["uncached_input_tokens"]
                * float(rates["input_usd_per_million"])
                / unit_tokens,
                "estimated_cache_write_cost_usd": row["cache_write_tokens"]
                * float(rates["cache_write_usd_per_million"])
                / unit_tokens,
                "estimated_cached_input_cost_usd": row["cached_tokens"]
                * float(rates["cached_input_usd_per_million"])
                / unit_tokens,
                "estimated_output_cost_usd": row["output_tokens"]
                * float(rates["output_usd_per_million"])
                / unit_tokens,
            }
            row.update({key: round(value, 8) for key, value in components.items()})
            row["estimated_total_cost_usd"] = round(sum(components.values()), 8)
        else:
            row.update({key: "" for key in COST_COLUMNS})
        priced_rows.append(row)
    return priced_rows


def summarize_usage(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate detailed calls into one row for each run and agent version."""
    token_columns = [
        "input_tokens",
        "cache_write_tokens",
        "cached_tokens",
        "uncached_input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
    ]
    groups: dict[tuple[str, str, str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "calls": 0,
            "figures_set": set(),
            "priced_calls": 0,
            **{key: 0 for key in token_columns},
            **{key: 0.0 for key in COST_COLUMNS},
        }
    )
    for row in rows:
        key = (
            row["run"],
            row["agent_name"],
            row["agent_version"],
            row.get("model", ""),
            row.get("pricing_profile", ""),
        )
        group = groups[key]
        group["calls"] += 1
        group["figures_set"].add(row["figure"])
        for column in token_columns:
            group[column] += _integer(row[column])
        if row.get("estimated_total_cost_usd") != "" and row.get(
            "estimated_total_cost_usd"
        ) is not None:
            group["priced_calls"] += 1
            for column in COST_COLUMNS:
                group[column] += float(row[column])

    summary = []
    for (
        run,
        agent_name,
        agent_version,
        model,
        pricing_profile,
    ), group in sorted(groups.items()):
        input_tokens = group["input_tokens"]
        all_calls_priced = group["priced_calls"] == group["calls"]
        summary.append(
            {
                "run": run,
                "agent_name": agent_name,
                "agent_version": agent_version,
                "model": model,
                "pricing_profile": pricing_profile,
                "calls": group["calls"],
                "figures": len(group["figures_set"]),
                **{column: group[column] for column in token_columns},
                "cache_hit_pct": round(100 * group["cached_tokens"] / input_tokens, 2)
                if input_tokens
                else 0.0,
                **{
                    column: round(group[column], 6) if all_calls_priced else ""
                    for column in COST_COLUMNS
                },
            }
        )
    return summary


def _write_csv(
    path: pathlib.Path, rows: Iterable[dict[str, Any]], columns: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            for column in COST_COLUMNS:
                value = serialized.get(column, "")
                if value != "" and value is not None:
                    serialized[column] = f"{float(value):.8f}"
            writer.writerow(serialized)


def generate_tables(
    runs_root: pathlib.Path,
    output_dir: pathlib.Path,
    pricing_path: pathlib.Path | None = None,
) -> tuple[int, int]:
    detail = collect_usage(runs_root)
    if pricing_path is not None:
        detail = apply_pricing(detail, load_pricing(pricing_path))
    summary = summarize_usage(detail)
    _write_csv(output_dir / "usage_by_call.csv", detail, DETAIL_COLUMNS)
    _write_csv(output_dir / "usage_by_run_agent.csv", summary, SUMMARY_COLUMNS)
    return len(detail), len(summary)


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=pathlib.Path,
        default=repository / "artifacts" / "runs",
        help="directory containing run folders (default: artifacts/runs)",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=repository / "evaluations" / "telemetry",
        help="CSV output directory (default: evaluations/telemetry)",
    )
    parser.add_argument(
        "--pricing-config",
        type=pathlib.Path,
        default=repository / "evaluations" / "telemetry_pricing.json",
        help="agent/model rates used for estimated cost columns",
    )
    args = parser.parse_args(argv)
    calls, groups = generate_tables(
        args.runs_root, args.output_dir, args.pricing_config
    )
    print(f"wrote {calls} calls and {groups} run/agent rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

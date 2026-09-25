import csv
import json

from evaluations.telemetry_usage import (
    apply_pricing,
    collect_usage,
    generate_tables,
    summarize_usage,
)


def _provenance(agent, version, response, input_tokens, write, cached, output, reasoning):
    return {
        "agent_name": agent,
        "agent_version": version,
        "response_id": response,
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {
                "cache_write_tokens": write,
                "cached_tokens": cached,
            },
            "output_tokens": output,
            "output_tokens_details": {"reasoning_tokens": reasoning},
            "total_tokens": input_tokens + output,
        },
        "created_at": "2026-09-24T00:00:00+00:00",
        "dry_run": False,
    }


def test_collects_sidecars_and_embedded_discovery(tmp_path):
    figure_dir = tmp_path / "run_01" / "fig3a"
    figure_dir.mkdir(parents=True)
    sidecar = _provenance("agent02", "4", "resp_2", 100, 60, 30, 20, 5)
    (figure_dir / "agent02_round1.provenance.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )

    discovery_dir = tmp_path / "run_01" / "discovery"
    discovery_dir.mkdir()
    discovery = _provenance("agent00", "6", "resp_0", 50, 0, 40, 10, 2)
    (discovery_dir / "agent00_find_figures.json").write_text(
        json.dumps([{"candidate_id": "xref:4", "provenance": discovery}]),
        encoding="utf-8",
    )

    rows = collect_usage(tmp_path)

    assert len(rows) == 2
    assert {row["agent_name"] for row in rows} == {"agent00", "agent02"}
    agent02 = next(row for row in rows if row["agent_name"] == "agent02")
    assert agent02["uncached_input_tokens"] == 10
    assert agent02["cache_hit_pct"] == 30.0
    agent00 = next(row for row in rows if row["agent_name"] == "agent00")
    assert agent00["step"] == "agent00_find_figures:xref:4"


def test_summary_and_csv_output(tmp_path):
    rows = [
        {
            "run": "run_01",
            "figure": "fig2a",
            "agent_name": "agent02",
            "agent_version": "4",
            "input_tokens": 100,
            "cache_write_tokens": 60,
            "cached_tokens": 30,
            "uncached_input_tokens": 10,
            "output_tokens": 20,
            "reasoning_tokens": 5,
            "total_tokens": 120,
        },
        {
            "run": "run_01",
            "figure": "fig3a",
            "agent_name": "agent02",
            "agent_version": "4",
            "input_tokens": 200,
            "cache_write_tokens": 0,
            "cached_tokens": 150,
            "uncached_input_tokens": 50,
            "output_tokens": 40,
            "reasoning_tokens": 8,
            "total_tokens": 240,
        },
    ]

    pricing = {
        "profile": "test-pricing",
        "unit_tokens": 1_000_000,
        "agent_versions": {"agent02@4": "test-model"},
        "models": {
            "test-model": {
                "input_usd_per_million": 2.0,
                "cache_write_usd_per_million": 1.0,
                "cached_input_usd_per_million": 0.5,
                "output_usd_per_million": 4.0,
            }
        },
    }
    summary = summarize_usage(apply_pricing(rows, pricing))

    assert summary == [
        {
            "run": "run_01",
            "agent_name": "agent02",
            "agent_version": "4",
            "model": "test-model",
            "pricing_profile": "test-pricing",
            "calls": 2,
            "figures": 2,
            "input_tokens": 300,
            "cache_write_tokens": 60,
            "cached_tokens": 180,
            "uncached_input_tokens": 60,
            "output_tokens": 60,
            "reasoning_tokens": 13,
            "total_tokens": 360,
            "cache_hit_pct": 60.0,
            "estimated_uncached_input_cost_usd": 0.00012,
            "estimated_cache_write_cost_usd": 0.00006,
            "estimated_cached_input_cost_usd": 0.00009,
            "estimated_output_cost_usd": 0.00024,
            "estimated_total_cost_usd": 0.00051,
        }
    ]

    runs_root = tmp_path / "runs"
    figure_dir = runs_root / "run_02" / "fig1"
    figure_dir.mkdir(parents=True)
    (figure_dir / "agent01.provenance.json").write_text(
        json.dumps(_provenance("agent01", "1", "resp_1", 10, 0, 0, 2, 0)),
        encoding="utf-8",
    )
    output_dir = tmp_path / "tables"
    assert generate_tables(runs_root, output_dir) == (1, 1)
    with (output_dir / "usage_by_run_agent.csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["agent_name"] == "agent01"
    assert csv_rows[0]["input_tokens"] == "10"

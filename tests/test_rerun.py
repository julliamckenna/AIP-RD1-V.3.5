"""Tests for selecting and rebuilding persisted reruns without model calls."""

import asyncio
import json
import shutil
import uuid

import pytest

import src.workflow.rerun as rerun_module
from src.settings import ROOT
from src.workflow.rerun import find_run, load_run, normalise_agent, rerun_all_inputs, select_panels


@pytest.fixture
def workspace_tmp_path():
    path = ROOT / "artifacts" / "test-tmp" / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("01", "01"), ("1", "01"), ("agent02", "02"), ("03", "03"), ("agent04", "04")],
)
def test_normalise_agent(value, expected):
    assert normalise_agent(value) == expected


def test_normalise_agent_rejects_unknown_agent():
    with pytest.raises(ValueError, match="01, 02, 03, 04"):
        normalise_agent("05")


def test_find_run_defaults_to_newest_summary(workspace_tmp_path):
    older = workspace_tmp_path / "older" / "discovery"
    newer = workspace_tmp_path / "newer" / "discovery"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    for path, title in ((older, "older"), (newer, "newer")):
        (path / "run_summary.json").write_text(json.dumps({"title": title}), encoding="utf-8")
    older_summary = older / "run_summary.json"
    newer_summary = newer / "run_summary.json"
    older_summary.touch()
    newer_summary.touch()
    # Make the ordering deterministic on filesystems with coarse timestamp precision.
    import os
    os.utime(older_summary, (100, 100))
    os.utime(newer_summary, (200, 200))

    run_dir, summary = find_run(output_dir=workspace_tmp_path)

    assert run_dir == workspace_tmp_path / "newer"
    assert summary["title"] == "newer"


def test_load_run_rehydrates_persisted_panels(workspace_tmp_path):
    run = load_run(
        workspace_tmp_path,
        {
            "source": "paper.pdf",
            "title": "Paper",
            "run_id": "run-1",
            "panels": [{"folder": "fig2a", "figure": {}, "discovery": {}, "panel": {},
                        "figure_id": "fig2a", "panel_id": "a", "rows": {"agent04": 2}}],
        },
    )

    assert run.document_dir == str(workspace_tmp_path)
    assert run.panels[0].folder == "fig2a"
    assert run.panels[0].rows == {"agent04": 2}


def test_find_run_recovers_agent03_ready_panel_from_interrupted_run(workspace_tmp_path):
    run_dir = workspace_tmp_path / "partial"
    discovery = run_dir / "discovery"
    panel_dir = run_dir / "figp1a"
    discovery.mkdir(parents=True)
    panel_dir.mkdir()
    figure = {
        "page": 1, "kind": "image", "file": "figure.png", "source_id": "image",
        "figure_number": None, "caption": None, "caption_candidates": [], "references": [],
    }
    answer = {
        "figure_is_relevant": True,
        "figure_number": None,
        "matched_caption": "",
        "panels": [{"panel": "A", "is_co2_isotherm": True, "bbox_norm": [0, 0, 1, 1]}],
    }
    (discovery / "python_inventory.json").write_text(json.dumps({
        "pdf": "paper.png", "title": "Partial", "figures": [figure],
    }), encoding="utf-8")
    (discovery / "agent00_find_figures.json").write_text(json.dumps([{
        "candidate_id": "image", "figure": figure, "answer": answer,
    }]), encoding="utf-8")
    for name in (
        "panel.png", "spec.json", "python_points.json", "python_overlay.png",
        "python_summary.json", "agent02_check_extraction.json",
    ):
        (panel_dir / name).write_text("{}", encoding="utf-8")
    (panel_dir / "python_points.csv").write_text("point_id\npt-1\n", encoding="utf-8")

    selected, summary = find_run("partial", output_dir=workspace_tmp_path, agent="03")

    assert selected == run_dir
    assert [panel["folder"] for panel in summary["panels"]] == ["figp1a"]
    assert summary["panels"][0]["rows"] == {"python": 1}
    assert (discovery / "run_summary.json").is_file()


def test_select_panels_uses_figure_id_without_dropping_full_run(workspace_tmp_path):
    run = load_run(workspace_tmp_path, {
        "source": "paper.png", "title": "Paper", "run_id": "run-1",
        "panels": [
            {"folder": "fig2a", "figure": {}, "discovery": {}, "panel": {}, "figure_id": "fig2a", "panel_id": "a"},
            {"folder": "fig2b", "figure": {}, "discovery": {}, "panel": {}, "figure_id": "fig2b", "panel_id": "b"},
        ],
    })

    selected = select_panels(run, "FIG2B")

    assert [panel.figure_id for panel in selected] == ["fig2b"]
    assert [panel.figure_id for panel in run.panels] == ["fig2a", "fig2b"]


def test_select_panels_reports_available_ids(workspace_tmp_path):
    run = load_run(workspace_tmp_path, {
        "panels": [{"folder": "fig2a", "figure": {}, "discovery": {}, "panel": {},
                    "figure_id": "fig2a", "panel_id": "a"}],
    })
    with pytest.raises(ValueError, match="available: fig2a"):
        select_panels(run, "fig9")


def test_rerun_all_inputs_processes_accepted_files_in_order(workspace_tmp_path, monkeypatch):
    for name in ("zeta.png", "alpha.pdf", "ignore.txt", ".keep"):
        (workspace_tmp_path / name).write_text("test", encoding="utf-8")
    calls = []

    async def fake_rerun(document, agent, all_downstream, figure=None):
        calls.append((document, agent, all_downstream, figure))
        if document == "zeta":
            raise FileNotFoundError("not ready")
        return f"completed {document}"

    monkeypatch.setattr(rerun_module, "rerun", fake_rerun)
    result = asyncio.run(rerun_all_inputs("03", True, "fig2a", workspace_tmp_path))

    assert calls == [
        ("alpha", "03", True, "fig2a"),
        ("zeta", "03", True, "fig2a"),
    ]
    assert "completed alpha" in result
    assert "Skipped: not ready" in result

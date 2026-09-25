"""Rerun one or more per-panel workflow stages from an existing run.

The normal workflow starts by reading a document.  A rerun starts from the files already
written under ``artifacts/runs/<document>`` instead, which makes it useful when a prompt
agent or its configuration has changed and the PDF does not need to be read again.
"""

from __future__ import annotations

import dataclasses
import csv
import json
import logging
import pathlib
import re
import traceback

from src.settings import path_of
from src.models import naming
from src.tools import table
from src.workflow.console import log_context
from src.workflow.state import DocumentRun, Panel, PENDING
from src.workflow.steps.agent01_read_chart import ReadChart
from src.workflow.steps.agent00_find_figures import panels_for_figure
from src.workflow.steps.agent02_check_extraction import CheckExtraction
from src.workflow.steps.agent03_review_points import ReviewPoints
from src.workflow.steps.agent04_final_check import FinalCheck
from src.workflow.steps.python_extract_points import ExtractPoints
from src.workflow.steps.python_publish_results import report, write_document_final
from src.workflow.resources import load_workflow


AGENT_IDS = ("01", "02", "03", "04")
log = logging.getLogger("chart_extract")


def normalise_agent(value: str) -> str:
    """Accept ``03`` and ``agent03`` and return the canonical two-digit number."""
    text = str(value).strip().lower()
    if text.startswith("agent"):
        text = text[5:]
    if text in {"1", "2", "3", "4"}:
        text = f"0{text}"
    if text not in AGENT_IDS:
        raise ValueError(f"agent must be one of {', '.join(AGENT_IDS)}")
    return text


def _summary_paths(output_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        (path for path in output_dir.iterdir()
         if path.is_dir() and (path / "discovery" / "run_summary.json").is_file()),
        key=lambda path: (path / "discovery" / "run_summary.json").stat().st_mtime,
        reverse=True,
    )


def _stage_requirements(agent: str) -> tuple[str, ...]:
    return {
        "01": (),
        "02": ("panel.png", "spec.json", "python_points.json", "python_summary.json"),
        "03": (
            "panel.png", "spec.json", "python_points.json", "python_points.csv",
            "python_overlay.png", "python_summary.json", "agent02_check_extraction.json",
        ),
        "04": (
            "panel.png", "spec.json", "agent03_points.json", "agent03_points.csv",
            "agent03_overlay.png", "agent03_hard_checks.json",
        ),
    }[agent]


def _csv_rows(path: pathlib.Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _recover_partial_summary(document_dir: pathlib.Path, agent: str) -> dict | None:
    """Rebuild rerun state for completed panel stages in an interrupted run."""
    discovery_dir = document_dir / "discovery"
    inventory_path = discovery_dir / "python_inventory.json"
    answers_path = discovery_dir / "agent00_find_figures.json"
    if not inventory_path.is_file() or not answers_path.is_file():
        return None
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        entries = json.loads(answers_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(entries, list):
        return None

    discovered, used = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        figure = entry.get("figure") or {}
        answer = entry.get("answer") or {}
        if answer.get("figure_is_relevant"):
            discovered.extend(panels_for_figure(figure, answer, used))

    requirements = _stage_requirements(agent)
    ready = []
    skipped = []
    for panel in discovered:
        panel_dir = document_dir / panel.folder
        missing = [name for name in requirements if not (panel_dir / name).is_file()]
        if missing:
            skipped.append(f"{panel.folder} ({', '.join(missing)})")
            continue
        panel.rows = {
            stage: count for stage, count in (
                ("python", _csv_rows(panel_dir / "python_points.csv")),
                ("agent03", _csv_rows(panel_dir / "agent03_points.csv")),
                ("agent04", _csv_rows(panel_dir / "agent04_points.csv")),
            ) if count
        }
        rounds = [
            path for path in panel_dir.glob("agent02_check_extraction_round*.json")
            if re.fullmatch(r"agent02_check_extraction_round\d+\.json", path.name)
        ]
        panel.check_rounds = len(rounds)
        check_path = panel_dir / "agent02_check_extraction.json"
        if check_path.is_file():
            try:
                panel.needs_review = not bool(json.loads(check_path.read_text(encoding="utf-8")).get("satisfied"))
            except (OSError, json.JSONDecodeError):
                panel.needs_review = True
        ready.append(panel)
    if not ready:
        return None

    note = f"Recovered interrupted run for agent {agent}; {len(ready)} ready panel(s)."
    if skipped:
        note += f" Skipped incomplete panels: {'; '.join(skipped)}."
    summary = {
        "source": str(inventory.get("pdf") or ""),
        "title": str(inventory.get("title") or document_dir.name),
        "run_id": f"recovered-{document_dir.name}",
        "note": note,
        "panels": [dataclasses.asdict(panel) for panel in ready],
    }
    summary_path = discovery_dir / "run_summary.json"
    temporary = summary_path.with_name(f"{summary_path.name}.tmp")
    temporary.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    temporary.replace(summary_path)
    log.warning("%s", note)
    return summary


def find_run(
    document: str | None = None,
    output_dir: pathlib.Path | None = None,
    agent: str | None = None,
) -> tuple[pathlib.Path, dict]:
    """Find an existing run, preferring the newest summary when no document is given."""
    root = pathlib.Path(output_dir or path_of("output"))
    if document:
        candidate = pathlib.Path(document)
        if not candidate.is_absolute() or not candidate.is_dir():
            candidate = root / document
        summary_path = candidate / "discovery" / "run_summary.json"
        if not summary_path.is_file():
            recovered = _recover_partial_summary(candidate, agent) if agent in AGENT_IDS else None
            if recovered is not None:
                return candidate, recovered
            raise FileNotFoundError(f"no rerunnable run summary found at {summary_path}")
    else:
        paths = _summary_paths(root) if root.is_dir() else []
        if not paths:
            raise FileNotFoundError(f"no completed runs found under {root}")
        summary_path = paths[0] / "discovery" / "run_summary.json"
        candidate = paths[0]
    with summary_path.open(encoding="utf-8") as handle:
        return candidate, json.load(handle)


def load_run(document_dir: pathlib.Path, summary: dict) -> DocumentRun:
    """Rebuild the small workflow message from the persisted run summary."""
    panel_fields = {field.name for field in dataclasses.fields(Panel)}
    panels = tuple(Panel(**{key: value for key, value in item.items() if key in panel_fields})
                   for item in summary.get("panels", []))
    return DocumentRun(
        source=str(summary.get("source") or ""),
        document_dir=str(document_dir),
        run_id=str(summary.get("run_id") or "rerun"),
        title=str(summary.get("title") or document_dir.name),
        panels=panels,
        note=str(summary.get("note") or ""),
    )


def select_panels(run: DocumentRun, figure: str | None) -> tuple[Panel, ...]:
    """Return the requested canonical figure id, or every panel when omitted."""
    if not figure:
        return run.panels
    wanted = str(figure).strip().lower()
    selected = tuple(
        panel for panel in run.panels
        if panel.figure_id.lower() == wanted or panel.folder.lower() == wanted
    )
    if not selected:
        available = ", ".join(panel.figure_id for panel in run.panels) or "none"
        raise ValueError(f"figure {figure!r} is not in run {pathlib.Path(run.document_dir).name!r}; available: {available}")
    return selected


def _step_instances():
    config = load_workflow()
    return {
        "01": ReadChart(config.steps["agent01_read_chart"]),
        "02": CheckExtraction(config.steps["agent02_check_extraction"]),
        "03": ReviewPoints(config.steps["agent03_review_points"]),
        "04": FinalCheck(config.steps["agent04_final_check"]),
        "extract": ExtractPoints(config.steps["python_extract_points"]),
    }


async def _run_stage(step, panel_run: DocumentRun, panel: Panel):
    """Give direct rerun calls the same terminal context as workflow dispatch."""
    with log_context(source=panel_run.source, figure=panel.folder, step=step.id):
        log.info("%s", step.config.title)
        try:
            return await step.run_panel(panel_run, panel)
        except Exception as error:
            error_path = pathlib.Path(panel_run.document_dir) / panel.folder / "error.txt"
            log.error("Failed: %s (details: %s)", error, error_path)
            raise


async def _run_agent02(panel_run: DocumentRun, steps: dict, panel: Panel) -> None:
    """Run agent 02 and its required Python rebuild loop."""
    panel.check_rounds = 0
    panel.needs_review = False
    while True:
        route = await _run_stage(steps["02"], panel_run, panel)
        if route != "adjust":
            return
        await _run_stage(steps["extract"], panel_run, panel)


async def _run_selected(panel_run: DocumentRun, agent: str, all_downstream: bool, steps: dict) -> None:
    """Run the requested stage for every panel, optionally continuing downstream."""
    for panel in panel_run.panels:
        try:
            # A full downstream rerun must not be skipped because the prior run ended in
            # done/review.  An --only rerun keeps that prior status because later stages
            # intentionally remain untouched.
            if all_downstream:
                panel.status = PENDING
                panel.message = ""
            if agent == "01":
                await _run_stage(steps["01"], panel_run, panel)
                if all_downstream and panel.status == PENDING:
                    await _run_stage(steps["extract"], panel_run, panel)
                    await _run_agent02(panel_run, steps, panel)
                    if panel.status == PENDING:
                        await _run_stage(steps["03"], panel_run, panel)
                    if panel.status == PENDING:
                        await _run_stage(steps["04"], panel_run, panel)
            elif agent == "02":
                await _run_agent02(panel_run, steps, panel)
                if all_downstream and panel.status == PENDING:
                    await _run_stage(steps["03"], panel_run, panel)
                    if panel.status == PENDING:
                        await _run_stage(steps["04"], panel_run, panel)
            elif agent == "03":
                await _run_stage(steps["03"], panel_run, panel)
                if all_downstream:
                    await _run_stage(steps["04"], panel_run, panel)
            else:
                await _run_stage(steps["04"], panel_run, panel)
        except Exception as error:
            panel.status = "failed"
            panel.message = f"rerun agent {agent}: {error}"
            panel_dir = pathlib.Path(panel_run.document_dir) / panel.folder
            panel_dir.mkdir(parents=True, exist_ok=True)
            (panel_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")


def _save_summary(run: DocumentRun, note: str) -> None:
    path = pathlib.Path(run.document_dir) / "discovery" / "run_summary.json"
    data = {
        "source": run.source,
        "title": run.title,
        "run_id": run.run_id,
        "note": note,
        "panels": [dataclasses.asdict(panel) for panel in run.panels],
    }
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=1, ensure_ascii=False, default=float)
    temporary.replace(path)


async def rerun(
    document: str | None,
    agent: str,
    all_downstream: bool,
    figure: str | None = None,
) -> str:
    """Rerun ``agent`` from the newest (or named) persisted document run."""
    agent = normalise_agent(agent)
    document_dir, summary = find_run(document, agent=agent)
    full_run = load_run(document_dir, summary)
    selected = select_panels(full_run, figure)
    run = dataclasses.replace(full_run, panels=selected, index=0)
    with log_context(source=run.source, figure="-", step=f"agent{agent}"):
        log.info(
            "Rerun %d panel%s%s",
            len(selected),
            "" if len(selected) == 1 else "s",
            " with downstream stages" if all_downstream else " only",
        )
    if not run.panels:
        raise ValueError(f"run {document_dir.name!r} contains no panels")

    steps = _step_instances()
    await _run_selected(run, agent, all_downstream, steps)

    mode = "all downstream stages" if all_downstream else "only this stage"
    note = f"Rerun agent {agent} ({mode})."
    if not all_downstream:
        note += " Downstream artifacts were left unchanged; run with --all to refresh the final output."
    run = run.next(note=note)
    # Selected Panel objects are shared with ``full_run`` and have now been
    # updated in place.  Persist every panel so a targeted rerun never drops
    # unrelated results from the document summary.
    full_run = dataclasses.replace(full_run, panels=full_run.panels, note=note)
    _save_summary(full_run, note)

    if all_downstream or agent == "04":
        with log_context(source=full_run.source, figure="-", step="python_publish_results"):
            log.info("Building review pages and CSV tables")
            write_document_final(full_run)
            master_table, n_points = table.write_master_table(path_of("output"))
        return report(full_run, master_table, n_points)
    return report(run, None, 0) + f"\n\nRerun artifacts: `{document_dir}`."


async def rerun_all_inputs(
    agent: str,
    all_downstream: bool,
    figure: str | None = None,
    input_dir: pathlib.Path | None = None,
) -> str:
    """Rerun every accepted input-backed artifact directory in filename order."""
    agent = normalise_agent(agent)
    root = pathlib.Path(input_dir or path_of("input"))
    sources = sorted(
        (path for path in root.iterdir() if path.is_file() and naming.accepts_input(path)),
        key=lambda path: path.name.lower(),
    ) if root.is_dir() else []
    if not sources:
        raise FileNotFoundError(f"no accepted input files found under {root}")

    reports = []
    for index, source in enumerate(sources, start=1):
        document = naming.document_name(source)
        log.info(
            "Batch rerun input %d/%d: %s -> artifacts/runs/%s",
            index,
            len(sources),
            source.name,
            document,
        )
        try:
            result = await rerun(document, agent, all_downstream, figure)
            reports.append(f"## {source.name}\n\n{result}")
        except (FileNotFoundError, ValueError) as error:
            log.warning("Skipping %s: %s", source.name, error)
            reports.append(f"## {source.name}\n\nSkipped: {error}")
    return "\n\n".join(reports)

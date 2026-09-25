"""Publishing - publish results (Python) - the workflow's output.

Writes the document's final.csv (every final row of its figures) and
discovery/run_summary.json, rebuilds the cross-document master tables in artifacts/runs/
(all_data.csv, all_data_accepted.csv, all_data_review.csv), the dashboard and every review
page, then answers the request with a short Markdown report.

    artifacts/runs/file1/fig2/        one folder per figure panel (the panel steps (agent 01 → agent 04))
    artifacts/runs/file1/fig3a/
    artifacts/runs/file1/discovery/   document inventory, agent 00 answers, run summary
    artifacts/runs/file1/final.csv    this document's final rows
"""

from __future__ import annotations

import csv
import dataclasses
import pathlib

from agent_framework import Message, WorkflowContext, handler

from src.settings import CFG, path_of
from src.tools import table
from src.tools.stage_artifacts import STAGE_COLUMNS
from src.workflow.console import log_context
from src.workflow.state import DocumentRun
from src.workflow.steps.base import WorkflowStep, log, save_json


def write_document_final(run: DocumentRun) -> tuple[pathlib.Path, int]:
    """Concatenate every finished figure's agent04_points.csv into <document>/final.csv."""
    document_dir = pathlib.Path(run.document_dir)
    rows = []
    for panel in run.panels:
        path = document_dir / panel.folder / "agent04_points.csv"
        if panel.status in ("done", "review") and path.exists():
            with open(path, newline="", encoding="utf-8-sig") as handle:
                rows.extend({**row, "status": panel.status} for row in csv.DictReader(handle))
    out = document_dir / "final.csv"
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*STAGE_COLUMNS, "status"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return out, len(rows)


def report(run: DocumentRun, master_table: pathlib.Path | None, n_points: int) -> str:
    if not run.source:
        return run.note
    lines = [f"**{pathlib.Path(run.source).name}** - run `{run.run_id}`", ""]
    if run.note:
        lines += [run.note, ""]
    if run.panels:
        lines += ["| panel | status | python | agent03 | agent04 | note |", "|---|---|---|---|---|---|"]
        for panel in run.panels:
            rows = panel.rows
            lines.append(
                f"| {panel.folder} | {panel.status} | {rows.get('python', '-')} | {rows.get('agent03', '-')} "
                f"| {rows.get('agent04', '-')} | {panel.message[:120]} |"
            )
        lines.append("")
    if run.source:
        lines.append(f"Document table: `{pathlib.Path(run.document_dir) / 'final.csv'}`.")
    if master_table is not None:
        lines.append(f"Master table: `{master_table}` ({n_points} points). "
                     f"Dashboard: `{path_of('output') / CFG['naming']['dashboard']}`.")
    return "\n".join(lines)


class PublishResults(WorkflowStep):
    @handler
    async def handle(self, run: DocumentRun, ctx: WorkflowContext[DocumentRun, Message]) -> None:
        master_table, n_points = None, 0
        if run.source:
            with log_context(source=run.source, figure="-", step=self.id):
                log.info("Building review pages and CSV tables")
                write_document_final(run)
                save_json(
                    pathlib.Path(run.document_dir) / "discovery" / "run_summary.json",
                    {"source": run.source, "title": run.title, "run_id": run.run_id, "note": run.note,
                     "panels": [dataclasses.asdict(panel) for panel in run.panels]},
                )
                master_table, n_points = table.write_master_table(path_of("output"))
        await ctx.yield_output(Message(role="assistant", contents=[report(run, master_table, n_points)]))

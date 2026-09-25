"""Shared plumbing for the panel steps (2-6).

A panel step receives the ``DocumentRun``, works on ``run.panel`` and returns the route for
the next hop.  If an earlier step already finished the panel (skipped, failed, needs
review) the step passes the run through untouched, so the graph stays a straight line.
Any exception marks only the current panel as failed; the document keeps going.
"""

from __future__ import annotations

import json
import logging
import traceback

from agent_framework import Executor, WorkflowContext, handler

from src.workflow.agents import StepAgent
from src.workflow.console import log_context
from src.workflow.resources import StepConfig
from src.workflow.state import CONTINUE, FAILED, PENDING, DocumentRun, Panel

log = logging.getLogger("chart_extract")


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=1, default=float, ensure_ascii=False)


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


class WorkflowStep(Executor):
    """Base executor: step config from workflow.yaml plus its agent when the step has one."""

    def __init__(self, config: StepConfig):
        super().__init__(id=config.id)
        self.config = config
        self.agent = StepAgent(config.agent) if config.agent else None


class PanelStep(WorkflowStep):
    """A step of the per-panel pipeline (the panel steps (agent 01 → agent 04))."""

    async def run_panel(self, run: DocumentRun, panel: Panel) -> str:
        """Do the work for ``run.panel`` and return the route (default ``continue``)."""
        raise NotImplementedError

    def finish(self, run: DocumentRun, route: str) -> DocumentRun:
        """Build the outgoing message. Agent 04 overrides this to advance to the next panel."""
        return run.next(route)

    @handler
    async def handle(self, run: DocumentRun, ctx: WorkflowContext[DocumentRun]) -> None:
        panel = run.panel
        route = CONTINUE
        if panel.status == PENDING:
            with log_context(source=run.source, figure=panel.folder, step=self.id):
                log.info("%s", self.config.title)
                run.panel_dir.mkdir(parents=True, exist_ok=True)
                try:
                    route = await self.run_panel(run, panel) or CONTINUE
                except Exception as error:  # one broken panel must not stop the document
                    panel.status, panel.message = FAILED, f"{self.id}: {error}"
                    error_path = run.panel_dir / "error.txt"
                    error_path.write_text(traceback.format_exc(), encoding="utf-8")
                    log.exception("Failed: %s (details: %s)", error, error_path)
        await ctx.send_message(self.finish(run, route))

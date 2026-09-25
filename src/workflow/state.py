"""The message that travels along the workflow edges.

Think of ``DocumentRun`` as the record a Power Automate flow passes from action to action:
small, explicit, and rebuilt at every hop.  Large intermediate data (specs, extractions,
agent responses) is written to the panel folder by the step that creates it and read back
from disk by the next step, so every step is auditable and restartable.
"""

from __future__ import annotations

import dataclasses
import pathlib
from dataclasses import dataclass, field

# Routes a step can choose; workflow.yaml edges select on them with "when".
CONTINUE = "continue"
NEXT_PANEL = "next_panel"
ADJUST = "adjust"          # agent 02 corrected the structure: run Python (the Python extraction) again
PUBLISH = "publish"

# Panel statuses.
PENDING = "pending"   # still moving through the panel steps (agent 01 → agent 04)
DONE = "done"         # final check accepted the extraction
REVIEW = "review"     # finished, but a human must review it
FAILED = "failed"     # a step raised; see error.txt in the panel folder


@dataclass
class Panel:
    """One physical chart panel selected by agent 00."""

    folder: str                 # folder under artifacts/runs/<document>/, same as figure_id
    figure: dict                # the document reader figure candidate, updated with agent 00 number/caption
    discovery: dict             # agent 00 response for the whole figure
    panel: dict                 # agent 00 entry for this panel (bbox, inset, reason)
    figure_id: str              # canonical id from config.yaml naming, e.g. fig2, fig3a, figp7
    panel_id: str               # naming.panel_id: lowercase panel letter ("a"), "x" for a single-panel figure
    status: str = PENDING
    message: str = ""
    check_rounds: int = 0       # how many times agent 02 has checked a Python extraction
    extract_error: str = ""     # why the latest Python extraction failed ("" = it succeeded)
    needs_review: bool = False  # agent 02 was never satisfied; the final result must go to a human
    rows: dict = field(default_factory=dict)   # row counts per stage: python / candidate / final
    verdict: str = ""


@dataclass(frozen=True)
class DocumentRun:
    """One document moving through the graph. Build a new one per hop with ``next``."""

    source: str                 # original PDF / image path ("" when nothing was provided)
    document_dir: str           # artifacts/runs/<document>
    run_id: str
    title: str = ""
    panels: tuple[Panel, ...] = ()
    index: int = 0              # the panel the panel steps (agent 01 → agent 04) are working on
    route: str = CONTINUE
    note: str = ""              # message for the final report (e.g. why nothing ran)

    def next(self, route: str = CONTINUE, **changes) -> "DocumentRun":
        return dataclasses.replace(self, route=route, **changes)

    def next_panel(self) -> "DocumentRun":
        """Move to the following panel, or to publishing after the last one."""
        index = self.index + 1
        return self.next(NEXT_PANEL if index < len(self.panels) else PUBLISH, index=index)

    @property
    def panel(self) -> Panel | None:
        return self.panels[self.index] if self.index < len(self.panels) else None

    @property
    def panel_dir(self) -> pathlib.Path:
        return pathlib.Path(self.document_dir) / self.panel.folder

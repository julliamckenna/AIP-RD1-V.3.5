"""Agent 00 - find figures (agent, one call per figure candidate).

Input: the document inventory from python_read_document.  For every figure candidate the agent sees the figure image, its
page preview, the page's caption candidates and the sentences that reference the figure, and
answers with schemas/agent00_find_figures.schema.json.  Every panel it marks as a CO2 isotherm
becomes one ``Panel`` for the panel steps (agent 01 → agent 04).
Output: artifacts/runs/<document>/discovery/agent00_find_figures.json (every answer) and the panel
list. Each panel gets its own folder artifacts/runs/<document>/<figure_id>/ (fig2, fig3a, figp7).
"""

from __future__ import annotations

import hashlib
import json
import pathlib

from agent_framework import WorkflowContext, handler

from src.models import naming
from src.workflow.console import log_context
from src.workflow.resources import load_json
from src.workflow.state import NEXT_PANEL, PUBLISH, DocumentRun, Panel
from src.workflow.steps.base import WorkflowStep, log, read_json, save_json

CO2_SCOPE = "schemas/co2_scope.policy.json"


def request_fingerprint(agent, values: dict, images: list[str]) -> str:
    """Identify the exact Prompt Agent request so interrupted discovery can resume safely."""
    digest = hashlib.sha256()
    for value in (agent.agent_name, agent.agent_version, agent.prompt(values)):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    for image in images:
        path = pathlib.Path(image)
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_discovery_checkpoint(path: pathlib.Path) -> list[dict]:
    """Read a prior incremental checkpoint; malformed checkpoints are ignored."""
    if not path.exists():
        return []
    try:
        entries = read_json(path)
    except (OSError, json.JSONDecodeError) as error:
        log.warning("ignoring unreadable Agent 00 checkpoint %s: %s", path, error)
        return []
    if not isinstance(entries, list):
        log.warning("ignoring Agent 00 checkpoint %s: expected a JSON array", path)
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def save_discovery_checkpoint(path: pathlib.Path, answers: list[dict]) -> None:
    """Atomically save completed candidates so an interrupted run can resume."""
    temporary = path.with_name(f"{path.name}.tmp")
    save_json(temporary, answers)
    temporary.replace(path)


def document_context(inventory: dict) -> str:
    captions = "\n".join(
        f"- page {item['page']}, Fig. {item['num']}: {item['text'][:300]}" for item in inventory.get("captions", [])
    )
    return f"Title: {inventory.get('title') or 'unknown'}\nCaptions:\n{captions or '- none found'}"


def figure_images(figure: dict, inventory: dict) -> list[str]:
    images = [figure["file"]]
    page = next((p for p in inventory.get("pages", []) if p.get("page_number") == figure.get("page")), None)
    if page and page.get("preview") and pathlib.Path(page["preview"]).is_file():
        images.append(page["preview"])
    return images


def unique_id(figure_id: str, used: set) -> str:
    """fig2, then fig2_2, fig2_3 ... when two candidates resolve to the same figure panel."""
    candidate, n = figure_id, 1
    while candidate in used:
        n += 1
        candidate = f"{figure_id}_{n}"
    used.add(candidate)
    return candidate


def panels_for_figure(figure: dict, discovery: dict, used: set) -> list[Panel]:
    """Turn one relevant agent 00 answer into panels (one per CO2 axis frame)."""
    figure = dict(figure)
    if discovery.get("figure_number") not in (None, ""):
        figure["figure_number"] = str(discovery["figure_number"])
    if discovery.get("matched_caption"):
        figure["caption"] = discovery["matched_caption"]
    panels = []
    for entry in discovery.get("panels", []):
        if not entry.get("is_co2_isotherm"):
            continue
        letter = naming.panel_id(entry.get("panel"))
        figure_id = unique_id(naming.canonical_figure_id(figure, letter), used)
        panels.append(
            Panel(folder=figure_id, figure=figure, discovery=discovery, panel=entry, figure_id=figure_id, panel_id=letter)
        )
    return panels


class FindFigures(WorkflowStep):
    @handler
    async def handle(self, run: DocumentRun, ctx: WorkflowContext[DocumentRun]) -> None:
        document_dir = pathlib.Path(run.document_dir)
        inventory = read_json(document_dir / "discovery" / "python_inventory.json")
        output_path = document_dir / "discovery" / "agent00_find_figures.json"
        with log_context(source=run.source, figure="-", step=self.id):
            cached = {
                (entry.get("candidate_id"), entry.get("input_fingerprint")): entry
                for entry in load_discovery_checkpoint(output_path)
                if entry.get("candidate_id") and entry.get("input_fingerprint")
            }
        answers, panels, used = [], [], set()
        figures = inventory["figures"]
        total = len(figures)
        for index, figure in enumerate(figures, start=1):
            candidate_id = figure.get("source_id") or pathlib.Path(figure["file"]).name
            caption = "\n".join(
                f"- Fig. {item['figure_number']}: {item['caption']}" for item in figure.get("caption_candidates", [])
            ) or (figure.get("caption") or "none found")
            values = {
                "candidate_id": candidate_id,
                "document_context": document_context(inventory),
                "caption": caption,
                "references": "\n".join(f"- {s}" for s in figure.get("references", [])) or "none found",
                "co2_scope": load_json(CO2_SCOPE),
            }
            images = figure_images(figure, inventory)
            fingerprint = request_fingerprint(self.agent, values, images)
            cached_entry = cached.get((candidate_id, fingerprint))
            cached_answer = cached_entry.get("answer") if cached_entry else None
            cache_errors = list(self.agent.validator.iter_errors(cached_answer)) if cached_answer is not None else [None]

            with log_context(source=run.source, figure=candidate_id, step=self.id):
                if cached_entry and not cache_errors:
                    answer = cached_answer
                    provenance = cached_entry.get("provenance") or {}
                    log.info("Candidate %d/%d: resumed cached answer", index, total)
                else:
                    log.info("Candidate %d/%d: checking %d image(s)", index, total, len(images))
                    answer = await self.agent.ask(values, images=images)
                    provenance = dict(self.agent.last_provenance)
                log.info("CO2 figure: %s", "yes" if answer.get("figure_is_relevant") else "no")

            answers.append({
                "candidate_id": candidate_id,
                "figure": figure,
                "answer": answer,
                "provenance": provenance,
                "input_fingerprint": fingerprint,
            })
            save_discovery_checkpoint(output_path, answers)
            if answer.get("figure_is_relevant"):
                panels.extend(panels_for_figure(figure, answer, used))
        save_discovery_checkpoint(output_path, answers)
        note = "" if panels else "No CO2 isotherm panels were found."
        await ctx.send_message(run.next(NEXT_PANEL if panels else PUBLISH, panels=tuple(panels), index=0, note=note))

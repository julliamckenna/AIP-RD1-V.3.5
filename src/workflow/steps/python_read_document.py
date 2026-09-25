"""The document reader - read the document (Python).

Input: the document given to ``python main.py --run``.  One request = one document: a file in
data/input (``paper_c.pdf``), a local path, or an https URL (an attached PDF / image file in the
request message is accepted too).
Output: the figure inventory (every raster/vector figure with caption candidates and page
previews) in artifacts/runs/<document>/discovery/, and a new ``DocumentRun``.
"""

from __future__ import annotations

import base64
import pathlib
import re
import shutil
import urllib.parse

import httpx
from agent_framework import Message, WorkflowContext, handler

from src.models import naming
from src.settings import CFG, ROOT, path_of
from src.tools import pdf_reader
from src.workflow.console import log_context
from src.workflow.state import CONTINUE, PUBLISH, DocumentRun
from src.workflow.steps.base import WorkflowStep, log, save_json

USAGE = (
    "Send one document per request: attach a PDF or chart image, or send the name of a file in "
    f"{CFG['folders']['input']}/ (for example `paper_c.pdf`), a local path, or an https URL."
)
MEDIA_SUFFIX = {"application/pdf": ".pdf", "image/png": ".png", "image/jpeg": ".jpg", "image/tiff": ".tif"}


def _save_attachment(content, input_dir: pathlib.Path) -> pathlib.Path | None:
    """Store an attached file (data URI) in the input folder and return its path."""
    uri = getattr(content, "uri", None) or ""
    media_type = getattr(content, "media_type", None) or ""
    if not uri.startswith("data:") or media_type not in MEDIA_SUFFIX:
        return None
    name = (getattr(content, "additional_properties", None) or {}).get("filename") or f"attachment{MEDIA_SUFFIX[media_type]}"
    path = input_dir / pathlib.Path(name).name
    path.write_bytes(base64.b64decode(uri.split(",", 1)[1]))
    return path


def _resolve_text(text: str, input_dir: pathlib.Path) -> pathlib.Path | None:
    """Turn the request text into a local file: input-folder name, path, or downloaded URL."""
    text = text.strip().strip("`\"'")
    if not text:
        return None
    if re.match(r"^https?://", text):
        name = pathlib.Path(urllib.parse.urlparse(text).path).name or "download.pdf"
        path = input_dir / name
        with httpx.stream("GET", text, follow_redirects=True, timeout=120) as response:
            response.raise_for_status()
            with open(path, "wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
        return path
    for candidate in (input_dir / text, ROOT / text, pathlib.Path(text)):
        if candidate.is_file():
            return candidate
    return None


def find_source(messages: list[Message]) -> pathlib.Path | None:
    input_dir = path_of("input")
    input_dir.mkdir(parents=True, exist_ok=True)
    for message in reversed(messages):
        for content in message.contents or []:
            saved = _save_attachment(content, input_dir)
            if saved:
                return saved
        found = _resolve_text(message.text or "", input_dir)
        if found:
            return found
    return None


def read_inventory(source: pathlib.Path, document_dir: pathlib.Path) -> dict:
    """Figure inventory for a PDF, or a one-figure inventory for a chart image."""
    inventory_dir = document_dir / "discovery" / "python_inventory"
    inventory_dir.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() != ".pdf":
        copy = inventory_dir / ("figure" + source.suffix.lower())
        shutil.copy(source, copy)
        return {
            "pdf": str(source), "title": source.stem, "producer": "image", "page_count": 1,
            "pages": [], "captions": [], "document_text": "",
            "figures": [{"page": 1, "kind": "image", "file": str(copy), "source_id": "image",
                         "figure_number": None, "caption": None, "caption_candidates": [], "references": []}],
        }
    pdf = CFG["pdf"]
    return pdf_reader.inspect(
        str(source), inventory_dir,
        min_px=pdf["min_figure_pixels"], render_dpi=pdf["vector_render_dpi"], discovery_dpi=pdf["discovery_page_dpi"],
    )


class ReadDocument(WorkflowStep):
    @handler
    async def handle(self, messages: list[Message], ctx: WorkflowContext[DocumentRun]) -> None:
        source = find_source(messages)
        if source is None or not naming.accepts_input(source):
            note = USAGE if source is None else f"`{source.name}` is not an accepted input type. {USAGE}"
            await ctx.send_message(DocumentRun(source="", document_dir="", run_id=naming.run_id(), route=PUBLISH, note=note))
            return
        document_dir = path_of("output") / naming.document_name(source)
        document_dir.mkdir(parents=True, exist_ok=True)
        with log_context(source=str(source), figure="-", step=self.id):
            inventory = read_inventory(source, document_dir)
            save_json(document_dir / "discovery" / "python_inventory.json", {k: v for k, v in inventory.items() if k != "document_text"})
            log.info("Found %d figure candidate(s)", len(inventory["figures"]))
        run = DocumentRun(
            source=str(source), document_dir=str(document_dir), run_id=naming.run_id(),
            title=inventory.get("title") or source.stem, route=CONTINUE,
            note="" if inventory["figures"] else "No figure candidates were found in the document.",
        )
        await ctx.send_message(run if inventory["figures"] else run.next(PUBLISH))

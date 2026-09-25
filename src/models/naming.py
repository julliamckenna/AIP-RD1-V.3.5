import datetime, pathlib, uuid, fnmatch, re
from src.settings import CFG

N = CFG["naming"]
I = CFG["input"]

# The three extraction stages, named after the step that writes them. Every stage file is
# "<stage>_<part>": python_points.csv, agent03_overlay.png, agent04_metadata.json ...
STAGES = ("python", "agent03", "agent04")


def stage_file(stage: str, part: str) -> str:
    """File name of one stage artifact, e.g. stage_file("agent04", "points.csv") -> agent04_points.csv."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; stages are {STAGES}")
    return f"{stage}_{part}"


def panel_id(letter) -> str:
    """Panel id: the lowercase panel letter from agent 00; 'x' when the figure has a single panel."""
    text = re.sub(r"^panel[_-]*", "", str(letter or "").strip(), flags=re.I).lower()
    return text or "x"


def series_label(label) -> str:
    """Canonical series label: single spaces, CO₂ -> CO2, temperature written '273K' (no space).

    Agents write 'CO2 273 K', 'CO₂ 273K' or 'co2  273 k'; code matches labels only after this.
    """
    text = " ".join(str(label or "").replace("₂", "2").replace("₄", "4").split())
    return re.sub(r"(\d+(?:\.\d+)?)\s*[kK]\b", r"\1K", text)


def figure_id(figure_number, page, panel_letter=""):
    """fig2 · fig3a · figp7 (unnumbered, page 7) - panel letter left out when the figure has one panel ('x' or empty)."""
    raw_figure = str(figure_number or "").strip()
    number = re.search(r"\d+", raw_figure)
    panel = re.sub(r"^panel[_-]*", "", str(panel_letter or ""), flags=re.I).lower()
    panel = "" if panel == "x" else panel
    if number:
        return N["figure_id"].format(figure=int(number.group()), panel=panel, page=page)
    return N["unnumbered_figure"].format(page=int(page), panel=panel)


def canonical_figure_id(figure, panel_id=""):
    """Apply the configured figure-ID pattern to discovered figure metadata."""
    if not isinstance(figure, dict):
        raise TypeError("figure metadata must be a mapping")
    if figure.get("page") in (None, ""):
        raise ValueError("figure metadata needs a page for canonical naming")
    return figure_id(figure.get("figure_number"), figure["page"], panel_id)


def run_id():
    return N["run_id"].format(date=datetime.datetime.now(), rand6=uuid.uuid4().hex[:6], rand4=uuid.uuid4().hex[:4])


def review_file(document, figure):
    """The file the review page downloads for artifacts/runs/<document>/<figure>/."""
    return N["review_file"].format(document=document, figure=figure)


def document_name(input_path):
    p = pathlib.Path(input_path)
    return I["document_name"].format(stem=p.stem, name=p.name, ext=p.suffix.lstrip("."))


def accepts_input(path):
    p = pathlib.Path(path)
    return p.suffix.lower() in [e.lower() for e in I["accept"]] and fnmatch.fnmatch(p.name, I.get("pattern", "*"))

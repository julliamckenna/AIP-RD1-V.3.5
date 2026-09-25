"""Turn workflow.yaml into a Microsoft Agent Framework workflow.

    workflow = build_workflow()            # the graph (python main.py --graph prints it)
    agent = workflow.as_agent()            # what python main.py --run runs

Adding a step = write src/workflow/steps/<id>.py, list it in STEP_CLASSES, add it and its
edges to workflow.yaml (plus prompt / schema / example when it is an agent step).
"""

from __future__ import annotations

from agent_framework import Workflow, WorkflowBuilder

from src.workflow.resources import load_workflow
from src.workflow.steps.python_read_document import ReadDocument
from src.workflow.steps.agent00_find_figures import FindFigures
from src.workflow.steps.agent01_read_chart import ReadChart
from src.workflow.steps.python_extract_points import ExtractPoints
from src.workflow.steps.agent02_check_extraction import CheckExtraction
from src.workflow.steps.agent03_review_points import ReviewPoints
from src.workflow.steps.agent04_final_check import FinalCheck
from src.workflow.steps.python_publish_results import PublishResults

STEP_CLASSES = {
    "python_read_document": ReadDocument,
    "agent00_find_figures": FindFigures,
    "agent01_read_chart": ReadChart,
    "python_extract_points": ExtractPoints,
    "agent02_check_extraction": CheckExtraction,
    "agent03_review_points": ReviewPoints,
    "agent04_final_check": FinalCheck,
    "python_publish_results": PublishResults,
}


def _route_is(route: str):
    def condition(run) -> bool:
        return getattr(run, "route", None) == route

    condition.__name__ = f"route_is_{route}"
    return condition


def build_workflow() -> Workflow:
    config = load_workflow()
    missing = set(config.steps) ^ set(STEP_CLASSES)
    if missing:
        raise ValueError(f"workflow.yaml steps and STEP_CLASSES differ: {sorted(missing)}")
    executors = {step_id: STEP_CLASSES[step_id](step) for step_id, step in config.steps.items()}
    builder = WorkflowBuilder(
        max_iterations=config.max_workflow_steps,  # Agent Framework's default (100) stops a document after ~8 panels
        name=config.name,
        description=config.description,
        start_executor=executors[config.start],
        output_from=[executors[config.output]],
    )
    for edge in config.edges:
        condition = _route_is(edge.when) if edge.when else None
        builder.add_edge(executors[edge.source], executors[edge.target], condition=condition)
    return builder.build()


def mermaid() -> str:
    """The workflow graph as a labelled Mermaid flowchart, straight from workflow.yaml (main.py --graph)."""
    config = load_workflow()
    lines = ["flowchart TD"]
    for step in config.steps.values():
        kind = f"agent: {step.agent.name}" if step.agent else "python"
        lines.append(f'  {step.id}["{step.id}<br/><small>{kind}</small>"]')
    for edge in config.edges:
        arrow = f"-- {edge.when} -->" if edge.when else "-->"
        lines.append(f"  {edge.source} {arrow} {edge.target}")
    return "\n".join(lines)

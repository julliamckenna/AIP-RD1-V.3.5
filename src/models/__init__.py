"""Naming helpers shared by the workflow steps (patterns come from config.yaml)."""

from .naming import (
    accepts_input,
    canonical_figure_id,
    document_name,
    figure_id,
    review_file,
    run_id,
)

__all__ = [
    "accepts_input",
    "canonical_figure_id",
    "document_name",
    "figure_id",
    "review_file",
    "run_id",
]

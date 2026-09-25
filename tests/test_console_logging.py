"""Terminal presentation checks using in-memory streams and no model calls."""

import asyncio
import io
import logging
from types import SimpleNamespace

from src.workflow import console
from src.workflow.console import ProgressFilter, ProgressFormatter, log_context
from src.workflow.rerun import _run_stage
from src.workflow.state import DocumentRun, Panel
from src.workflow.steps.python_publish_results import report


def _logger():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(ProgressFilter())
    handler.setFormatter(ProgressFormatter())
    logger = logging.Logger("chart_extract", logging.INFO)
    logger.addHandler(handler)
    return logger, stream


def test_context_fields_and_nested_stage_reset():
    logger, stream = _logger()
    with log_context(source=r"C:\data\calf20.png", figure="fig1b", step="agent03_review_points"):
        logger.info("Reviewing points")
        with log_context(step="python_extract_points"):
            logger.info("Rebuilding\n  points")
        logger.info("Review complete")
    logger.info("Idle")
    lines = stream.getvalue().splitlines()
    assert lines[0].split(" | ")[1:] == ["calf20.png", "fig1b", "agent 03", "Reviewing points"]
    assert lines[1].endswith(" | python | Rebuilding points")
    assert lines[2].endswith(" | agent 03 | Review complete")
    assert lines[3].endswith(" | - | - | - | Idle")


def test_concurrent_documents_keep_their_own_context():
    logger, stream = _logger()

    async def worker(source, figure):
        with log_context(source=source, figure=figure, step="agent04_final_check"):
            await asyncio.sleep(0)
            logger.info("Complete")

    async def run():
        await asyncio.gather(worker("first.pdf", "fig2a"), worker("second.pdf", "fig3b"))

    asyncio.run(run())
    assert {line.split(" | ", 1)[1] for line in stream.getvalue().splitlines()} == {
        "first.pdf | fig2a | agent 04 | Complete",
        "second.pdf | fig3b | agent 04 | Complete",
    }


def test_console_suppresses_sdk_info_but_keeps_warnings():
    progress_filter = ProgressFilter()
    for name in ("httpx", "azure.core.pipeline.policies.http_logging_policy", "agent_framework"):
        info = logging.LogRecord(name, logging.INFO, "", 0, "raw request body", (), None)
        warning = logging.LogRecord(name, logging.WARNING, "", 0, "retry needed", (), None)
        assert not progress_filter.filter(info)
        assert progress_filter.filter(warning)
    assert progress_filter.filter(logging.LogRecord("chart_extract", logging.INFO, "", 0, "Ready", (), None))


def test_short_error_keeps_traceback_available_for_file_handlers():
    try:
        raise ValueError("invalid response")
    except ValueError:
        import sys
        record = logging.LogRecord("chart_extract", logging.ERROR, "", 0,
                                   "Failed (details: error.txt)", (), sys.exc_info())
    line = ProgressFormatter().format(record)
    assert "ERROR: Failed (details: error.txt): invalid response" in line
    assert "\n" not in line
    assert record.exc_text is None
    assert "Traceback" in logging.Formatter().format(record)


def test_configure_preserves_file_handler_and_replaces_console(monkeypatch):
    root = logging.Logger("isolated_root")
    root.setLevel(logging.DEBUG)
    original_console = logging.StreamHandler(io.StringIO())
    file_handler = logging.FileHandler("not-opened.log", delay=True)
    root.addHandler(original_console)
    root.addHandler(file_handler)
    bokeh = logging.Logger("isolated_bokeh")
    monkeypatch.setattr(console.logging, "getLogger", lambda name=None: bokeh if name == "bokeh" else root)
    try:
        first = console.configure_console_logging(io.StringIO())
        second = console.configure_console_logging(io.StringIO())
        assert root.handlers == [file_handler, second]
        assert first not in root.handlers
        assert original_console not in root.handlers
        assert root.level == logging.DEBUG
        assert bokeh.level == logging.WARNING
    finally:
        file_handler.close()


def test_direct_rerun_uses_stage_context_and_keeps_jobs_table(monkeypatch):
    logger, stream = _logger()
    monkeypatch.setattr("src.workflow.rerun.log", logger)
    panel = Panel("fig1b", {}, {}, {}, "fig1b", "b", status="review",
                  rows={"python": 117, "agent03": 104, "agent04": 105}, message="Check overlap")
    run = DocumentRun("calf20.png", "runs/calf20", "run-1", panels=(panel,))
    calls = []

    async def run_panel(actual_run, actual_panel):
        calls.append((actual_run, actual_panel))
        logger.info("Completed")
        return "continue"

    step = SimpleNamespace(id="agent04_final_check", config=SimpleNamespace(title="Final QA"), run_panel=run_panel)
    assert asyncio.run(_run_stage(step, run, panel)) == "continue"
    assert calls == [(run, panel)]
    assert all(" | calf20.png | fig1b | agent 04 | " in line for line in stream.getvalue().splitlines())
    assert [line for line in report(run, None, 0).splitlines() if line.startswith("|")] == [
        "| panel | status | python | agent03 | agent04 | note |",
        "|---|---|---|---|---|---|",
        "| fig1b | review | 117 | 104 | 105 | Check overlap |",
    ]

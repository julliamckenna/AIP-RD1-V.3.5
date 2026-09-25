"""One place for settings.  from src.settings import CFG, ROOT, path_of
   CFG   -> config.yaml   (folders, naming patterns, deterministic tuning)
   ROOT  -> the project folder (every path in config.yaml and workflow.yaml is relative to it)"""
import os, pathlib, yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def path_of(folder_key):
    """Absolute path of a folder in config.yaml; CHART_EXTRACT_<KEY>_DIR (e.g. CHART_EXTRACT_OUTPUT_DIR) overrides it."""
    override = os.environ.get(f"CHART_EXTRACT_{folder_key.upper()}_DIR")
    return pathlib.Path(override).expanduser() if override else ROOT / CFG["folders"][folder_key]

"""Rebuild the master table + dashboard from what is on disk (after moving review files into the reviews folder).
   python scripts/rebuild_table.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.settings import path_of
from src.tools import table
out, n = table.write_master_table(path_of("output"))
print(f"{n} points -> {out}")

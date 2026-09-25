# One-time setup on Windows (run from the project folder in a VS Code terminal):  .\scripts\setup.ps1
# Needs Python 3.11+.
$ErrorActionPreference = "Stop"
python --version
if (-not (Test-Path ".venv")) { python -m venv .venv }
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
New-Item -ItemType Directory -Force data\input, artifacts\runs, artifacts\reviews | Out-Null
if (-not (Test-Path ".env")) { Copy-Item .env.example .env; Write-Host "`n>> .env created - fill in the Foundry endpoint and per-agent name/model values." }
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe main.py --graph

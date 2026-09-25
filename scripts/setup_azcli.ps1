param(
    [string]$AzureCliVersion = "2.89.1"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$cliRoot = Join-Path $projectRoot ".azcli"
$cliPython = Join-Path $cliRoot "Scripts\python.exe"
$cliConfig = Join-Path $cliRoot "pyvenv.cfg"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Create the project .venv before installing Azure CLI."
}

$needsVenv = -not (Test-Path -LiteralPath $cliPython) -or -not (Test-Path -LiteralPath $cliConfig)
if (-not $needsVenv) {
    & $cliPython -m pip --version | Out-Null
    $needsVenv = $LASTEXITCODE -ne 0
}

if ($needsVenv) {
    & $python -m venv --clear $cliRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the .azcli virtual environment."
    }
}

& $cliPython -m pip install --upgrade "azure-cli==$AzureCliVersion"
if ($LASTEXITCODE -ne 0) {
    throw "Azure CLI installation failed."
}

Write-Output "Azure CLI $AzureCliVersion installed in .azcli. Next run:"
Write-Output ".\.azcli\Scripts\az.bat login"

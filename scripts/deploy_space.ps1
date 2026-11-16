# Sync the web app + current ONNX bundle to the private Hugging Face Space, wait for the build, smoke-test it.
#   $env:HF_TOKEN must be set.   .\scripts\deploy_space.ps1 [-Space AnupamZZZZ/cervical-cell-demo] [-NoSmoke]
param([string]$Space = "AnupamZZZZ/cervical-cell-demo", [switch]$NoSmoke)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$pyArgs = @("$root\scripts\deploy_space.py", "--space", $Space, "--wait")
if (-not $NoSmoke) { $pyArgs += "--smoke" }
& $py @args
exit $LASTEXITCODE

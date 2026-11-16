# One command to swap the web app's segmenter: re-export ONNX -> parity test -> redeploy the Space.
#   .\scripts\swap_segmenter.ps1                                  (re-export what config.yaml names)
#   .\scripts\swap_segmenter.ps1 -Weights models/seg/V4/best.pt   [-Variant V1]  [-NoDeploy]
param([string]$Weights = "", [string]$Variant = "", [switch]$NoDeploy)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$pyArgs = @("$root\scripts\swap_segmenter.py")
if ($Weights) { $pyArgs += @("--weights", $Weights) }
if ($Variant) { $pyArgs += @("--variant", $Variant) }
if ($NoDeploy) { $pyArgs += "--no-deploy" }
& $py @args
exit $LASTEXITCODE

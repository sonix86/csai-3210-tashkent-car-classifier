# Use the project's Python and local model caches without activating a venv.
$projectPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    Write-Error 'Python environment missing. Follow README Setup to create .venv first.'
    exit 1
}
$env:TORCH_HOME = Join-Path $PSScriptRoot '.cache/torch'
$env:YOLO_CONFIG_DIR = Join-Path $PSScriptRoot '.cache/ultralytics'
Push-Location -LiteralPath $PSScriptRoot
try {
    & $projectPython (Join-Path $PSScriptRoot 'project.py') @args
    $projectExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $projectExitCode

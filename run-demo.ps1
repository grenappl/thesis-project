<#
.SYNOPSIS
    Starts everything needed to try the Pipeline 2 (demo feature-extraction)
    visualizer: the FastAPI backend and the Next.js frontend.

.DESCRIPTION
    Feature extraction itself (api/pipelines/demo/) is not a standing
    process - the backend shells out to WSL2 per request instead (see
    PIPELINE_SETUP.md#why-pipeline-2-needs-wsl2 for why). So "running the
    pipeline" here means: verify WSL2 and the model checkpoints it needs
    are in place, then start the two real servers.

    Both servers run with -NoNewWindow, so everything stays inside this
    terminal (e.g. VS Code's integrated terminal) instead of popping open
    external console windows. Their output interleaves live in this window.
    Press Ctrl+C to stop both.
#>

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

function Test-ModelFile {
    param([string]$RelativePath)
    $path = Join-Path $root $RelativePath
    if (Test-Path $path) {
        Write-Host "  [ok] $RelativePath" -ForegroundColor Green
        return $true
    }
    Write-Host "  [missing] $RelativePath" -ForegroundColor Yellow
    return $false
}

function Stop-ProcessTree {
    param([int]$ProcessId)
    Get-CimInstance Win32_Process -Filter "ParentProcessId=$ProcessId" -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-ProcessTree -ProcessId $_.ProcessId }
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
}

Write-Host "== Preflight checks ==" -ForegroundColor Cyan

# WSL distro - wsl.exe emits UTF-16 with embedded nulls that PowerShell
# needs stripped before string matching works reliably.
$distroName = "Ubuntu-24.04"
$distros = (wsl.exe -l -q 2>$null) | ForEach-Object { ($_ -replace "`0", "").Trim() }
if ($distros -contains $distroName) {
    Write-Host "  [ok] WSL distro '$distroName' found" -ForegroundColor Green
} else {
    Write-Host "  [missing] WSL distro '$distroName' not found - Pipeline 2 requests will fail." -ForegroundColor Yellow
    Write-Host "            See PIPELINE_SETUP.md for one-time WSL2 setup." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "  Model checkpoints (feature_extraction.md has download commands):"
Test-ModelFile "models\panns\Cnn14_mAP=0.431.pth" | Out-Null
Test-ModelFile "models\vggish_ridge\valence.joblib" | Out-Null
Test-ModelFile "models\vggish_ridge\acousticness.joblib" | Out-Null
Test-ModelFile "models\vggish_ridge\instrumentalness.joblib" | Out-Null
Test-ModelFile "models\vggish_ridge\danceability.joblib" | Out-Null
Test-ModelFile "models\vggish_ridge\energy.joblib" | Out-Null
Test-ModelFile "models\vggish_ridge\speechiness.joblib" | Out-Null
Test-ModelFile "models\vggish_ridge\loudness.joblib" | Out-Null

Write-Host ""
Write-Host "== Starting servers (this terminal, Ctrl+C to stop both) ==" -ForegroundColor Cyan
Write-Host ""

$backend = $null
$frontend = $null

try {
    $backend = Start-Process -FilePath "uv" `
        -ArgumentList "run", "uvicorn", "api.main:app", "--reload", "--port", "8010" `
        -WorkingDirectory $root `
        -NoNewWindow -PassThru

    $frontend = Start-Process -FilePath "npm.cmd" `
        -ArgumentList "run", "dev" `
        -WorkingDirectory (Join-Path $root "app") `
        -NoNewWindow -PassThru

    Write-Host ""
    Write-Host "Visualizer: http://localhost:3005/test-feature-extraction" -ForegroundColor Green
    Write-Host "API docs:   http://localhost:8010/docs" -ForegroundColor Green
    Write-Host ""

    Wait-Process -Id $backend.Id, $frontend.Id
}
finally {
    Write-Host ""
    Write-Host "Stopping servers..." -ForegroundColor Cyan
    if ($backend) { Stop-ProcessTree -ProcessId $backend.Id }
    if ($frontend) { Stop-ProcessTree -ProcessId $frontend.Id }
    Write-Host "Done."
}

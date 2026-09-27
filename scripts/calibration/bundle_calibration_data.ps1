<#
.SYNOPSIS
    Copies everything the GPU PC needs for calibration into one folder
    (e.g. a USB drive), with a checksum manifest to verify it on arrival.

.DESCRIPTION
    Copies the project folder itself (code, uncommitted changes included) plus
    the gitignored data and models, and the Spotify dataset CSV as
    data\songs.csv. Skips things the other machine rebuilds or doesn't need:
    .venv, node_modules, .next, caches, the unused Essentia models, and (unless
    -IncludeExtraEmbeddings) the MERT / mel-statistics embeddings.

    Writes data\bundle_manifest.json (path, size, SHA-256 of every data and
    model file); scripts/calibration/check_environment.py checks it.

.EXAMPLE
    .\scripts\calibration\bundle_calibration_data.ps1 -Destination E:\thesis-project
#>
param(
    [Parameter(Mandatory = $true)][string]$Destination,
    [string]$SongsCsv = "C:\Users\User\Downloads\songs(1).csv",
    [switch]$IncludeExtraEmbeddings,
    [switch]$DryRun  # list what would be copied, copy nothing
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

if (-not (Test-Path $SongsCsv)) { throw "Songs CSV not found at '$SongsCsv' - pass -SongsCsv <path>." }
New-Item -ItemType Directory -Force $Destination | Out-Null

$excludeDirs = @(
    (Join-Path $root ".venv"), (Join-Path $root "app\node_modules"), (Join-Path $root "app\.next"),
    (Join-Path $root "models\essentia"), (Join-Path $root "api\pipelines\demo\_uploads"),
    "__pycache__", ".pytest_cache"
)
$excludeFiles = @()
if (-not $IncludeExtraEmbeddings) {
    $excludeFiles += (Join-Path $root "data\kaggle_embeddings\mert_embeddings.npz")
    $excludeFiles += (Join-Path $root "data\kaggle_embeddings\mel_stats_embeddings.npz")
}

Write-Host "Copying project from $root to $Destination ..." -ForegroundColor Cyan
$roboArgs = @($root, $Destination, "/E", "/NFL", "/NDL", "/NP", "/R:2", "/W:2", "/XD") + $excludeDirs
if ($excludeFiles.Count) { $roboArgs += @("/XF") + $excludeFiles }
if ($DryRun) { $roboArgs += "/L" }
robocopy @roboArgs | Out-Host
if ($LASTEXITCODE -ge 8) { throw "robocopy failed (exit code $LASTEXITCODE)" }

if ($DryRun) { Write-Host "Dry run: nothing copied." -ForegroundColor Yellow; exit 0 }

Write-Host "Copying songs CSV -> data\songs.csv ..." -ForegroundColor Cyan
Copy-Item $SongsCsv (Join-Path $Destination "data\songs.csv") -Force

Write-Host "Hashing data and model files (a minute or two) ..." -ForegroundColor Cyan
$entries = foreach ($dir in @("data", "models")) {
    Get-ChildItem (Join-Path $Destination $dir) -Recurse -File |
        Where-Object { $_.Name -ne "bundle_manifest.json" -and $_.DirectoryName -notmatch "calibration_logs" } |
        ForEach-Object {
            [pscustomobject]@{
                path   = $_.FullName.Substring($Destination.TrimEnd('\').Length + 1).Replace('\', '/')
                size   = $_.Length
                sha256 = (Get-FileHash $_.FullName -Algorithm SHA256).Hash
            }
        }
}
$entries | ConvertTo-Json -Depth 2 | Out-File (Join-Path $Destination "data\bundle_manifest.json") -Encoding utf8

$totalGb = [math]::Round((($entries | Measure-Object size -Sum).Sum) / 1GB, 2)
Write-Host "Done: $($entries.Count) data/model files ($totalGb GB) + project code in $Destination" -ForegroundColor Green
Write-Host "Next, on the GPU PC: follow docs\gpu_calibration_setup.md"

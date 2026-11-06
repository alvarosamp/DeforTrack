param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$Images,
    [Parameter(Mandatory=$true)][string]$ModelsJson,
    [string]$Output = ".\results\replacement_test_892\computational_profile\standardized_profile_892.csv"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$script = Join-Path $PSScriptRoot "profile_model_local.py"
$Output = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
New-Item -ItemType Directory -Path (Split-Path -Parent $Output) -Force | Out-Null
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $Images)) { throw "Image directory not found: $Images" }
if (-not (Test-Path -LiteralPath $ModelsJson)) { throw "Model manifest not found: $ModelsJson" }
$models = Get-Content -LiteralPath $ModelsJson -Raw | ConvertFrom-Json

$completed = @{}
if (Test-Path -LiteralPath $Output) {
    foreach ($row in (Import-Csv -LiteralPath $Output)) {
        if ($completed.ContainsKey($row.model)) {
            throw "Duplicate model in existing profile output: $($row.model)"
        }
        $completed[$row.model] = $true
    }
    Write-Host "Resuming profile run with $($completed.Count) completed models."
}

foreach ($model in $models) {
    if ($completed.ContainsKey($model.name)) {
        Write-Host "Skipping completed model $($model.name)."
        continue
    }
    if (-not (Test-Path -LiteralPath $model.path)) { throw "Checkpoint not found: $($model.path)" }
    Write-Host "Profiling $($model.name) on all 892 images..."
    & $Python $script --family $model.family --model-name $model.name --model-path $model.path --images $Images --output $Output --warmup 892 --repetitions 3
    if ($LASTEXITCODE -ne 0) {
        throw "Profiling failed for $($model.name) with exit code $LASTEXITCODE"
    }
}

Write-Host "Completed standardized profiles: $Output"

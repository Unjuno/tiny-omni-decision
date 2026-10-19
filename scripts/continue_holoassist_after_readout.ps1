$ErrorActionPreference = 'Stop'

$Repo = 'C:\Users\junny\.codex\worktrees\r2-holoassist-textfree\tiny-omni-decision'
$Base = 'C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-recovery-02'
$VideoRun = Join-Path $Base 'video-features-v1-attempt-03'
$StatePath = Join-Path $VideoRun 'pipeline-state.json'
$PackedOut = Join-Path $Base 'packed-cache-v1'
$FitOut = Join-Path $Base 'readout-v1-attempt-01'
$Output = Join-Path $Base 'readout-v1-intervention-attempt-01'
$WatchDir = Join-Path $Base 'readout-v1-intervention-watch-attempt-01'
$WatchState = Join-Path $WatchDir 'watch-state.json'
$LogPath = Join-Path $WatchDir 'watch.log'
$ParentPython = 'C:\Users\junny\AppData\Local\Programs\Python\Python311\python.exe'
$MlPython = 'C:\CodexArtifacts\venvs\embeddinggemma2-v518\Scripts\python.exe'
$Guard = Join-Path $Repo 'scripts\run_with_hotgpu_guard.py'
$Evaluate = Join-Path $Repo 'scripts\evaluate_holoassist_temporal_interventions.py'

if (Test-Path -LiteralPath $WatchDir) { throw "Refusing existing watcher path: $WatchDir" }
if (Test-Path -LiteralPath $Output) { throw "Refusing existing result path: $Output" }
New-Item -ItemType Directory -Path $WatchDir | Out-Null
$env:PYTHONPATH = "$Repo;$Repo\src"
$env:PYTHONUTF8 = '1'

function Write-WatchState([string]$Status, [object]$Details) {
    $payload = [pscustomobject]@{
        experiment_id = 'holoassist-temporal-av-interventions-v1'
        status = $Status
        updated_at_utc = (Get-Date).ToUniversalTime().ToString('o')
        details = $Details
    }
    $temporary = "$WatchState.tmp"
    $payload | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $WatchState -Force
}

try {
    Write-WatchState 'waiting_for_base_run' @{base_pipeline_state=$StatePath; output=$Output}
    Add-Content -LiteralPath $LogPath -Value "[$(Get-Date -Format o)] waiting for fixed three-arm readout" -Encoding utf8
    while ($true) {
        if (Test-Path -LiteralPath $StatePath) {
            $stateValue = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
            $stateRows = @($stateValue)
            $terminal = $stateRows | Where-Object { $_.stage -eq 'pipeline' } | Select-Object -Last 1
            if ($terminal -and $terminal.status -eq 'failed') {
                throw "Base HoloAssist pipeline failed: $($terminal.details.error)"
            }
            if ($terminal -and $terminal.status -eq 'complete') { break }
        }
        Start-Sleep -Seconds 60
    }

    $runMetadataPath = Join-Path $FitOut 'run-metadata.json'
    if (-not (Test-Path -LiteralPath $runMetadataPath)) { throw 'Base run metadata is missing after pipeline completion.' }
    $runMetadata = Get-Content -LiteralPath $runMetadataPath -Raw | ConvertFrom-Json
    if ($runMetadata.status -ne 'complete' -or $runMetadata.sealed_audit_loaded -ne $false) {
        throw 'Base readout is not complete or sealed-audit isolation is not explicit.'
    }
    $jointResult = $runMetadata.results | Where-Object { $_.arm -eq 'joint' } | Select-Object -First 1
    if (-not $jointResult) { throw 'Completed base readout has no joint-arm result.' }
    if (Test-Path -LiteralPath $Output) { throw "Refusing existing result path: $Output" }
    Write-WatchState 'running_fixed_validation_interventions' @{base_readout=$FitOut; output=$Output}
    Add-Content -LiteralPath $LogPath -Value "[$(Get-Date -Format o)] starting fixed validation interventions" -Encoding utf8
    & $ParentPython $Guard --max-wait-seconds 21600 --reserve-mib 3200 --block-if-free-mib 14000 --min-child-free-mib 8000 -- `
        $MlPython $Evaluate --run-dir $FitOut --train-features (Join-Path $PackedOut 'train.npz') `
        --validation-features (Join-Path $PackedOut 'validation.npz') `
        --feature-receipt (Join-Path $PackedOut 'receipt.json') --output-dir $Output --device cuda --seed 17 `
        2>&1 | Tee-Object -FilePath $LogPath -Append
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) { throw "Fixed validation intervention evaluator exited with $exitCode." }
    $resultPath = Join-Path $Output 'result.json'
    if (-not (Test-Path -LiteralPath $resultPath)) { throw 'Intervention evaluator did not publish result.json.' }
    $result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
    if ($result.status -ne 'complete_fixed_validation_interventions') {
        throw 'Intervention result status is not complete.'
    }
    Write-WatchState 'complete' @{output=$Output; best_step=$jointResult.best_step; result=$resultPath}
    Add-Content -LiteralPath $LogPath -Value "[$(Get-Date -Format o)] intervention evaluation complete" -Encoding utf8
} catch {
    $message = $_ | Out-String
    Add-Content -LiteralPath $LogPath -Value "[$(Get-Date -Format o)] ERROR $message" -Encoding utf8
    Write-WatchState 'failed' @{error=$message; output=$Output}
    exit 1
}

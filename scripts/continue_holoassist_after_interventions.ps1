$ErrorActionPreference = 'Stop'

$Repo = 'C:\Users\junny\.codex\worktrees\r2-holoassist-textfree\tiny-omni-decision'
$Base = 'C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-recovery-02'
$FitOut = Join-Path $Base 'readout-v1-attempt-01'
$InterventionOut = Join-Path $Base 'readout-v1-intervention-attempt-01'
$InterventionWatchDir = Join-Path $Base 'readout-v1-intervention-watch-attempt-01'
$InterventionWatchState = Join-Path $InterventionWatchDir 'watch-state.json'
$Output = Join-Path $Base 'readout-v1-intervention-bootstrap-attempt-01'
$WatchDir = Join-Path $Base 'readout-v1-intervention-bootstrap-watch-attempt-01'
$WatchState = Join-Path $WatchDir 'watch-state.json'
$LogPath = Join-Path $WatchDir 'watch.log'
$MlPython = 'C:\CodexArtifacts\venvs\embeddinggemma2-v518\Scripts\python.exe'
$Bootstrap = Join-Path $Repo 'scripts\analyze_holoassist_paired_bootstrap.py'

if (Test-Path -LiteralPath $WatchDir) { throw "Refusing existing watcher path: $WatchDir" }
if (Test-Path -LiteralPath $Output) { throw "Refusing existing bootstrap path: $Output" }
New-Item -ItemType Directory -Path $WatchDir | Out-Null
$env:PYTHONPATH = "$Repo;$Repo\src"
$env:PYTHONUTF8 = '1'

function Write-WatchState([string]$Status, [object]$Details) {
    $payload = [pscustomobject]@{
        experiment_id = 'holoassist-av-paired-session-bootstrap-v1'
        status = $Status
        updated_at_utc = (Get-Date).ToUniversalTime().ToString('o')
        details = $Details
    }
    $temporary = "$WatchState.tmp"
    $payload | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $WatchState -Force
}

try {
    Write-WatchState 'waiting_for_intervention_results' @{source=$InterventionWatchState; output=$Output}
    Add-Content -LiteralPath $LogPath -Value "[$(Get-Date -Format o)] waiting for fixed validation interventions" -Encoding utf8
    while ($true) {
        if (Test-Path -LiteralPath $InterventionWatchState) {
            $interventionState = Get-Content -LiteralPath $InterventionWatchState -Raw | ConvertFrom-Json
            if ($interventionState.status -eq 'failed') {
                throw "Intervention evaluator failed: $($interventionState.details.error)"
            }
            if ($interventionState.status -eq 'complete') { break }
        }
        Start-Sleep -Seconds 60
    }

    if (Test-Path -LiteralPath $Output) { throw "Refusing existing bootstrap path: $Output" }
    $runMetadata = Get-Content -LiteralPath (Join-Path $FitOut 'run-metadata.json') -Raw | ConvertFrom-Json
    $interventionMetadata = Get-Content -LiteralPath (Join-Path $InterventionOut 'run-metadata.json') -Raw | ConvertFrom-Json
    if ($runMetadata.status -ne 'complete' -or $interventionMetadata.status -ne 'complete') {
        throw 'Both base readout and intervention evaluation must be complete.'
    }
    $bestStepByArm = @{}
    foreach ($armResult in $runMetadata.results) { $bestStepByArm[[string]$armResult.arm] = [int]$armResult.best_step }
    foreach ($arm in @('video_only','audio_only','joint')) {
        if (-not $bestStepByArm.ContainsKey($arm)) { throw "Missing selected checkpoint for $arm." }
    }
    $jointPrediction = Join-Path (Join-Path $FitOut 'joint') ('predictions-step-{0:D6}.json' -f $bestStepByArm['joint'])
    $comparisons = [System.Collections.Generic.List[object]]::new()
    foreach ($arm in @('video_only','audio_only')) {
        $baseline = Join-Path (Join-Path $FitOut $arm) ('predictions-step-{0:D6}.json' -f $bestStepByArm[$arm])
        $comparisons.Add(@{name="joint_vs_$arm"; baseline=$baseline; candidate=$jointPrediction})
    }
    foreach ($intervention in @('audio_missing','video_missing','audio_reverse_time','video_reverse_time','audio_shuffle_session')) {
        $candidate = Join-Path $InterventionOut "predictions-$intervention.json"
        $comparisons.Add(@{name="joint_vs_$intervention"; baseline=$jointPrediction; candidate=$candidate})
    }
    New-Item -ItemType Directory -Path $Output | Out-Null
    Write-WatchState 'running_session_cluster_bootstraps' @{comparison_count=$comparisons.Count; output=$Output; replicates=10000; seed=1701}
    $summary = [System.Collections.Generic.List[object]]::new()
    foreach ($comparison in $comparisons) {
        if (-not (Test-Path -LiteralPath $comparison.baseline) -or -not (Test-Path -LiteralPath $comparison.candidate)) {
            throw "Missing paired predictions for $($comparison.name)."
        }
        $result = Join-Path $Output "$($comparison.name).json"
        if (Test-Path -LiteralPath $result) { throw "Refusing existing bootstrap result: $result" }
        Add-Content -LiteralPath $LogPath -Value "[$(Get-Date -Format o)] START $($comparison.name)" -Encoding utf8
        & $MlPython $Bootstrap --baseline $comparison.baseline --candidate $comparison.candidate `
            --output $result --replicates 10000 --seed 1701 2>&1 | Tee-Object -FilePath $LogPath -Append
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) { throw "Bootstrap $($comparison.name) exited with $exitCode." }
        $summary.Add(@{
            name=$comparison.name
            baseline=$comparison.baseline
            candidate=$comparison.candidate
            output=$result
            output_sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $result).Hash.ToLower()
        })
    }
    $summaryPath = Join-Path $Output 'comparison-index.json'
    $summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $summaryPath -Encoding utf8
    Write-WatchState 'complete' @{output=$Output; comparison_count=$summary.Count; summary=$summaryPath}
    Add-Content -LiteralPath $LogPath -Value "[$(Get-Date -Format o)] all paired session bootstraps complete" -Encoding utf8
} catch {
    $message = $_ | Out-String
    Add-Content -LiteralPath $LogPath -Value "[$(Get-Date -Format o)] ERROR $message" -Encoding utf8
    Write-WatchState 'failed' @{error=$message; output=$Output}
    exit 1
}

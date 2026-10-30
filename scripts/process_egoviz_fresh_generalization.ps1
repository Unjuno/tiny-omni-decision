param(
    [Parameter(Mandatory = $true)][string]$PlanPath,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [string]$Python = 'C:\Users\junny\AppData\Local\Programs\Python\Python311\python.exe',
    [string]$VideoPython = 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\venv\Scripts\python.exe',
    [string]$VideoSource = 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\source',
    [string]$VideoCheckpoint = 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-f16r288.npz',
    [string]$SensorRoot = 'C:\CodexArtifacts\pretrained-reuse-r1\moment-small-sensor-r1-attempt-01',
    [string]$DependencyRoot = 'C:\CodexArtifacts\egoviz-mcap-preflight-20261029-attempt-01\python-deps'
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$PlanPath = (Resolve-Path -LiteralPath $PlanPath).Path
$outputParent = Split-Path -Parent $OutputRoot
if (-not (Test-Path -LiteralPath $outputParent -PathType Container)) { throw "Output parent missing: $outputParent" }
if (Test-Path -LiteralPath $OutputRoot) { throw "Refusing to overwrite output root: $OutputRoot" }
foreach ($required in @($Python, $VideoPython, $VideoSource, $VideoCheckpoint, "$SensorRoot\model", "$SensorRoot\moment-source", $DependencyRoot)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required runtime/input is missing: $required" }
}

$plan = Get-Content -LiteralPath $PlanPath -Raw | ConvertFrom-Json
if ($plan.format -ne 'egoviz-unseen-session-temporal-correspondence-plan-v1' -or
    $plan.status -ne 'METADATA_ONLY_FRESH_DEVELOPMENT_PLAN_MEDIA_NOT_DOWNLOADED' -or
    $plan.splits.train.Count -ne 0 -or $plan.splits.sealed_audit.Count -ne 0 -or
    $plan.source.prior_sealed_media_accessed -ne $false -or
    $plan.source.publisher_action_labels_read -ne $false) {
    throw 'Fresh-development plan or sealed-data isolation gate failed'
}
$planHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $PlanPath).Hash.ToLowerInvariant()
$expected = @($plan.splits.development)
if ($expected.Count -ne [int]$plan.totals.eligible_unseen_groups -or $expected.Count -eq 0) {
    throw 'Frozen development count mismatch'
}
if (@($expected | ForEach-Object group_id | Sort-Object -Unique).Count -ne $expected.Count) {
    throw 'Development groups are not unique'
}

$env:PYTHONPATH = $DependencyRoot
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:JAX_PLATFORMS = 'cpu'
$env:HF_HUB_OFFLINE = '1'
$env:TF_CPP_MIN_LOG_LEVEL = '2'
Set-Location -LiteralPath $repo
New-Item -ItemType Directory -Path $OutputRoot -ErrorAction Stop | Out-Null
$root = (Resolve-Path -LiteralPath $OutputRoot).Path
$rawRoot = Join-Path $root 'raw-sequential'
$sessionsRoot = Join-Path $root 'sessions'
New-Item -ItemType Directory -Path $rawRoot, $sessionsRoot -ErrorAction Stop | Out-Null

$state = [ordered]@{
    status = 'RUNNING'; pid = $PID; source_commit = (git rev-parse HEAD).Trim()
    plan_sha256 = $planHash; started_at = (Get-Date).ToString('o')
    expected_groups = $expected.Count; completed_groups = @()
    sealed_audit_accessed = $false; training = $false; device = 'cpu'
    current_group = $null; current_stage = 'preflight'
}
$statePath = Join-Path $root 'status.json'
$commandsPath = Join-Path $root 'commands.jsonl'
function Save-State {
    $state | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding utf8
}
function Invoke-Stage([string]$Stage, [string]$Executable, [string[]]$Arguments, [string]$LogPath) {
    $state.current_stage = $Stage
    Save-State
    [ordered]@{
        group_id = $state.current_group; stage = $Stage; executable = $Executable
        arguments = $Arguments; started_at = (Get-Date).ToString('o')
    } | ConvertTo-Json -Compress | Add-Content -LiteralPath $commandsPath -Encoding utf8
    & $Executable @Arguments 2>&1 | Out-File -LiteralPath $LogPath -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw "$Stage failed; inspect $LogPath" }
}

Save-State
try {
    for ($index = 0; $index -lt $expected.Count; $index++) {
        if ((Get-FileHash -Algorithm SHA256 -LiteralPath $PlanPath).Hash.ToLowerInvariant() -ne $planHash) {
            throw 'Frozen plan hash changed during processing'
        }
        $row = $expected[$index]
        $chunk = [string]$row.chunk_id
        if ($chunk -notmatch '^[a-f0-9]+$' -or $row.split -ne 'development') { throw 'Invalid frozen row identity' }
        $state.current_group = [string]$row.group_id
        $started = Get-Date
        $raw = Join-Path $rawRoot $chunk
        $package = Join-Path $sessionsRoot $chunk
        if ((Test-Path -LiteralPath $raw) -or (Test-Path -LiteralPath $package)) { throw "Refusing to overwrite group $chunk" }
        $free = (Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'").FreeSpace
        if ($free -lt (3000000000 + [long]$row.expected_bytes)) {
            throw "Insufficient reserve before ${chunk}: free=$free bytes, required=$([long]$row.expected_bytes + 3000000000)"
        }

        Invoke-Stage 'download' $Python @(
            '-m','scripts.download_egoviz_unseen_group_media','--plan',$PlanPath,
            '--output',$raw,'--splits','development','--offset-per-split',[string]$index,
            '--limit-per-split','1','--max-total-bytes','1000000000','--reserve-bytes','3000000000'
        ) (Join-Path $root "download-$chunk.log")

        New-Item -ItemType Directory -Path $package -ErrorAction Stop | Out-Null
        Copy-Item -LiteralPath (Join-Path $raw 'acquisition-manifest.json') -Destination (Join-Path $package 'acquisition-manifest.json') -ErrorAction Stop
        Invoke-Stage 'window_corpus' $Python @(
            '-m','scripts.build_egoviz_vs_sync_window_corpus','--manifest',
            (Join-Path $package 'acquisition-manifest.json'),'--output',
            (Join-Path $package 'window-corpus'),'--seed','17'
        ) (Join-Path $package 'window-build.log')
        Invoke-Stage 'video_features_cpu' $VideoPython @(
            '-m','scripts.extract_egoviz_vs_sync_video_features','--corpus',
            (Join-Path $package 'window-corpus'),'--acquisition-manifest',
            (Join-Path $package 'acquisition-manifest.json'),'--checkpoint',$VideoCheckpoint,
            '--source-root',$VideoSource,'--output',(Join-Path $package 'video-features')
        ) (Join-Path $package 'video-extraction.log')
        Invoke-Stage 'sensor_features_cpu' $Python @(
            '-m','scripts.extract_egoviz_vs_sync_sensor_features','--corpus',
            (Join-Path $package 'window-corpus'),'--checkpoint-dir',(Join-Path $SensorRoot 'model'),
            '--moment-source',(Join-Path $SensorRoot 'moment-source'),'--output',
            (Join-Path $package 'sensor-features'),'--batch-size','1','--device','cpu'
        ) (Join-Path $package 'sensor-extraction.log')
        Invoke-Stage 'package_validation' $Python @(
            '-m','scripts.aggregate_egoviz_unseen_group_features','--plan',$PlanPath,
            '--runs-root',$sessionsRoot,'--validate-chunk-id',$chunk
        ) (Join-Path $package 'validation.log')

        $verified = Get-Content -LiteralPath (Join-Path $package 'validation.log') -Raw | ConvertFrom-Json
        if ($verified.status -ne 'FROZEN_SESSION_PACKAGE_VALIDATED_NO_TRAINING_OR_QUALITY_EVALUATION' -or
            $verified.chunk_id -ne $chunk -or $verified.sealed_audit_accessed -ne $false) {
            throw "Session package failed identity/holdout validation: $chunk"
        }
        $media = Join-Path $raw "$chunk.mcap"
        $resolvedMedia = (Resolve-Path -LiteralPath $media).Path
        if (-not $resolvedMedia.StartsWith($rawRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Temporary media escaped new output root'
        }
        $item = Get-Item -LiteralPath $resolvedMedia
        if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
            $item.Length -ne [long]$row.expected_bytes -or
            (Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedMedia).Hash.ToLowerInvariant() -ne $verified.raw_media_sha256) {
            throw "Temporary media integrity check failed: $chunk"
        }
        [ordered]@{
            path = $resolvedMedia; bytes = $item.Length; sha256 = $verified.raw_media_sha256
            source_receipt = (Join-Path $raw 'acquisition-manifest.json')
            retained_package = $package; verified_at = (Get-Date).ToString('o')
            status = 'VERIFIED_FOR_RELEASE'
        } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $package 'temporary-media-release.json') -Encoding utf8
        Remove-Item -LiteralPath $resolvedMedia -ErrorAction Stop
        [ordered]@{
            path = $resolvedMedia; bytes = $item.Length; sha256 = $verified.raw_media_sha256
            released_at = (Get-Date).ToString('o'); status = 'RELEASED_NEW_TEMPORARY_MEDIA'
        } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $package 'temporary-media-release.json') -Encoding utf8

        $state.completed_groups += $chunk
        $state.last_group_seconds = ((Get-Date) - $started).TotalSeconds
        $state.free_bytes = (Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'").FreeSpace
        $state.current_group = $null
        Save-State
        [ordered]@{
            completed = $state.completed_groups.Count; expected = $expected.Count
            chunk_id = $chunk; seconds = $state.last_group_seconds
            free_bytes = $state.free_bytes; time = (Get-Date).ToString('o')
        } | ConvertTo-Json -Compress | Write-Output
    }

    Invoke-Stage 'full_aggregate' $Python @(
        '-m','scripts.aggregate_egoviz_unseen_group_features','--plan',$PlanPath,
        '--runs-root',$sessionsRoot,'--output',(Join-Path $root 'full-aggregate')
    ) (Join-Path $root 'full-aggregate.log')
    $aggregate = Get-Content -LiteralPath (Join-Path $root 'full-aggregate\aggregate-report.json') -Raw | ConvertFrom-Json
    if ($aggregate.status -ne 'FROZEN_FEATURES_AGGREGATED_NO_TRAINING_OR_QUALITY_EVALUATION' -or
        $aggregate.development_sessions -ne $expected.Count -or $aggregate.sealed_audit_accessed -ne $false -or
        $aggregate.training -ne $false) { throw 'Final aggregate report failed integrity checks' }
    $state.status = 'ALL_FRESH_DEVELOPMENT_GROUPS_AGGREGATED_NO_TRAINING_OR_EVALUATION'
    $state.ended_at = (Get-Date).ToString('o')
    $state.current_stage = 'complete'
    Save-State
} catch {
    $state.status = 'FAILED'
    $state.error = $_.Exception.Message
    $state.ended_at = (Get-Date).ToString('o')
    Save-State
    throw
}

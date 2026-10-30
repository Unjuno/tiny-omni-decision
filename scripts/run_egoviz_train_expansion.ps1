$ErrorActionPreference = 'Stop'
$taskRoot = 'C:\CodexArtifacts\egoviz-train-expansion-20261030-attempt-01'
$taskRepo = 'C:\CodexWorktrees\r2-egoviz-mcap-preflight-20261029'
$taskPlanPath = Join-Path $taskRepo 'manifests\pretrained_reuse\egoviz-train-expansion-plan-20261030.json'
$taskVideoPython = 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\venv\Scripts\python.exe'
$taskPython = 'C:\Users\junny\AppData\Local\Programs\Python\Python311\python.exe'
$taskVideoSource = 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\source'
$taskVideoCheckpoint = 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-f16r288.npz'
$taskSensorRoot = 'C:\CodexArtifacts\pretrained-reuse-r1\moment-small-sensor-r1-attempt-01'
$env:PYTHONPATH = 'C:\CodexArtifacts\egoviz-mcap-preflight-20261029-attempt-01\python-deps'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:JAX_PLATFORMS = 'cpu'
$env:HF_HUB_OFFLINE = '1'
$env:TF_CPP_MIN_LOG_LEVEL = '2'
Set-Location -LiteralPath $taskRepo
$taskCommit = (git rev-parse HEAD)
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $taskPlanPath).Hash -ine '18958dcece44ae8a387895cbb5b64ffeb218f4988a8d59401522d797f8703b3') { throw 'Frozen plan hash differs' }
$taskPlan = Get-Content -LiteralPath $taskPlanPath -Raw | ConvertFrom-Json
if ($taskPlan.splits.train.Count -ne 24 -or $taskPlan.splits.development.Count -ne 0 -or $taskPlan.splits.sealed_audit.Count -ne 0) { throw 'Unexpected plan splits' }
if (Test-Path -LiteralPath $taskRoot) { throw 'Existing run: inspect its status and process instead of restarting' }
New-Item -ItemType Directory -Path $taskRoot -ErrorAction Stop | Out-Null
$taskResolvedRoot = (Resolve-Path -LiteralPath $taskRoot).Path
if ($taskResolvedRoot -ine $taskRoot -or ((Get-Item -LiteralPath $taskRoot).Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Unexpected execution-root identity' }
$taskRawRoot = Join-Path $taskRoot 'raw-sequential'
New-Item -ItemType Directory -Path $taskRawRoot -ErrorAction Stop | Out-Null
$taskRawRoot = (Resolve-Path -LiteralPath $taskRawRoot).Path
if ((Get-Item -LiteralPath $taskRawRoot).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Unexpected raw-root reparse point' }
$taskState = [ordered]@{status='RUNNING';pid=$PID;source_commit=$taskCommit;controller_sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $PSCommandPath).Hash;started_at=(Get-Date).ToString('o');completed_sessions=@();expected_sessions=24;sealed_audit_accessed=$false;training=$false}
function Save-TaskState { $taskState | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath "$taskRoot\status.json" -Encoding utf8 }
function Invoke-TaskStage([string]$Stage, [string]$Executable, [string[]]$Arguments, [string]$Log) {
    $taskState['current_stage']=$Stage
    Save-TaskState
    [ordered]@{session=$taskState.current_session;stage=$Stage;executable=$Executable;arguments=$Arguments;started_at=(Get-Date).ToString('o')} | ConvertTo-Json -Compress | Add-Content -LiteralPath "$taskRoot\commands.jsonl" -Encoding utf8
    & $Executable @Arguments 2>&1 | Out-File -LiteralPath $Log -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw "$Stage failed; inspect $Log" }
}
Save-TaskState
try {
    for ($taskOffset=0; $taskOffset -lt $taskPlan.splits.train.Count; $taskOffset++) {
        $taskRow=$taskPlan.splits.train[$taskOffset]
        $taskChunk=[string]$taskRow.chunk_id
        if ($taskChunk -notmatch '^[a-f0-9]+$') { throw 'Invalid session identity' }
        git diff --quiet $taskCommit -- scripts/download_egoviz_unseen_group_media.py scripts/build_egoviz_vs_sync_window_corpus.py scripts/extract_egoviz_vs_sync_video_features.py scripts/extract_egoviz_vs_sync_sensor_features.py scripts/aggregate_egoviz_unseen_group_features.py
        if ($LASTEXITCODE -ne 0) { throw 'Active extraction source changed' }
        $taskStarted=Get-Date
        $taskState['current_session']=$taskChunk
        $taskRaw=Join-Path $taskRawRoot $taskChunk
        $taskPackage=Join-Path "$taskRoot\sessions" $taskChunk
        if (Test-Path -LiteralPath $taskPackage) { throw 'Refusing to overwrite an existing session' }
        $taskFree=(Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'").FreeSpace
        if ($taskFree -lt (3000000000 + [long]$taskRow.expected_bytes)) { throw 'Insufficient disk reserve for next session' }
        Invoke-TaskStage 'download' $taskPython @('-m','scripts.download_egoviz_unseen_group_media','--plan',$taskPlanPath,'--output',$taskRaw,'--splits','train','--offset-per-split',[string]$taskOffset,'--limit-per-split','1','--reserve-bytes','3000000000') "$taskRoot\download-$taskChunk.log"
        New-Item -ItemType Directory -Path $taskPackage -ErrorAction Stop | Out-Null
        Copy-Item -LiteralPath "$taskRaw\acquisition-manifest.json" -Destination "$taskPackage\acquisition-manifest.json" -ErrorAction Stop
        Invoke-TaskStage 'window_corpus' $taskPython @('-m','scripts.build_egoviz_vs_sync_window_corpus','--manifest',"$taskPackage\acquisition-manifest.json",'--output',"$taskPackage\window-corpus",'--seed','17') "$taskPackage\window-build.log"
        Invoke-TaskStage 'video_features' $taskVideoPython @('-m','scripts.extract_egoviz_vs_sync_video_features','--corpus',"$taskPackage\window-corpus",'--acquisition-manifest',"$taskPackage\acquisition-manifest.json",'--checkpoint',$taskVideoCheckpoint,'--source-root',$taskVideoSource,'--output',"$taskPackage\video-features") "$taskPackage\video-extraction.log"
        Invoke-TaskStage 'sensor_features' $taskPython @('-m','scripts.extract_egoviz_vs_sync_sensor_features','--corpus',"$taskPackage\window-corpus",'--checkpoint-dir',"$taskSensorRoot\model",'--moment-source',"$taskSensorRoot\moment-source",'--output',"$taskPackage\sensor-features",'--batch-size','1','--device','cpu') "$taskPackage\sensor-extraction.log"
        Invoke-TaskStage 'package_validation' $taskPython @('-m','scripts.aggregate_egoviz_unseen_group_features','--plan',$taskPlanPath,'--runs-root',"$taskRoot\sessions",'--validate-chunk-id',$taskChunk) "$taskPackage\validation.log"
        $taskVerified=Get-Content -LiteralPath "$taskPackage\validation.log" -Raw | ConvertFrom-Json
        if ($taskVerified.status -ne 'FROZEN_SESSION_PACKAGE_VALIDATED_NO_TRAINING_OR_QUALITY_EVALUATION' -or $taskVerified.chunk_id -ne $taskChunk -or $taskVerified.sealed_audit_accessed -ne $false) { throw 'Unexpected package validation result' }
        $taskMedia=(Resolve-Path -LiteralPath "$taskRaw\$taskChunk.mcap").Path
        if (-not $taskMedia.StartsWith($taskRawRoot+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Temporary media escaped this execution root' }
        $taskInfo=Get-Item -LiteralPath $taskMedia
        if ($taskInfo.PSIsContainer -or ($taskInfo.Attributes -band [IO.FileAttributes]::ReparsePoint) -or $taskInfo.Length -ne [long]$taskRow.expected_bytes) { throw 'Temporary media identity changed' }
        if ((Get-FileHash -Algorithm SHA256 -LiteralPath $taskMedia).Hash -ine $taskVerified.raw_media_sha256) { throw 'Temporary media checksum changed' }
        $taskRelease=[ordered]@{path=$taskMedia;bytes=$taskInfo.Length;sha256=$taskVerified.raw_media_sha256;retained_package=$taskPackage;source_receipt="$taskRaw\acquisition-manifest.json";verified_at=(Get-Date).ToString('o');status='VERIFIED_FOR_RELEASE'}
        $taskRelease | ConvertTo-Json | Set-Content -LiteralPath "$taskPackage\temporary-media-release.json" -Encoding utf8
        Remove-Item -LiteralPath $taskMedia -ErrorAction Stop
        $taskRelease['status']='RELEASED_NEW_TEMPORARY_MEDIA'
        $taskRelease['released_at']=(Get-Date).ToString('o')
        $taskRelease | ConvertTo-Json | Set-Content -LiteralPath "$taskPackage\temporary-media-release.json" -Encoding utf8
        $taskState['completed_sessions'] += $taskChunk
        $taskState['last_session_seconds']=((Get-Date)-$taskStarted).TotalSeconds
        $taskState['free_bytes']=(Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'").FreeSpace
        Save-TaskState
        [ordered]@{completed=$taskState.completed_sessions.Count;expected=24;chunk_id=$taskChunk;seconds=$taskState.last_session_seconds;free_bytes=$taskState.free_bytes;time=(Get-Date).ToString('o')} | ConvertTo-Json -Compress | Write-Output
    }
    $taskState['current_session']=$null
    Invoke-TaskStage 'full_aggregate' $taskPython @('-m','scripts.aggregate_egoviz_unseen_group_features','--plan',$taskPlanPath,'--runs-root',"$taskRoot\sessions",'--output',"$taskRoot\full-aggregate") "$taskRoot\full-aggregate.log"
    $taskState['status']='ALL_24_TRAIN_SESSIONS_AGGREGATED_NO_TRAINING'
    $taskState['ended_at']=(Get-Date).ToString('o')
    Save-TaskState
} catch {
    $taskState['status']='FAILED'
    $taskState['error']=$_.Exception.Message
    $taskState['ended_at']=(Get-Date).ToString('o')
    Save-TaskState
    throw
}

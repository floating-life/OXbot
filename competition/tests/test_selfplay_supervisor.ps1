[CmdletBinding()]
param([string]$ShellExecutable = (Get-Process -Id $PID).Path)
$ErrorActionPreference = 'Stop'
$entry = (Resolve-Path (Join-Path $PSScriptRoot '../scripts/selfplay_eval_5080.ps1')).Path
$fixture = (Resolve-Path (Join-Path $PSScriptRoot 'fixtures/fake_supervised_pipeline.ps1')).Path
$testRoot = Join-Path ([IO.Path]::GetTempPath()) ('oxbot supervisor ' + [Guid]::NewGuid().ToString('N') + ' ' + [char]0x6D4B + [char]0x8BD5)
New-Item -ItemType Directory -Path $testRoot | Out-Null
$passed = 0

function Assert-True([bool]$Condition, [string]$Description) {
    if (-not $Condition) { throw "FAILED: $Description. Artifacts: $testRoot" }
    $script:passed++
    Write-Output "PASS: $Description"
}
function Invoke-Entry([string]$Out, [string]$Scenario, [string]$Mode = 'all') {
    $tag = [Guid]::NewGuid().ToString('N').Substring(0, 8)
    $stdout = Join-Path $testRoot "$tag.launcher.log"
    $stderr = Join-Path $testRoot "$tag.launcher.stderr.log"
    $arguments = @('-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File',
        ('"' + $entry + '"'), '-Mode', $Mode, '-Out', ('"' + $Out + '"'), '-HeartbeatSeconds', '1',
        '-NoDesktopNotification', '-TestChildScript', ('"' + $fixture + '"'), '-TestScenario', $Scenario)
    $process = Start-Process -FilePath $ShellExecutable -ArgumentList $arguments -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $null = $process.Handle
    if (-not $process.WaitForExit(25000)) { throw "Launcher did not exit; PID $($process.Id). Artifacts: $testRoot" }
    $process.WaitForExit()
    $process.Refresh()
    return @{ code = $process.ExitCode; stdout = $stdout; stderr = $stderr }
}
function Read-State([string]$Out) {
    return Get-Content -LiteralPath (Join-Path $Out 'supervisor_status.json') -Raw -Encoding UTF8 | ConvertFrom-Json
}
function Wait-Terminal([string]$Out) {
    $deadline = [DateTimeOffset]::UtcNow.AddSeconds(20)
    do {
        $state = Read-State $Out
        if ($state.status -ne 'running') { return $state }
        Start-Sleep -Milliseconds 250
    } while ([DateTimeOffset]::UtcNow -lt $deadline)
    throw "No terminal state: $Out"
}

$successfulOut = Join-Path $testRoot 'successful all'
$launch = Invoke-Entry $successfulOut 'success'
Assert-True ($launch.code -eq 0) 'launcher acknowledged the detached supervisor'
$running = Read-State $successfulOut
Assert-True ($running.status -eq 'running') 'launcher exits while the pipeline is still running'
$launchConfig = Get-Content -LiteralPath $running.launch_path -Raw -Encoding UTF8 | ConvertFrom-Json
Assert-True ($launchConfig.safe_cuda -eq 1 -and $launchConfig.evaluation_seed -eq 20261101) 'manifest fixes the safe CUDA mode and evaluation seed for the continued experiment'
Assert-True ($launchConfig.micro_batch -eq 128) 'manifest fixes the memory-safe learner micro-batch'
$workerProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($running.supervisor_pid)"
$workerParent = Get-CimInstance Win32_Process -Filter "ProcessId=$($workerProcess.ParentProcessId)"
Assert-True ($workerParent.Name -eq 'WmiPrvSE.exe') 'supervisor is parented by WMI, outside the caller process tree'
$liveStatus = Invoke-Entry $successfulOut 'success' 'status'
$liveReport = Get-Content -LiteralPath $liveStatus.stdout -Raw -Encoding UTF8 | ConvertFrom-Json
Assert-True ($liveReport.status -eq 'running') 'status validates live process identity and heartbeat'
$terminal = Wait-Terminal $successfulOut
Assert-True ($terminal.status -eq 'completed' -and $terminal.mode -eq 'all' -and $terminal.pipeline_status -eq 'completed') 'all completes only after matching pipeline completion'
$notice = Get-Content -LiteralPath (Join-Path $successfulOut 'notification-needed.json') -Raw -Encoding UTF8 | ConvertFrom-Json
Assert-True ($notice.run_id -eq $terminal.run_id -and $notice.notification_required) 'completion is recorded for a later notifier'
Assert-True ((Get-Content -LiteralPath $terminal.stdout_path -Raw -Encoding UTF8).Contains('fake evaluation started')) 'detached stdout survives entry process exit and contains evaluation'

$failedOut = Join-Path $testRoot 'failed child'
$null = Invoke-Entry $failedOut 'failure'
$failure = Wait-Terminal $failedOut
Assert-True ($failure.status -eq 'failed' -and $failure.exit_code -eq 7) 'nonzero child exit creates a failed terminal state'
Assert-True ((Get-Content -LiteralPath $failure.stderr_path -Raw -Encoding UTF8).Contains('synthetic child failure')) 'stderr retains the original child failure'
Assert-True ((Test-Path -LiteralPath (Join-Path $failedOut 'notification-needed.json'))) 'failure creates the persistent notification marker'

$incompleteOut = Join-Path $testRoot 'zero without completed report'
$null = Invoke-Entry $incompleteOut 'incomplete'
$incomplete = Wait-Terminal $incompleteOut
Assert-True ($incomplete.status -eq 'failed' -and $incomplete.failure_reason -eq 'child_exited_without_matching_completed_pipeline') 'exit zero alone cannot be reported as full completion'

$lockedOut = Join-Path $testRoot 'concurrent attempts'
$null = Invoke-Entry $lockedOut 'success'
$first = Read-State $lockedOut
$secondLaunch = Invoke-Entry $lockedOut 'success'
$afterSecond = Read-State $lockedOut
Assert-True ($secondLaunch.code -ne 0 -and $afterSecond.run_id -eq $first.run_id) 'output lock rejects a competing launch without replacing active status'
$null = Wait-Terminal $lockedOut

# Read-only status must flag a lost supervisor, including a reused PID, even
# when its last heartbeat file still says running.
$orphanOut = Join-Path $testRoot 'orphan status'
New-Item -ItemType Directory -Path $orphanOut | Out-Null
$orphan = $running
$orphan.supervisor_pid = $PID
$orphan.supervisor_started_at = [DateTimeOffset]::UtcNow.AddDays(-1).ToString('o')
$orphan.heartbeat_at = [DateTimeOffset]::UtcNow.ToString('o')
[IO.File]::WriteAllText((Join-Path $orphanOut 'supervisor_status.json'), ($orphan | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
$orphanResult = Invoke-Entry $orphanOut 'success' 'status'
$orphanStatus = Get-Content -LiteralPath $orphanResult.stdout -Raw -Encoding UTF8 | ConvertFrom-Json
Assert-True ($orphanStatus.status -eq 'attention_required' -and $orphanStatus.failure_reason -eq 'supervisor_missing') 'status detects a missing supervisor and rejects recycled process identity'

$progressOut = Join-Path $testRoot 'stale progress status'
New-Item -ItemType Directory -Path $progressOut | Out-Null
$progressState = $running | ConvertTo-Json -Depth 12 | ConvertFrom-Json
$progressState.supervisor_pid = $PID
$progressState.supervisor_started_at = (Get-Process -Id $PID).StartTime.ToUniversalTime().ToString('o')
$progressState.heartbeat_at = [DateTimeOffset]::UtcNow.ToString('o')
$progressState.stage = 'train'
$progressState.progress_path = Join-Path $progressOut 'training_progress.json'
$progressStatusPath = Join-Path $progressOut 'supervisor_status.json'
$progressFixture = @{
    run_id = $progressState.run_id; phase = 'collect'
    progress_at = [DateTimeOffset]::UtcNow.AddMinutes(-20).ToUnixTimeMilliseconds() / 1000.0
    updated_at = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
}
[IO.File]::WriteAllText($progressState.progress_path, ($progressFixture | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
[IO.File]::WriteAllText($progressStatusPath, ($progressState | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
$staleResult = Invoke-Entry $progressOut 'success' 'status'
$staleState = Get-Content -LiteralPath $staleResult.stdout -Raw -Encoding UTF8 | ConvertFrom-Json
Assert-True ($staleState.status -eq 'attention_required' -and $staleState.failure_reason -eq 'training_progress_stale') 'stalled trainer is detected even while its supervisor and heartbeat are alive'
Assert-True ((Read-State $progressOut).status -eq 'running') 'status diagnostics do not mutate saved run state'

$progressFixture.progress_at = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
[IO.File]::WriteAllText($progressState.progress_path, ($progressFixture | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
$freshResult = Invoke-Entry $progressOut 'success' 'status'
$freshState = Get-Content -LiteralPath $freshResult.stdout -Raw -Encoding UTF8 | ConvertFrom-Json
Assert-True ($freshState.status -eq 'running') 'fresh Unix epoch training progress is accepted'
$progressFixture.progress_at = [DateTimeOffset]::UtcNow.AddMinutes(-20).ToUnixTimeMilliseconds() / 1000.0
$progressFixture.phase = 'eval_rule'
[IO.File]::WriteAllText($progressState.progress_path, ($progressFixture | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
$internalEvalResult = Invoke-Entry $progressOut 'success' 'status'
$internalEvalState = Get-Content -LiteralPath $internalEvalResult.stdout -Raw -Encoding UTF8 | ConvertFrom-Json
Assert-True ($internalEvalState.status -eq 'running') 'internal evaluation uses fresh heartbeat even when samples and learner steps do not change'

$progressFixture.phase = 'saving_final'
$progressFixture.updated_at = [DateTimeOffset]::UtcNow.AddMinutes(-20).ToUnixTimeMilliseconds() / 1000.0
[IO.File]::WriteAllText($progressState.progress_path, ($progressFixture | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
$savingResult = Invoke-Entry $progressOut 'success' 'status'
$savingState = Get-Content -LiteralPath $savingResult.stdout -Raw -Encoding UTF8 | ConvertFrom-Json
Assert-True ($savingState.status -eq 'attention_required' -and $savingState.failure_reason -eq 'training_heartbeat_stale') 'all training phases detect a stopped trainer heartbeat'

$progressState.stage = 'eval'
[IO.File]::WriteAllText($progressStatusPath, ($progressState | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
$evalResult = Invoke-Entry $progressOut 'success' 'status'
$evalState = Get-Content -LiteralPath $evalResult.stdout -Raw -Encoding UTF8 | ConvertFrom-Json
Assert-True ($evalState.status -eq 'running') 'long formal evaluation is not marked stalled from training progress'

$progressState.stage = 'train'
$progressFixture.run_id = 'a-previous-run'
[IO.File]::WriteAllText($progressState.progress_path, ($progressFixture | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
[IO.File]::WriteAllText($progressStatusPath, ($progressState | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
$oldResult = Invoke-Entry $progressOut 'success' 'status'
$oldState = Get-Content -LiteralPath $oldResult.stdout -Raw -Encoding UTF8 | ConvertFrom-Json
Assert-True ($oldState.status -eq 'running') 'stale progress from another run is ignored'

$progressState.pipeline_updated_at = [DateTimeOffset]::UtcNow.AddMinutes(-20).ToString('o')
[IO.File]::WriteAllText($progressStatusPath, ($progressState | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
$missingResult = Invoke-Entry $progressOut 'success' 'status'
$missingState = Get-Content -LiteralPath $missingResult.stdout -Raw -Encoding UTF8 | ConvertFrom-Json
Assert-True ($missingState.status -eq 'attention_required' -and $missingState.failure_reason -eq 'training_progress_missing') 'a training stage without matching progress is reported after fifteen minutes'

$progressState.heartbeat_at = [DateTimeOffset]::UtcNow.AddMinutes(-10).ToString('o')
[IO.File]::WriteAllText($progressStatusPath, ($progressState | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
$staleSupervisorResult = Invoke-Entry $progressOut 'success' 'status'
$staleSupervisorState = Get-Content -LiteralPath $staleSupervisorResult.stdout -Raw -Encoding UTF8 | ConvertFrom-Json
Assert-True ($staleSupervisorState.status -eq 'attention_required' -and $staleSupervisorState.failure_reason -eq 'supervisor_heartbeat_stale') 'a live but unresponsive supervisor is detected'

Write-Output "$passed assertions passed. Artifacts: $testRoot"

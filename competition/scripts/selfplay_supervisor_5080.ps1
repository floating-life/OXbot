[CmdletBinding()]
param([Parameter(Mandatory)][string]$LaunchPath)

$ErrorActionPreference = 'Stop'
$config = Get-Content -LiteralPath $LaunchPath -Raw -Encoding UTF8 | ConvertFrom-Json
$lock = $null
$child = $null
$terminalRecorded = $false
$sleepGuard = $false
$state = $null

function Write-AtomicJson([string]$Path, $Value) {
    $temporary = "$Path.$PID.tmp"
    [IO.File]::WriteAllText($temporary, ($Value | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
    for ($attempt = 0; ; $attempt++) {
        try {
            if ([IO.File]::Exists($Path)) { [IO.File]::Replace($temporary, $Path, [NullString]::Value) }
            else { [IO.File]::Move($temporary, $Path) }
            break
        } catch [IO.IOException] {
            if ($attempt -ge 4) { throw }
            Start-Sleep -Milliseconds 100
        }
    }
}

function Write-SupervisorLog([string]$Message) {
    $line = '[{0}] {1}{2}' -f [DateTimeOffset]::UtcNow.ToString('o'), $Message, [Environment]::NewLine
    [IO.File]::AppendAllText($config.supervisor_log_path, $line, [Text.UTF8Encoding]::new($false))
}

function ConvertTo-WindowsArgument([string]$Value) {
    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') { return $Value }
    $result = [Text.StringBuilder]::new('"')
    $slashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq '\') { $slashes++; continue }
        if ($character -eq '"') { [void]$result.Append(('\' * ($slashes * 2 + 1))) }
        else { [void]$result.Append(('\' * $slashes)) }
        [void]$result.Append($character)
        $slashes = 0
    }
    [void]$result.Append(('\' * ($slashes * 2)))
    [void]$result.Append('"')
    return $result.ToString()
}

function Read-Pipeline {
    if (Test-Path -LiteralPath $config.pipeline_path) {
        try {
            $pipeline = Get-Content -LiteralPath $config.pipeline_path -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($pipeline.run_id -eq $config.run_id) { return $pipeline }
        } catch { Write-SupervisorLog "Cannot read pipeline status yet: $($_.Exception.Message)" }
    }
    return $null
}

function Write-State {
    $state.heartbeat_at = [DateTimeOffset]::UtcNow.ToString('o')
    $state.updated_at = $state.heartbeat_at
    $pipeline = Read-Pipeline
    if ($pipeline) {
        $state.stage = $pipeline.stage
        $state.pipeline_status = $pipeline.status
        $state.pipeline_updated_at = $pipeline.updated_at
        $state.eval_directory = $pipeline.eval_dir
    }
    foreach ($kind in @('stdout', 'stderr')) {
        $path = $config."${kind}_path"
        if (Test-Path -LiteralPath $path) {
            $info = Get-Item -LiteralPath $path
            $state["${kind}_updated_at"] = $info.LastWriteTimeUtc.ToString('o')
            $state["${kind}_bytes"] = $info.Length
        }
    }
    Write-AtomicJson $config.run_status_path $state
    Write-AtomicJson $config.status_path $state
}

function Write-Terminal([string]$Status, $ExitCode, [string]$Reason) {
    $state.status = $Status
    $state.exit_code = $ExitCode
    $state.finished_at = [DateTimeOffset]::UtcNow.ToString('o')
    $state.failure_reason = $Reason
    $state.notification_required = $true
    Write-State
    $notice = [ordered]@{
        run_id = $config.run_id; status = $Status; mode = $config.mode; stage = $state.stage
        created_at = $state.finished_at; notification_required = $true; exit_code = $ExitCode
        reason = $Reason; status_path = $config.status_path; launch_path = $LaunchPath
        stdout_path = $config.stdout_path; stderr_path = $config.stderr_path
        pipeline_path = $config.pipeline_path; eval_directory = $state.eval_directory
        desktop_notification_requested = $config.desktop_notification
    }
    Write-AtomicJson $config.notification_path $notice
    Write-AtomicJson (Join-Path $config.output_directory ("notification-{0}.json" -f $config.run_id)) $notice
    $script:terminalRecorded = $true
    Write-SupervisorLog "Terminal state: $Status; exit_code=$ExitCode; reason=$Reason"
}

function Show-TerminalNotification {
    if (-not $config.desktop_notification) { return }
    $tray = $null
    try {
        Add-Type -AssemblyName System.Windows.Forms
        Add-Type -AssemblyName System.Drawing
        $tray = [Windows.Forms.NotifyIcon]::new()
        $tray.Icon = if ($state.status -eq 'completed') { [Drawing.SystemIcons]::Information } else { [Drawing.SystemIcons]::Error }
        $tray.Visible = $true
        # JSON escapes keep the script compatible with Windows PowerShell
        # 5.1's legacy source encoding while the actual notification is Chinese.
        if ($state.status -eq 'completed') {
            $tray.BalloonTipTitle = if ($config.mode -eq 'train') {
                '"OXbot \u8bad\u7ec3\u9636\u6bb5\u7ed3\u675f"' | ConvertFrom-Json
            } elseif ($config.mode -eq 'eval') {
                '"OXbot \u9a8c\u6536\u5df2\u7ed3\u675f"' | ConvertFrom-Json
            } else { '"OXbot \u672c\u8f6e\u8bad\u7ec3\u4e0e\u9a8c\u6536\u5df2\u7ed3\u675f"' | ConvertFrom-Json }
            $tray.BalloonTipText = ('"\u67e5\u770b\u62a5\u544a\uff1a"' | ConvertFrom-Json) + $config.output_directory
        } else {
            $tray.BalloonTipTitle = '"OXbot \u8bad\u7ec3/\u9a8c\u6536\u4e2d\u65ad\uff0c\u9700\u8981\u5904\u7406"' | ConvertFrom-Json
            $tray.BalloonTipText = ('"\u672a\u81ea\u52a8\u91cd\u8bd5\uff1b\u65e5\u5fd7\uff1a"' | ConvertFrom-Json) + $config.output_directory
        }
        $tray.ShowBalloonTip(10000)
        for ($i = 0; $i -lt 10; $i++) {
            [Windows.Forms.Application]::DoEvents()
            Start-Sleep -Seconds 1
        }
        Write-SupervisorLog 'Desktop notification submitted (Windows notification preferences still apply).'
    } catch { Write-SupervisorLog "Desktop notification unavailable: $($_.Exception.Message)" }
    finally {
        if ($tray) { $tray.Visible = $false; $tray.Dispose() }
    }
}

try {
    # An OS-held lock is released even after a host crash. A competing launch
    # must never replace the active run's heartbeat or terminal notification.
    $lockPath = Join-Path $config.output_directory '.supervisor.lock'
    $lock = [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    $state = [ordered]@{
        schema_version = 1; run_id = $config.run_id; mode = $config.mode; status = 'running'; stage = 'starting'
        started_at = [DateTimeOffset]::UtcNow.ToString('o'); finished_at = $null; updated_at = $null
        heartbeat_at = $null; heartbeat_seconds = $config.heartbeat_seconds
        supervisor_pid = $PID; supervisor_started_at = (Get-Process -Id $PID).StartTime.ToUniversalTime().ToString('o')
        launch_backend = $config.launch_backend
        child_pid = $null; child_started_at = $null; exit_code = $null; failure_reason = $null
        notification_required = $false; pipeline_status = $null; pipeline_updated_at = $null; eval_directory = $null
        output_directory = $config.output_directory; launch_path = $LaunchPath; pipeline_path = $config.pipeline_path
        stdout_path = $config.stdout_path; stderr_path = $config.stderr_path; progress_path = $config.progress_path
        supervisor_log_path = $config.supervisor_log_path; supervisor_stderr_path = $config.supervisor_stderr_path
        stdout_updated_at = $null; stderr_updated_at = $null; stdout_bytes = 0; stderr_bytes = 0
        simulated = [bool]$config.test_child_script; sleep_inhibited = $false
    }
    Write-State
    Write-SupervisorLog "Supervisor $PID owns run $($config.run_id), mode=$($config.mode)."

    if (-not $config.test_child_script) {
        try {
            if (-not ('OXBot.Supervisor.PowerGuard' -as [type])) {
                Add-Type -TypeDefinition @'
namespace OXBot.Supervisor {
    public static class PowerGuard {
        [System.Runtime.InteropServices.DllImport("kernel32.dll", SetLastError = true)]
        public static extern uint SetThreadExecutionState(uint flags);
    }
}
'@
            }
            $sleepGuard = [OXBot.Supervisor.PowerGuard]::SetThreadExecutionState([uint32]2147483649) -ne 0
            $state.sleep_inhibited = $sleepGuard
            if (-not $sleepGuard) { Write-SupervisorLog 'Warning: could not request a run-scoped sleep guard.' }
        } catch { Write-SupervisorLog "Sleep guard unavailable: $($_.Exception.Message)" }
        $linuxCompetition = (& wsl.exe -d $config.distribution --exec wslpath -a ($config.competition_root -replace '\\', '/')).Trim()
        if ($LASTEXITCODE -ne 0 -or -not $linuxCompetition) { throw 'Could not resolve the competition directory in WSL.' }
        $linuxOutput = (& wsl.exe -d $config.distribution --exec wslpath -a ($config.output_directory -replace '\\', '/')).Trim()
        if ($LASTEXITCODE -ne 0 -or -not $linuxOutput) { throw 'Could not resolve the output directory in WSL.' }
        $executable = (Get-Command wsl.exe -ErrorAction Stop).Source
        $hoursText = ([double]$config.hours).ToString([Globalization.CultureInfo]::InvariantCulture)
        $microBatch = if ($null -ne $config.micro_batch) { [int]$config.micro_batch } else { 128 }
        if ($microBatch -lt 1 -or $microBatch -gt 4096) { throw 'micro_batch must be between 1 and 4096.' }
        $arguments = @('-d', $config.distribution, '--cd', $linuxCompetition, '--exec', 'env',
            "OXBOT_HOURS=$hoursText", "OXBOT_DEALS=$($config.deals)", "OXBOT_JUDGE_GAMES=$($config.judge_games)",
            "OXBOT_OUT=$linuxOutput", "OXBOT_RUN_ID=$($config.run_id)",
            "OXBOT_MICRO_BATCH=$microBatch",
            "OXBOT_SAFE_CUDA=$($config.safe_cuda)", "OXBOT_EVAL_SEED=$($config.evaluation_seed)",
            'bash', 'scripts/selfplay_eval_5080_wsl.sh', $config.mode)
    } else {
        $executable = $config.host_executable
        $arguments = @('-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
            '-File', $config.test_child_script, '-LaunchPath', $LaunchPath)
    }
    $commandLine = ($arguments | ForEach-Object { ConvertTo-WindowsArgument $_ }) -join ' '
    $child = Start-Process -FilePath $executable -ArgumentList $commandLine -WindowStyle Hidden -PassThru `
        -WorkingDirectory $config.competition_root -RedirectStandardOutput $config.stdout_path -RedirectStandardError $config.stderr_path
    # Keep a handle before the child exits. Windows PowerShell 5.1 otherwise
    # can return a null ExitCode after WaitForExit on Start-Process objects.
    $null = $child.Handle
    $state.child_pid = $child.Id
    $state.child_started_at = $child.StartTime.ToUniversalTime().ToString('o')
    Write-State
    Write-SupervisorLog "Child $($child.Id) started; stdout=$($config.stdout_path); stderr=$($config.stderr_path)"

    while (-not $child.WaitForExit([int]$config.heartbeat_seconds * 1000)) { Write-State }
    # WaitForExit also drains redirected streams before the final report read.
    $child.WaitForExit()
    $child.Refresh()
    $exitCode = $child.ExitCode
    $pipeline = Read-Pipeline
    if ($exitCode -eq 0 -and $pipeline -and $pipeline.status -eq 'completed') {
        Write-Terminal 'completed' $exitCode ''
    } elseif ($exitCode -ne 0) {
        Write-Terminal 'failed' $exitCode "child_exit_$exitCode"
    } else {
        Write-Terminal 'failed' $exitCode 'child_exited_without_matching_completed_pipeline'
    }
} catch {
    [IO.File]::AppendAllText($config.supervisor_stderr_path, ($_ | Out-String), [Text.UTF8Encoding]::new($false))
    if ($lock -and $state) {
        Write-Terminal 'failed' 1 $_.Exception.Message
    } else {
        # In particular, lock contention is an unsuccessful new launch, not
        # a failure of the already-running pipeline.
        Write-Error $_ -ErrorAction Continue
        exit 1
    }
} finally {
    if ($sleepGuard) {
        [void][OXBot.Supervisor.PowerGuard]::SetThreadExecutionState([uint32]2147483648)
        $state.sleep_inhibited = $false
        if ($terminalRecorded) { Write-State }
    }
    if ($lock) {
        if ($state -and -not $terminalRecorded) {
            Write-Terminal 'interrupted' $null 'supervisor_stopped_before_child_terminal_state'
        }
        $lock.Dispose()
    }
}

Show-TerminalNotification
if ($state.status -eq 'completed') { exit 0 }
exit 1

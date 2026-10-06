[CmdletBinding()]
param(
    [ValidateSet('all', 'train', 'eval', 'status')]
    [string]$Mode = 'all',
    [ValidateRange(0.001, 8760)]
    [double]$Hours = 24,
    [ValidateRange(2, 2147483647)]
    [int]$Deals = 1000,
    [ValidateRange(1, 2147483647)]
    [int]$JudgeGames = 200,
    [ValidateRange(1, 4096)]
    [int]$MicroBatch = 128,
    [string]$Out = 'ckpts/dmc-realv2',
    [string]$WarmStart = '',
    [string]$ChampionDir = '',
    [string]$Cf8Model = '',
    [string]$Distribution = 'Ubuntu-24.04',
    [switch]$Foreground,
    [switch]$NoDesktopNotification,
    [ValidateRange(1, 60)]
    [int]$HeartbeatSeconds = 15,
    [Parameter(DontShow)]
    [string]$TestChildScript = '',
    [Parameter(DontShow)]
    [string]$TestScenario = 'success'
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$competitionRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

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

# Start-Process joins ArgumentList itself. Quote for the Windows command-line
# parser, including quotes and trailing backslashes, rather than for a shell.
function ConvertTo-WindowsArgument([string]$Value) {
    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') { return $Value }
    $result = [Text.StringBuilder]::new('"')
    $slashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq '\') { $slashes++; continue }
        if ($character -eq '"') {
            [void]$result.Append(('\' * ($slashes * 2 + 1)))
        } else { [void]$result.Append(('\' * $slashes)) }
        [void]$result.Append($character)
        $slashes = 0
    }
    [void]$result.Append(('\' * ($slashes * 2)))
    [void]$result.Append('"')
    return $result.ToString()
}

function ConvertFrom-ProgressTime($Value) {
    if ($null -eq $Value -or $Value -eq '') { return $null }
    if ($Value -is [ValueType] -and $Value -isnot [DateTime] -and $Value -isnot [DateTimeOffset]) {
        return [DateTimeOffset]::FromUnixTimeMilliseconds([long]([double]$Value * 1000))
    }
    return [DateTimeOffset]$Value
}

if ($Out.StartsWith('/') -and -not $Out.StartsWith('//')) {
    $outputDirectory = (& wsl.exe -d $Distribution --exec wslpath -w $Out).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $outputDirectory) { throw 'Could not resolve Out from WSL.' }
} elseif ([IO.Path]::IsPathRooted($Out)) {
    $outputDirectory = [IO.Path]::GetFullPath($Out)
} else {
    $outputDirectory = [IO.Path]::GetFullPath((Join-Path $competitionRoot ($Out -replace '/', '\')))
}
$statusPath = Join-Path $outputDirectory 'supervisor_status.json'

if ($Mode -eq 'status') {
    if (-not (Test-Path -LiteralPath $statusPath)) {
        [ordered]@{ status = 'not_started'; output_directory = $outputDirectory } | ConvertTo-Json
        return
    }
    $state = Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($state.status -eq 'running') {
        $process = Get-Process -Id $state.supervisor_pid -ErrorAction SilentlyContinue
        $sameProcess = $process -and $process.StartTime.ToUniversalTime().Ticks -eq ([DateTimeOffset]$state.supervisor_started_at).UtcDateTime.Ticks
        $age = ([DateTimeOffset]::UtcNow - [DateTimeOffset]$state.heartbeat_at).TotalSeconds
        if (-not $sameProcess -or $age -gt [Math]::Max(120, 4 * $state.heartbeat_seconds)) {
            # Read-only diagnosis: a lost supervisor does not prove its WSL
            # child also died. The monitor must inspect the recorded child.
            $state.status = 'attention_required'
            $state.notification_required = $true
            $state.failure_reason = if ($sameProcess) { 'supervisor_heartbeat_stale' } else { 'supervisor_missing' }
        }
        if ($state.status -eq 'running' -and $state.stage -in @('train', 'training')) {
            try {
                $progress = $null
                if (Test-Path -LiteralPath $state.progress_path) {
                    $progress = Get-Content -LiteralPath $state.progress_path -Raw -Encoding UTF8 | ConvertFrom-Json
                }
                $stageStarted = if ($state.pipeline_updated_at) { [DateTimeOffset]$state.pipeline_updated_at } else { [DateTimeOffset]$state.started_at }
                if (-not $progress -or $progress.run_id -ne $state.run_id -or $null -eq $progress.updated_at) {
                    if (([DateTimeOffset]::UtcNow - $stageStarted).TotalMinutes -gt 15) {
                        $state.status = 'attention_required'
                        $state.notification_required = $true
                        $state.failure_reason = 'training_progress_missing'
                    }
                } else {
                    $trainerUpdated = ConvertFrom-ProgressTime $progress.updated_at
                    $progressTime = ConvertFrom-ProgressTime $progress.progress_at
                    if (([DateTimeOffset]::UtcNow - $trainerUpdated).TotalMinutes -gt 15) {
                        $state.status = 'attention_required'
                        $state.notification_required = $true
                        $state.failure_reason = 'training_heartbeat_stale'
                    } elseif ($progress.phase -in @('collect', 'learn') -and $progressTime -and
                        ([DateTimeOffset]::UtcNow - $progressTime).TotalMinutes -gt 15) {
                        # Internal evaluation/checkpoint phases have no new
                        # samples or learner steps. Their updated_at heartbeat
                        # is authoritative; only collect/learn require progress.
                        $state.status = 'attention_required'
                        $state.notification_required = $true
                        $state.failure_reason = 'training_progress_stale'
                    }
                }
            } catch {
                $state.status = 'attention_required'
                $state.notification_required = $true
                $state.failure_reason = 'training_progress_unreadable'
            }
        }
    }
    $state | ConvertTo-Json -Depth 12
    return
}

if ($TestChildScript) {
    $TestChildScript = (Resolve-Path -LiteralPath $TestChildScript).Path
    if (-not $NoDesktopNotification) { throw 'TestChildScript requires NoDesktopNotification.' }
}
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$logsDirectory = Join-Path $outputDirectory 'logs'
New-Item -ItemType Directory -Path $logsDirectory -Force | Out-Null
$runId = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
$launchPath = Join-Path $outputDirectory "launch-$runId.json"
$supervisorScript = Join-Path $PSScriptRoot 'selfplay_supervisor_5080.ps1'
$hostExecutable = (Get-Process -Id $PID).Path
$manifest = [ordered]@{
    schema_version = 1; run_id = $runId; created_at = [DateTimeOffset]::UtcNow.ToString('o')
    mode = $Mode; hours = $Hours; deals = $Deals; judge_games = $JudgeGames
    micro_batch = $MicroBatch
    safe_cuda = 1; evaluation_seed = 20261101
    distribution = $Distribution; competition_root = $competitionRoot; output_directory = $outputDirectory
    requested_out = $Out; host_executable = $hostExecutable; supervisor_script = $supervisorScript
    warm_start = $WarmStart; champion_dir = $ChampionDir; cf8_model = $Cf8Model
    launch_path = $launchPath; supervisor_pid = $null; heartbeat_seconds = $HeartbeatSeconds
    launch_backend = if ($Foreground) { 'foreground' } else { 'windows_wmi' }
    stdout_path = Join-Path $logsDirectory "$runId.log"
    stderr_path = Join-Path $logsDirectory "$runId.stderr.log"
    supervisor_log_path = Join-Path $logsDirectory "$runId.supervisor.log"
    supervisor_stderr_path = Join-Path $logsDirectory "$runId.supervisor.stderr.log"
    status_path = $statusPath; run_status_path = Join-Path $logsDirectory "$runId.status.json"
    notification_path = Join-Path $outputDirectory 'notification-needed.json'
    pipeline_path = Join-Path $outputDirectory 'last_pipeline.json'
    progress_path = Join-Path $outputDirectory 'training_progress.json'
    desktop_notification = -not $NoDesktopNotification.IsPresent
    test_child_script = $TestChildScript; test_scenario = $TestScenario
}
Write-AtomicJson $launchPath $manifest

if ($Foreground) {
    & $supervisorScript -LaunchPath $launchPath
    exit $LASTEXITCODE
}

$arguments = @('-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
    '-File', $supervisorScript, '-LaunchPath', $launchPath)
$commandLine = (@($hostExecutable) + $arguments | ForEach-Object { ConvertTo-WindowsArgument $_ }) -join ' '
# WMI creates the worker from its Windows service, outside a Codex/terminal
# job object. Start-Process alone can inherit that job and die on app exit.
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $commandLine; CurrentDirectory = $competitionRoot; ProcessStartupInformation = $startup
}
if ($created.ReturnValue -ne 0) { throw "Windows WMI could not launch the supervisor (code $($created.ReturnValue))." }
$manifest.supervisor_pid = [int]$created.ProcessId
Write-AtomicJson $launchPath $manifest

# Only wait for an acknowledged launch. The detached supervisor owns the
# complete train/eval lifetime after this caller, terminal or chat exits.
$deadline = [DateTimeOffset]::UtcNow.AddSeconds(20)
do {
    if (Test-Path -LiteralPath $statusPath) {
        $state = Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($state.run_id -eq $runId) {
            $state | ConvertTo-Json -Depth 12
            if ($state.status -in @('failed', 'interrupted')) { exit 1 }
            return
        }
    }
    if (-not (Get-Process -Id $manifest.supervisor_pid -ErrorAction SilentlyContinue)) {
        $detail = Get-Content -LiteralPath $manifest.supervisor_stderr_path -Tail 8 -ErrorAction SilentlyContinue
        throw "Supervisor failed before launch acknowledgement. See $($manifest.supervisor_stderr_path): $detail"
    }
    Start-Sleep -Milliseconds 200
} while ([DateTimeOffset]::UtcNow -lt $deadline)
throw "Supervisor launch acknowledgement timed out; inspect $statusPath and $($manifest.supervisor_stderr_path)."

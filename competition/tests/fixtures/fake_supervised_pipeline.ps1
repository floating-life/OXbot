param([Parameter(Mandatory)][string]$LaunchPath)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$config = Get-Content -LiteralPath $LaunchPath -Raw -Encoding UTF8 | ConvertFrom-Json
function Write-Pipeline([string]$Status, [string]$Stage, [int]$Code = 0) {
    $pipeline = [ordered]@{
        run_id = $config.run_id; mode = $config.mode; stage = $Stage; status = $Status
        started_at = [DateTimeOffset]::UtcNow.ToString('o'); updated_at = [DateTimeOffset]::UtcNow.ToString('o')
        finished_at = $null; exit_code = $Code; child_pid = $PID; eval_dir = (Join-Path $config.output_directory 'eval/fake')
    }
    $tempPath = "$($config.pipeline_path).fake.tmp"
    [IO.File]::WriteAllText($tempPath, ($pipeline | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
    if ([IO.File]::Exists($config.pipeline_path)) { [IO.File]::Replace($tempPath, $config.pipeline_path, [NullString]::Value) }
    else { [IO.File]::Move($tempPath, $config.pipeline_path) }
}
Write-Pipeline 'running' 'train'
Write-Output "fake run=$($config.run_id); out=$($config.output_directory)"
if ($config.test_scenario -eq 'failure') {
    Write-Pipeline 'failed' 'train' 7
    [Console]::Error.WriteLine('synthetic child failure')
    exit 7
}
if ($config.test_scenario -eq 'incomplete') { exit 0 }
Start-Sleep -Seconds 4
Write-Pipeline 'running' 'eval'
Write-Output 'fake evaluation started'
Start-Sleep -Seconds 4
Write-Pipeline 'completed' 'complete'
Write-Output 'fake pipeline completed'
exit 0

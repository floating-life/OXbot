[CmdletBinding()]
param(
    [switch]$Smoke,
    [string]$Out = 'ckpts/posttrain-v1',
    [int]$Seed = 52000,
    [string]$Distribution = 'Ubuntu-24.04',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'
$competitionRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$linuxCompetition = (& wsl.exe -d $Distribution -- wslpath -a ($competitionRoot -replace '\\', '/')).Trim()
if ($LASTEXITCODE -ne 0 -or -not $linuxCompetition) {
    throw 'Could not resolve the competition directory in WSL.'
}
$trainingArgs = @('posttrain.py', '--out', $Out, '--seed', "$Seed")
if ($Smoke) { $trainingArgs += '--smoke' }
if ($ExtraArgs) { $trainingArgs += $ExtraArgs }
& wsl.exe -d $Distribution --cd $linuxCompetition -- /home/ggcle/.venvs/oxbot/bin/python @trainingArgs
exit $LASTEXITCODE

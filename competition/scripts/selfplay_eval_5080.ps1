[CmdletBinding()]
param(
    [ValidateSet('all', 'train', 'eval')]
    [string]$Mode = 'all',
    [double]$Hours = 24,
    [int]$Deals = 1000,
    [string]$Out = 'ckpts/dmc-realv2',
    [string]$Distribution = 'Ubuntu-24.04'
)

$ErrorActionPreference = 'Stop'
$competitionRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$linuxCompetition = (& wsl.exe -d $Distribution -- wslpath -a ($competitionRoot -replace '\\', '/')).Trim()
if ($LASTEXITCODE -ne 0 -or -not $linuxCompetition) {
    throw 'Could not resolve the competition directory in WSL.'
}
& wsl.exe -d $Distribution --cd $linuxCompetition -- env "OXBOT_HOURS=$Hours" "OXBOT_DEALS=$Deals" "OXBOT_OUT=$Out" bash scripts/selfplay_eval_5080_wsl.sh $Mode
exit $LASTEXITCODE

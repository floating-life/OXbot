[CmdletBinding()]
param(
    [string]$Distribution = 'Ubuntu-24.04'
)

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$linuxRoot = (& wsl.exe -d $Distribution -- wslpath -a ($root -replace '\\','/') 2>$null).Trim()
if (-not $linuxRoot) {
    throw "无法将工作区路径转换为 WSL 路径：$root"
}

& wsl.exe -d $Distribution -- bash -lc "cd '$linuxRoot' && bash tools/build_wsl.sh"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

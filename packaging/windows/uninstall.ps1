<#
Hard Subtitle Extractor (HSE) - uninstall / restore previous version.

  uninstall.ps1                 앱 폴더와 바로가기 제거 (백업 폴더·사용자 데이터는 유지)
  uninstall.ps1 -Restore        현재 버전을 지우지 않고 보관한 뒤, 가장 최근 백업으로 되돌림
  uninstall.ps1 -RemoveRuntime  설치 때 받은 uv/Python 런타임 폴더도 제거
  uninstall.ps1 -DryRun         실제 변경 없이 할 일만 표시

Never removed: your videos/subtitles, %LOCALAPPDATA%\HardSubtitleExtractor (user data),
backup folders "<InstallDir>.backup-*" (delete them yourself when no longer needed).
#>
[CmdletBinding()]
param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'Programs\HardSubtitleExtractor'),
    [switch]$Restore,
    [switch]$RemoveRuntime,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
$InstallDir = [IO.Path]::GetFullPath($InstallDir)
$RuntimeDir = $InstallDir + '-runtime'
$Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$parent = Split-Path $InstallDir -Parent
$leaf = Split-Path $InstallDir -Leaf

function Say([string]$m) { if ($DryRun) { Write-Host "[DryRun] $m" } else { Write-Host $m } }

$shortcuts = @(
    (Join-Path ([Environment]::GetFolderPath('Desktop')) 'Hard Subtitle Extractor.lnk'),
    (Join-Path ([Environment]::GetFolderPath('Programs')) 'Hard Subtitle Extractor.lnk')
)

if ($Restore) {
    $backups = @(Get-ChildItem -LiteralPath $parent -Directory -Filter ($leaf + '.backup-*') -ErrorAction SilentlyContinue | Sort-Object Name -Descending)
    if ($backups.Count -eq 0) { throw "복원할 백업이 없습니다: $InstallDir.backup-*" }
    $latest = $backups[0].FullName
    if (Test-Path -LiteralPath $InstallDir) {
        $parked = $InstallDir + '.replaced-' + $Stamp
        Say "현재 버전 보관: $parked"
        if (-not $DryRun) { Move-Item -LiteralPath $InstallDir -Destination $parked }
    }
    Say "백업 복원: $latest -> $InstallDir"
    if (-not $DryRun) { Move-Item -LiteralPath $latest -Destination $InstallDir }
    Say '복원 완료. 바로가기는 그대로 사용할 수 있습니다.'
    exit 0
}

foreach ($s in $shortcuts) {
    if (Test-Path -LiteralPath $s) {
        Say "바로가기 제거: $s"
        if (-not $DryRun) { Remove-Item -LiteralPath $s -Force }
    }
}
if (Test-Path -LiteralPath $InstallDir) {
    # Safety: only remove a folder that looks like an HSE install.
    if (-not (Test-Path -LiteralPath (Join-Path $InstallDir 'launch_hse.pyw'))) {
        throw "HSE 설치 폴더로 보이지 않아 중단합니다: $InstallDir"
    }
    Say "앱 폴더 제거: $InstallDir"
    if (-not $DryRun) { Remove-Item -LiteralPath $InstallDir -Recurse -Force }
} else {
    Say "앱 폴더 없음: $InstallDir"
}
if ($RemoveRuntime -and (Test-Path -LiteralPath $RuntimeDir)) {
    Say "런타임 제거: $RuntimeDir"
    if (-not $DryRun) { Remove-Item -LiteralPath $RuntimeDir -Recurse -Force }
}
Say '사용자 데이터(%LOCALAPPDATA%\HardSubtitleExtractor), 영상·자막, 백업 폴더는 남겨 두었습니다.'

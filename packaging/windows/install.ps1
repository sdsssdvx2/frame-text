<#
Hard Subtitle Extractor (HSE) - Windows installer / updater (per-user, no admin).

What it does (all inside the current user profile, no system changes):
  1. Downloads the pinned uv tool (SHA-256 verified) and a private Python 3.12.
  2. Builds a NEW app folder next to the current one ("staging"), installs the
     exact pinned Python packages, copies bundled FFmpeg/FFprobe/yt-dlp
     (SHA-256 verified), and downloads the pinned OCR models (SHA-256 verified,
     re-using the models of an existing install when they still verify).
  3. Runs the HSE self-test in the staging folder.
  4. Only if everything succeeded: renames the current install to
     "<InstallDir>.backup-<timestamp>" and moves the new one into place.
     On any failure the current install is left untouched.
  5. Creates Desktop / Start-menu shortcuts.

Never touched: your videos, subtitle outputs, %LOCALAPPDATA%\HardSubtitleExtractor
(managed tools / settings), browser profiles.

Usage:
  Install-HSE.cmd                       (double-click)
  powershell -ExecutionPolicy Bypass -File install.ps1 [-DryRun] [-InstallDir <path>]
                                        [-NoShortcut] [-SkipModels] [-SkipSeconv]
#>
[CmdletBinding()]
param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'Programs\HardSubtitleExtractor'),
    [switch]$DryRun,
    [switch]$NoShortcut,
    [switch]$SkipModels,
    [switch]$SkipSeconv
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-StrictMode -Version 2.0
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch { }

# ---------------------------------------------------------------- pins
$UvVersion   = '0.11.32'
$UvUrl       = "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
$UvZipSha256 = 'acfde570451cfdb8689fa159a138ee805ba4e241c466432750302c86254b0984'
$PythonSpec  = '3.12'
$BinarySha256 = @{
    'ffmpeg.exe'  = 'a4073d6e566266c8bc240a191bedfca66132592f5ae033313bbbe1b8d2de0bb5'
    'ffprobe.exe' = 'efe458c219a4f0ed53b50c65abc53e49ebfff94029e1cf9504d920ae7af78463'
    'yt-dlp.exe'  = 'f5389d3698ecde4ef06f4e6b1ce0294fe9ac1396228c8990b8d3416d6d177a7d'
}
# Fallback download if the release ZIP was built without bundled FFmpeg.
$FfmpegZipUrl    = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-11-01-12-49/ffmpeg-n9.0.2-42-gd7643a0fe3-win64-lgpl-9.0.zip'
$FfmpegZipSha256 = '187fef3a0221148eaf9c342c82309209277d84b2257efcac7d230687b77f2862'
$YtDlpUrl        = 'https://github.com/yt-dlp/yt-dlp/releases/download/2026.10.26/yt-dlp.exe'
$SeconvUrl       = 'https://github.com/SubtitleEdit/subtitleedit/releases/download/v5.3.0/SeConv-Windows-x64.zip'
$SeconvZipSha256 = '1bcafcd5fd84d445b92d33dbfef3b95d268e5d36df34272c003d83d1f37b72f7'

# ---------------------------------------------------------------- paths
$PackageRoot = $PSScriptRoot                       # extracted release folder
$AppPayload  = Join-Path $PackageRoot 'app'        # src, models/manifest.json, tools, ...
$InstallDir  = [IO.Path]::GetFullPath($InstallDir)
$RuntimeDir  = $InstallDir + '-runtime'            # uv + private Python (shared by versions)
$Stamp       = Get-Date -Format 'yyyyMMdd-HHmmss'
$StageDir    = $InstallDir + '.new-' + $Stamp
$BackupDir   = $InstallDir + '.backup-' + $Stamp
$LogFile     = Join-Path $env:TEMP ("HSE-install-$Stamp.log")

function Write-Step([string]$Message) {
    $line = '[' + (Get-Date -Format 'HH:mm:ss') + '] ' + $Message
    Write-Host $line
    Add-Content -LiteralPath $LogFile -Value $line -Encoding UTF8
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-Sha256([string]$Path, [string]$Expected, [string]$Label) {
    $actual = Get-Sha256 $Path
    if ($actual -ne $Expected.ToLowerInvariant()) {
        throw "$Label SHA-256 불일치 (expected $Expected, got $actual)"
    }
}

function Save-Download([string]$Url, [string]$Destination, [string]$Expected, [string]$Label) {
    if ((Test-Path -LiteralPath $Destination) -and ((Get-Sha256 $Destination) -eq $Expected)) {
        Write-Step "$Label : 캐시 사용 (SHA-256 확인됨)"
        return
    }
    $tmp = $Destination + '.part'
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            Write-Step "$Label 다운로드 중 ($attempt/3)"
            Invoke-WebRequest -Uri $Url -OutFile $tmp -UseBasicParsing -MaximumRedirection 10
            Assert-Sha256 $tmp $Expected $Label
            Move-Item -LiteralPath $tmp -Destination $Destination -Force
            return
        } catch {
            Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
            if ($attempt -eq 3) { throw }
            Start-Sleep -Seconds (3 * $attempt)
        }
    }
}

function Invoke-Checked([string]$Exe, [string[]]$Arguments, [string]$Label) {
    Write-Step "$Label"
    # Windows PowerShell 5.1 turns native stderr lines into terminating
    # errors under 'Stop' when merged with 2>&1; uv/pip log progress there.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & $Exe @Arguments 2>&1 | ForEach-Object { "$_" }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    $output | ForEach-Object { Add-Content -LiteralPath $LogFile -Value ("    " + $_) -Encoding UTF8 }
    if ($code -ne 0) {
        $tail = ($output | Select-Object -Last 15) -join [Environment]::NewLine
        throw "$Label 실패 (exit $code)`n$tail"
    }
}

function New-Shortcut([string]$Path, [string]$Target, [string]$Arguments, [string]$WorkDir) {
    $shell = New-Object -ComObject WScript.Shell
    $lnk = $shell.CreateShortcut($Path)
    $lnk.TargetPath = $Target
    $lnk.Arguments = $Arguments
    $lnk.WorkingDirectory = $WorkDir
    $lnk.Description = 'Hard Subtitle Extractor'
    $lnk.Save()
}

# ---------------------------------------------------------------- preflight
Write-Step "HSE 설치 시작 (로그: $LogFile)"
if (-not [Environment]::Is64BitOperatingSystem) { throw '64비트 Windows가 필요합니다.' }
if (-not (Test-Path -LiteralPath (Join-Path $AppPayload 'src\hardsub_extractor\gui_entry.py'))) {
    throw "설치 파일이 불완전합니다: $AppPayload\src 가 없습니다. ZIP을 완전히 압축 해제한 뒤 다시 실행하세요."
}
$existing = Test-Path -LiteralPath $InstallDir
Write-Step ("설치 위치: $InstallDir" + $(if ($existing) { ' (기존 설치 발견 → 성공 시 백업 후 교체)' } else { ' (신규 설치)' }))

if ($DryRun) {
    Write-Step '[DryRun] 실제 변경 없이 계획만 출력합니다.'
    Write-Step "[DryRun] 런타임 폴더: $RuntimeDir (uv $UvVersion, Python $PythonSpec)"
    Write-Step "[DryRun] 새 버전 준비 폴더: $StageDir"
    if ($existing) { Write-Step "[DryRun] 기존 설치 백업 위치: $BackupDir" }
    Write-Step '[DryRun] 바로가기: 바탕화면, 시작 메뉴 "Hard Subtitle Extractor"'
    Write-Step '[DryRun] 사용자 데이터(%LOCALAPPDATA%\HardSubtitleExtractor, 영상, 자막)는 변경하지 않습니다.'
    exit 0
}

$success = $false
try {
    # ------------------------------------------------------------ runtime (uv + Python)
    New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
    $cacheDir = Join-Path $RuntimeDir 'downloads'
    New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
    $uvExe = Join-Path $RuntimeDir "uv-$UvVersion\uv.exe"
    if (-not (Test-Path -LiteralPath $uvExe)) {
        $uvZip = Join-Path $cacheDir "uv-$UvVersion.zip"
        Save-Download $UvUrl $uvZip $UvZipSha256 'uv'
        $uvTmp = Join-Path $RuntimeDir ("uv-extract-" + $Stamp)
        Expand-Archive -LiteralPath $uvZip -DestinationPath $uvTmp -Force
        New-Item -ItemType Directory -Force -Path (Split-Path $uvExe) | Out-Null
        Copy-Item -LiteralPath (Join-Path $uvTmp 'uv.exe') -Destination $uvExe -Force
        Remove-Item -LiteralPath $uvTmp -Recurse -Force -ErrorAction SilentlyContinue
    }
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $RuntimeDir 'python'
    $env:UV_CACHE_DIR = Join-Path $RuntimeDir 'uv-cache'
    $env:UV_NO_CONFIG = '1'
    $env:UV_PYTHON_PREFERENCE = 'only-managed'
    Invoke-Checked $uvExe @('python', 'install', $PythonSpec) "Python $PythonSpec 준비"

    # ------------------------------------------------------------ stage new app folder
    Write-Step "새 버전 준비: $StageDir"
    New-Item -ItemType Directory -Force -Path $StageDir | Out-Null
    Get-ChildItem -LiteralPath $AppPayload -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $StageDir -Recurse -Force
    }
    Copy-Item -LiteralPath (Join-Path $PackageRoot 'launch_hse.pyw') -Destination $StageDir -Force
    $venv = Join-Path $StageDir '.venv'
    Invoke-Checked $uvExe @('venv', '--relocatable', '--python', $PythonSpec, $venv) '가상환경 생성'
    $py = Join-Path $venv 'Scripts\python.exe'
    $pyw = Join-Path $venv 'Scripts\pythonw.exe'
    Invoke-Checked $uvExe @('pip', 'install', '--python', $py, '-r', (Join-Path $PackageRoot 'requirements-windows.txt')) '고정 버전 Python 패키지 설치'
    Invoke-Checked $py @('-c', 'import tkinter, cv2, numpy, rapidocr, onnxruntime; print("imports ok")') '필수 모듈 확인'

    # ------------------------------------------------------------ binaries
    $binDir = Join-Path $StageDir 'bin'
    New-Item -ItemType Directory -Force -Path $binDir | Out-Null
    $bundledBin = Join-Path $PackageRoot 'bin'
    foreach ($name in @('ffmpeg.exe', 'ffprobe.exe', 'yt-dlp.exe')) {
        $src = Join-Path $bundledBin $name
        if (Test-Path -LiteralPath $src) {
            Assert-Sha256 $src $BinarySha256[$name] $name
            Copy-Item -LiteralPath $src -Destination (Join-Path $binDir $name) -Force
            Write-Step "$name 복사 (SHA-256 확인됨)"
        }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $binDir 'ffmpeg.exe')) -or -not (Test-Path -LiteralPath (Join-Path $binDir 'ffprobe.exe'))) {
        $ffZip = Join-Path $cacheDir 'ffmpeg-n9.0.2-lgpl.zip'
        Save-Download $FfmpegZipUrl $ffZip $FfmpegZipSha256 'FFmpeg (LGPL)'
        $ffTmp = Join-Path $RuntimeDir ('ffmpeg-extract-' + $Stamp)
        Expand-Archive -LiteralPath $ffZip -DestinationPath $ffTmp -Force
        foreach ($name in @('ffmpeg.exe', 'ffprobe.exe')) {
            $found = Get-ChildItem -LiteralPath $ffTmp -Filter $name -File -Recurse | Select-Object -First 1
            if (-not $found) { throw "FFmpeg 압축에 $name 이 없습니다." }
            Assert-Sha256 $found.FullName $BinarySha256[$name] $name
            Copy-Item -LiteralPath $found.FullName -Destination (Join-Path $binDir $name) -Force
        }
        Remove-Item -LiteralPath $ffTmp -Recurse -Force -ErrorAction SilentlyContinue
    }
    if (-not (Test-Path -LiteralPath (Join-Path $binDir 'yt-dlp.exe'))) {
        try {
            $yt = Join-Path $cacheDir 'yt-dlp-2026.10.26.exe'
            Save-Download $YtDlpUrl $yt $BinarySha256['yt-dlp.exe'] 'yt-dlp'
            Copy-Item -LiteralPath $yt -Destination (Join-Path $binDir 'yt-dlp.exe') -Force
        } catch {
            Write-Step "경고: yt-dlp 준비 실패 → '웹 자막 우선' 모드만 제한됩니다. ($($_.Exception.Message))"
        }
    }
    if (-not $SkipSeconv) {
        # Optional: converts PGS/VobSub image subtitle tracks of local MKVs.
        try {
            $seZip = Join-Path $cacheDir 'SeConv-5.3.0-Windows-x64.zip'
            Save-Download $SeconvUrl $seZip $SeconvZipSha256 'SeConv 5.3 (선택)'
            $seTmp = Join-Path $RuntimeDir ('seconv-extract-' + $Stamp)
            Expand-Archive -LiteralPath $seZip -DestinationPath $seTmp -Force
            $seExe = Get-ChildItem -LiteralPath $seTmp -Filter 'seconv.exe' -File -Recurse | Select-Object -First 1
            if (-not $seExe) { throw 'seconv.exe 없음' }
            $seDest = Join-Path $binDir 'seconv'
            New-Item -ItemType Directory -Force -Path $seDest | Out-Null
            Get-ChildItem -LiteralPath $seExe.DirectoryName -Force | ForEach-Object {
                Copy-Item -LiteralPath $_.FullName -Destination $seDest -Recurse -Force
            }
            Remove-Item -LiteralPath $seTmp -Recurse -Force -ErrorAction SilentlyContinue
        } catch {
            Write-Step "경고: SeConv(선택 기능) 준비 실패 → 이미지형 내장 자막 변환만 제한됩니다. ($($_.Exception.Message))"
        }
    }

    # ------------------------------------------------------------ OCR models
    $modelOut = Join-Path $StageDir 'models\runtime'
    New-Item -ItemType Directory -Force -Path $modelOut | Out-Null
    $oldModels = Join-Path $InstallDir 'models\runtime'
    if (Test-Path -LiteralPath $oldModels) {
        # Re-use; prepare_models.py re-verifies every file's SHA-256 and
        # replaces anything that does not match the manifest.
        Get-ChildItem -LiteralPath $oldModels -File | ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination $modelOut -Force
        }
        Write-Step '기존 설치의 OCR 모델을 재사용 후보로 복사 (SHA-256 재검증 예정)'
    }
    if (-not $SkipModels) {
        Invoke-Checked $py @((Join-Path $StageDir 'tools\prepare_models.py'), '--out', $modelOut, '--manifest', (Join-Path $StageDir 'models\manifest.json'), '--languages', 'ko,en,zh') 'OCR 모델 준비 (한국어/영어/중국어·일본어, SHA-256 검증)'
    }

    # ------------------------------------------------------------ self-test
    $report = Join-Path $StageDir 'install-selftest.json'
    $selfTestArgs = @((Join-Path $StageDir 'launch_hse.pyw'), '--self-test-json', $report)
    Write-Step '자체검사 실행'
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $py @selfTestArgs 2>&1 | ForEach-Object { Add-Content -LiteralPath $LogFile -Value ("    " + "$_") -Encoding UTF8 }
        $selfCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($selfCode -ne 0) {
        $detail = ''
        if (Test-Path -LiteralPath $report) { $detail = Get-Content -LiteralPath $report -Raw -Encoding UTF8 }
        if ($SkipModels -and $selfCode -eq 2) {
            Write-Step '경고: -SkipModels 로 모델 검사가 실패했습니다(예상된 결과). 하드자막 OCR은 모델 준비 전까지 사용할 수 없습니다.'
        } else {
            throw "자체검사 실패 (exit $selfCode). 기존 설치는 그대로 유지됩니다.`n$detail"
        }
    }

    # ------------------------------------------------------------ swap
    if ($existing) {
        Write-Step "기존 설치 백업: $BackupDir"
        Move-Item -LiteralPath $InstallDir -Destination $BackupDir
    }
    try {
        Move-Item -LiteralPath $StageDir -Destination $InstallDir
    } catch {
        if ($existing -and (Test-Path -LiteralPath $BackupDir) -and -not (Test-Path -LiteralPath $InstallDir)) {
            Move-Item -LiteralPath $BackupDir -Destination $InstallDir
            Write-Step '교체 실패 → 기존 설치를 원래 위치로 복구했습니다.'
        }
        throw
    }
    $success = $true

    if (-not $NoShortcut) {
        $target = Join-Path $InstallDir '.venv\Scripts\pythonw.exe'
        $launchArgs = '"' + (Join-Path $InstallDir 'launch_hse.pyw') + '"'
        $desktop = [Environment]::GetFolderPath('Desktop')
        $programs = [Environment]::GetFolderPath('Programs')
        New-Shortcut (Join-Path $desktop 'Hard Subtitle Extractor.lnk') $target $launchArgs $InstallDir
        New-Shortcut (Join-Path $programs 'Hard Subtitle Extractor.lnk') $target $launchArgs $InstallDir
        Write-Step '바로가기 생성: 바탕화면 / 시작 메뉴 "Hard Subtitle Extractor"'
    }

    Write-Step '설치 완료.'
    if ($existing) {
        Write-Step "이전 버전 백업: $BackupDir  (문제 있으면 uninstall.ps1 -Restore 로 되돌릴 수 있음)"
    }
} catch {
    Write-Step ('설치 실패: ' + $_.Exception.Message)
    if (-not $success -and (Test-Path -LiteralPath $StageDir)) {
        Remove-Item -LiteralPath $StageDir -Recurse -Force -ErrorAction SilentlyContinue
        Write-Step '준비 중이던 새 버전 폴더만 정리했습니다. 기존 설치는 변경되지 않았습니다.'
    }
    Write-Step "자세한 로그: $LogFile"
    exit 1
}
exit 0

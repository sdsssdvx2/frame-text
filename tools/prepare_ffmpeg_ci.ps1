param(
    [string]$Destination = (Join-Path (Get-Location) "bin")
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Pinned, immutable-by-name CI/runtime input. Do not replace this with a moving
# "latest" URL: the archive digest is part of the trust boundary.
$provider = "BtbN/FFmpeg-Builds"
$releaseTag = "autobuild-2026-11-01-12-49"
$assetName = "ffmpeg-n9.0.2-42-gd7643a0fe3-win64-lgpl-9.0.zip"
$expectedArchiveSha256 = "187fef3a0221148eaf9c342c82309209277d84b2257efcac7d230687b77f2862"
$downloadUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/$releaseTag/$assetName"

$tempRootBase = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } elseif ($env:TEMP) { $env:TEMP } else { [System.IO.Path]::GetTempPath() }
$tempRoot = Join-Path $tempRootBase "hse-ffmpeg-$releaseTag"
$archivePath = Join-Path $tempRoot $assetName
$extractPath = Join-Path $tempRoot "extract"

New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null

function Get-LowerSha256([string]$Path) {
    return (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$needsDownload = $true
if (Test-Path $archivePath) {
    $cachedSha = Get-LowerSha256 $archivePath
    if ($cachedSha -eq $expectedArchiveSha256) {
        $needsDownload = $false
        Write-Host "Using verified cached FFmpeg archive: $assetName"
    }
    else {
        Write-Warning "Discarding cached FFmpeg archive with unexpected SHA-256: $cachedSha"
        Remove-Item -Force $archivePath
    }
}

if ($needsDownload) {
    $downloaded = $false
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            Write-Host "Downloading pinned FFmpeg archive (attempt $attempt/3)..."
            Invoke-WebRequest -Uri $downloadUrl -OutFile $archivePath -MaximumRedirection 10
            $downloaded = $true
            break
        }
        catch {
            Remove-Item -Force $archivePath -ErrorAction SilentlyContinue
            if ($attempt -eq 3) { throw }
            Start-Sleep -Seconds (3 * $attempt)
        }
    }
    if (-not $downloaded) { throw "Pinned FFmpeg archive download did not complete" }
}

$archiveSha = Get-LowerSha256 $archivePath
if ($archiveSha -ne $expectedArchiveSha256) {
    throw "FFmpeg archive SHA-256 mismatch. Expected $expectedArchiveSha256, got $archiveSha"
}

Remove-Item -Recurse -Force $extractPath -ErrorAction SilentlyContinue
Expand-Archive -LiteralPath $archivePath -DestinationPath $extractPath -Force

$ffmpegSource = Get-ChildItem -Path $extractPath -Filter "ffmpeg.exe" -File -Recurse | Select-Object -First 1
$ffprobeSource = Get-ChildItem -Path $extractPath -Filter "ffprobe.exe" -File -Recurse | Select-Object -First 1
if (-not $ffmpegSource) { throw "Pinned archive did not contain ffmpeg.exe" }
if (-not $ffprobeSource) { throw "Pinned archive did not contain ffprobe.exe" }

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$ffmpegDest = Join-Path $Destination "ffmpeg.exe"
$ffprobeDest = Join-Path $Destination "ffprobe.exe"
Copy-Item -Force $ffmpegSource.FullName $ffmpegDest
Copy-Item -Force $ffprobeSource.FullName $ffprobeDest

$ffmpegOutput = @(& $ffmpegDest -version 2>&1)
if ($LASTEXITCODE -ne 0) { throw "Pinned ffmpeg.exe smoke check failed" }
$ffprobeOutput = @(& $ffprobeDest -version 2>&1)
if ($LASTEXITCODE -ne 0) { throw "Pinned ffprobe.exe smoke check failed" }

$configLine = $ffmpegOutput | Where-Object { "$_" -match "^configuration:" } | Select-Object -First 1
if ($configLine -and "$configLine" -match "--enable-gpl(?:\s|$)") {
    throw "Pinned FFmpeg unexpectedly reports --enable-gpl"
}

$destinationResolved = (Resolve-Path $Destination).Path
if ($env:GITHUB_PATH) {
    $destinationResolved | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
}

$artifactsDir = Join-Path (Get-Location) "artifacts"
New-Item -ItemType Directory -Force -Path $artifactsDir | Out-Null
$provenance = [ordered]@{
    provider = $provider
    release_tag = $releaseTag
    asset = $assetName
    license_variant = "LGPL"
    archive_sha256 = $archiveSha
    ffmpeg_sha256 = Get-LowerSha256 $ffmpegDest
    ffprobe_sha256 = Get-LowerSha256 $ffprobeDest
    ffmpeg_version = if ($ffmpegOutput.Count -gt 0) { "$($ffmpegOutput[0])" } else { "unknown" }
    ffprobe_version = if ($ffprobeOutput.Count -gt 0) { "$($ffprobeOutput[0])" } else { "unknown" }
    configuration = if ($configLine) { "$configLine" } else { $null }
}
$provenance | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $artifactsDir "ffmpeg-provenance.json")

Write-Host $provenance.ffmpeg_version
Write-Host $provenance.ffprobe_version
Write-Host "Verified FFmpeg archive SHA-256: $archiveSha"

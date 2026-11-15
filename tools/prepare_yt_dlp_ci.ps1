# Prepare the exact official yt-dlp Windows executable bundled with HSE.
# No browser, credentials, user PATH downloader or site access is involved.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$release = '2026.10.26'
$asset = 'yt-dlp.exe'
$expected = 'f5389d3698ecde4ef06f4e6b1ce0294fe9ac1396228c8990b8d3416d6d177a7d'
$url = "https://github.com/yt-dlp/yt-dlp/releases/download/$release/$asset"
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$bin = Join-Path $root 'bin'
$dest = Join-Path $bin $asset
$artifactDir = Join-Path $root 'artifacts'
New-Item -ItemType Directory -Force -Path $bin, $artifactDir | Out-Null

function Assert-Hash([string]$path) {
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) {
        throw "Pinned yt-dlp archive hash mismatch: expected $expected, got $actual"
    }
    return $actual
}

if (Test-Path -LiteralPath $dest) {
    $hash = Assert-Hash $dest
} else {
    $temporary = Join-Path $bin ("yt-dlp.download-" + [guid]::NewGuid().ToString('N') + '.tmp')
    try {
        Invoke-WebRequest -Uri $url -OutFile $temporary -MaximumRedirection 5 -ErrorAction Stop
        $hash = Assert-Hash $temporary
        if (Test-Path -LiteralPath $dest) {
            throw 'Pinned yt-dlp file appeared during download; refusing replacement'
        }
        Move-Item -LiteralPath $temporary -Destination $dest -ErrorAction Stop
    } finally {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -Force
        }
    }
}

# StrictMode does not guarantee LASTEXITCODE exists in a fresh pwsh
# process. Capture the native executable's exit status explicitly.
$global:LASTEXITCODE = -999
$versionLines = @(& $dest --version)
$versionExitCode = $global:LASTEXITCODE
$version = [string]($versionLines | Select-Object -First 1)
if ($versionExitCode -ne 0 -or $version.Trim() -ne $release) {
    throw "Bundled yt-dlp version mismatch or failed execution (exit=$versionExitCode)"
}
[ordered]@{
    provider = 'yt-dlp/yt-dlp'
    url = $url
    release = $release
    filename = $asset
    size_bytes = (Get-Item -LiteralPath $dest).Length
    sha256 = $hash
    license = 'Unmodified upstream standalone binary; upstream bundled licenses apply'
} | ConvertTo-Json -Depth 3 | Set-Content -Encoding UTF8 (Join-Path $artifactDir 'yt-dlp-provenance.json')
Write-Host "YT_DLP_PIN_VERIFIED=$release"
Write-Host "YT_DLP_SHA256=$hash"

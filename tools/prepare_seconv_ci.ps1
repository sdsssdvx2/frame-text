param([string] $ArchivePath = "")
# Pin and stage the FREE, official Subtitle Edit 5.3 portable seconv release.
# Release-build-only; never touches user's applications or system installs.
# One immutable input is downloaded (or verified in runner temp), then a
# complete portable directory is staged under checkout/bin/seconv.
# Do not extract this archive directly into a pre-existing runtime directory.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-StrictMode -Version Latest

$tag = 'v5.3.0'
$asset = 'SeConv-Windows-x64.zip'
$expectedArchive = '1bcafcd5fd84d445b92d33dbfef3b95d268e5d36df34272c003d83d1f37b72f7'
$url = "https://github.com/SubtitleEdit/subtitleedit/releases/download/$tag/$asset"
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$targetRoot = Join-Path $root 'bin'
$target = Join-Path $targetRoot 'seconv'
$tempBase = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } elseif ($env:TEMP) { $env:TEMP } else { [IO.Path]::GetTempPath() }
$cache = Join-Path $tempBase 'hse-pinned-seconv-v5.3.0'
$zip = if ($ArchivePath) { $ArchivePath } else { Join-Path $cache $asset }
$unpacked = Join-Path $cache 'expanded'
$stage = Join-Path $targetRoot ('.seconv-new-' + [guid]::NewGuid().ToString('N'))

function Get-Sha([string] $Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

if (Test-Path -LiteralPath $target) {
    throw 'Existing portable seconv directory must never be overwritten by the release preparer.'
}
New-Item -ItemType Directory -Path $cache,$targetRoot -Force | Out-Null

if (Test-Path -LiteralPath $zip) {
    if ((Get-Sha $zip) -ne $expectedArchive) {
        throw 'Cached seconv archive is present but SHA-256 mismatches the pinned release.'
    }
} elseif ($ArchivePath) {
    throw 'Caller-supplied verified seconv archive is missing.'
} else {
    $download = Join-Path $cache ('download-' + [guid]::NewGuid().ToString('N') + '.tmp')
    try {
        Invoke-WebRequest -Uri $url -OutFile $download -MaximumRedirection 8
        if ((Get-Sha $download) -ne $expectedArchive) {
            throw 'Official seconv release archive SHA-256 mismatch.'
        }
        Move-Item -LiteralPath $download -Destination $zip
    } finally {
        Remove-Item -LiteralPath $download -Force -ErrorAction SilentlyContinue
    }
}
if ((Get-Sha $zip) -ne $expectedArchive) {
    throw 'Pinned seconv archive changed during staging.'
}
if (Test-Path -LiteralPath $unpacked) {
    Remove-Item -LiteralPath $unpacked -Recurse -Force
}
Expand-Archive -LiteralPath $zip -DestinationPath $unpacked
try {
    $all = @(Get-ChildItem -LiteralPath $unpacked -Recurse -Force)
    if (@($all | Where-Object {
        ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
    }).Count -ne 0) {
        throw 'Symlink or reparse point in upstream executable archive is rejected.'
    }
    $matches = @($all | Where-Object { -not $_.PSIsContainer -and $_.Name -ieq 'seconv.exe' })
    if ($matches.Count -ne 1) {
        throw 'Expected exactly one seconv.exe from the official portable archive.'
    }
    $app = $matches[0].DirectoryName
    if (-not $app.StartsWith($unpacked, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Portable executable resolved outside pinned extract directory.'
    }
    New-Item -ItemType Directory -Path $stage | Out-Null
    Get-ChildItem -LiteralPath $app -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $stage -Recurse -Force
    }
    $exe = Join-Path $stage 'seconv.exe'
    if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
        throw 'Missing standalone seconv.exe after copying portable runtime.'
    }
    # Run the official version command from its eventual bundled folder:
    # this also proves all adjacent self-contained .NET dependencies exist.
    $global:LASTEXITCODE = -999
    $ver = @(& $exe --version 2>&1)
    if ($global:LASTEXITCODE -ne 0 -or
        -not (($ver -join ' ') -match '5\.3\.0')) {
        throw 'Free portable seconv executable did not start as version 5.3.0.'
    }
    $files = @(
        Get-ChildItem -LiteralPath $stage -Recurse -File -Force |
        Sort-Object -Property FullName |
        ForEach-Object {
            [ordered]@{
                path = $_.FullName.Substring($stage.Length + 1).Replace('\','/')
                size_bytes = $_.Length
                sha256 = Get-Sha $_.FullName
            }
        }
    )
    if ($files.Count -lt 1 -or $files.Count -gt 3000) {
        throw 'Unexpected number of portable seconv files.'
    }
    $provenance = [ordered]@{
        provider = 'SubtitleEdit/subtitleedit'
        release = $tag
        archive = $asset
        archive_sha256 = $expectedArchive
        portable_exe_sha256 = Get-Sha $exe
        file_count = $files.Count
        files = $files
        free_software = $true
        dependency = 'optional image-subtitle original-track conversion'
        no_installer_or_user_files_changed = $true
    }
    $artifacts = Join-Path $root 'artifacts'
    New-Item -ItemType Directory -Path $artifacts -Force | Out-Null
    $report = Join-Path $artifacts 'seconv-provenance.json'
    $provenance | ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath $report -Encoding UTF8
    if (Test-Path -LiteralPath $target) {
        throw 'Unexpected portable converter target appeared during staging.'
    }
    Move-Item -LiteralPath $stage -Destination $target
    Write-Host "HSE_SECONV_PINNED=PASS; FILES=$($files.Count); FREE=YES; INSTALL=NONE"
} finally {
    if (Test-Path -LiteralPath $stage) {
        Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path -LiteralPath $unpacked) {
        Remove-Item -LiteralPath $unpacked -Recurse -Force -ErrorAction SilentlyContinue
    }
}

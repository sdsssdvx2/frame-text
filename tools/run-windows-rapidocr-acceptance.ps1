$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$venv = Join-Path $root '.venv-ocr-acceptance'
if (-not (Test-Path $venv)) {
    py -3.11 -m venv $venv
}
$python = Join-Path $venv 'Scripts\python.exe'

& $python -m pip install --upgrade pip
& $python -m pip install -e '.[ocr]'
& $python tools\prepare_models.py
& $python tools\make_synthetic_fixture.py
& $python tools\make_stress_fixture.py
& $python -m unittest discover -s tests -v
& $python tools\rapidocr_acceptance.py --assert-quality

Write-Host ''
Write-Host 'RapidOCR acceptance completed.'
Write-Host 'Result: artifacts\rapidocr-acceptance.json'

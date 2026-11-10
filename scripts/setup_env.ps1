# Creates .venv with GPU PyTorch and the pinned project dependencies.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path '.venv\Scripts\python.exe')) { python -m venv .venv }
$py = Join-Path $root '.venv\Scripts\python.exe'

& $py -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }

& $py -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'torch install failed' }

& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'requirements install failed' }

& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

$ErrorActionPreference = "Stop"

Write-Host "[1/4] Kiem tra Python 3.12..."
py -3.12 --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Dang cai Python 3.12 bang Python Install Manager..."
    py install 3.12
    if ($LASTEXITCODE -ne 0) { throw "Khong the cai Python 3.12." }
}

Write-Host "[2/4] Tao moi truong ao..."
if (-not (Test-Path -LiteralPath ".venv")) {
    py -3.12 -m venv .venv
}

Write-Host "[3/4] Cai thu vien..."
& .\.venv\Scripts\python.exe -m pip install --upgrade pip

# PyPI mac dinh co the cai wheel CPU tren Windows. Chon wheel CUDA ro rang
# neu co NVIDIA de pipeline camera khong bi gioi han o ~1 FPS.
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $cudaReady = (& .\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())" 2>$null) -eq "True"
    if (-not $cudaReady) {
        Write-Host "Phat hien NVIDIA GPU - thay PyTorch CPU bang CUDA 12.8..."
        & .\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
        if ($LASTEXITCODE -ne 0) { throw "Khong the cai PyTorch CUDA." }
    } else {
        Write-Host "PyTorch CUDA da san sang."
    }
} else {
    Write-Host "Khong co NVIDIA GPU - su dung PyTorch CPU."
}
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "[4/4] Kiem tra ung dung..."
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -c "import torch; print('PyTorch device:', 'GPU' if torch.cuda.is_available() else 'CPU')"

Write-Host "Hoan tat. Chay: .\start.ps1"

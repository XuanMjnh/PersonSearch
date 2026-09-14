$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    throw "Chua co moi truong. Hay chay .\setup.ps1 truoc."
}
$hasNvidia = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
$hasCuda = (& .\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())") -eq "True"
if ($hasNvidia -and -not $hasCuda) {
    Write-Warning "Dang dung PyTorch CPU du may co NVIDIA. Chay .\setup.ps1 de cai CUDA va tang FPS."
}
& .\.venv\Scripts\python.exe run.py

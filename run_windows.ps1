[CmdletBinding()]
param(
    [ValidateSet("control", "bev", "record")]
    [string]$SensorMode = "control",
    [string]$EnvironmentName = "carla",
    [switch]$SetupOnly,
    [switch]$SkipInstall,
    [switch]$AllowCpu
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda bulunamadi. Miniconda/Anaconda kurup terminali yeniden acin."
}

& conda run -n $EnvironmentName python --version *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[SETUP] '$EnvironmentName' ortami Python 3.12 ile olusturuluyor."
    & conda create -y -n $EnvironmentName python=3.12
    if ($LASTEXITCODE -ne 0) {
        throw "Conda ortami olusturulamadi."
    }
}

if (-not $SkipInstall) {
    & conda run --no-capture-output -n $EnvironmentName `
        python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip guncellenemedi." }

    & conda run -n $EnvironmentName python -c `
        "import sys,torch; ok=torch.cuda.is_available() and ('sm_120' in torch.cuda.get_arch_list()); sys.exit(0 if ok else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[SETUP] RTX 50 serisi icin CUDA 12.8 PyTorch kuruluyor."
        & conda run --no-capture-output -n $EnvironmentName `
            python -m pip install --force-reinstall torch torchvision `
            --index-url https://download.pytorch.org/whl/cu128
        if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch kurulamadi." }
    }

    & conda run --no-capture-output -n $EnvironmentName `
        python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Proje bagimliliklari kurulamadi." }
}

& conda run -n $EnvironmentName python -c `
    "import sys,torch; ok=torch.cuda.is_available() and ('sm_120' in torch.cuda.get_arch_list()); sys.exit(0 if ok else 1)"
$CudaReady = $LASTEXITCODE -eq 0
if (-not $CudaReady -and -not $AllowCpu) {
    throw "RTX GPU PyTorch tarafindan kullanilamiyor. Surucuyu kontrol edin veya -AllowCpu kullanin."
}

& conda run --no-capture-output -n $EnvironmentName `
    python scripts/check_setup.py
if ($LASTEXITCODE -ne 0) { throw "Kurulum kontrolu basarisiz." }

if ($SetupOnly) {
    Write-Host "[OK] Windows kurulumu hazir."
    exit 0
}

$env:SENSOR_MODE = $SensorMode
if (-not $env:VEHICLE_DEVICE) { $env:VEHICLE_DEVICE = "auto" }
if (-not $env:SIGN_DEVICE) { $env:SIGN_DEVICE = "auto" }
if (-not $env:LANE_DEVICE) { $env:LANE_DEVICE = "auto" }
if (-not $env:ENABLE_FP16_INFERENCE) {
    $env:ENABLE_FP16_INFERENCE = "true"
}
$env:CAMERA_WAIT_TIMEOUT_MS = "10"
$env:PERCEPTION_EVERY_N_FRAMES = "1"
$env:VEHICLE_IMAGE_SIZE = "640"
$env:ENABLE_SIGN_DETECTION = "false"
$env:PYTHONUNBUFFERED = "1"

Write-Host "[RUN] Windows | RTX 5070 profili | mode=$SensorMode"
& conda run --no-capture-output -n $EnvironmentName python -u main.py
exit $LASTEXITCODE

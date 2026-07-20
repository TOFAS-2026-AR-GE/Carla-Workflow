#!/usr/bin/env bash

set -euo pipefail

SENSOR_PROFILE="control"
SETUP_ONLY=0
SKIP_INSTALL=0
ALLOW_CPU=0

usage() {
    cat <<'EOF'
Usage: bash run_linux.sh [options]

  --control       Low-latency control mode (default)
  --bev           Enable the full 15-sensor BEV mode
  --record        Enable synchronized sensor recording
  --setup-only    Install and verify dependencies without starting the app
  --skip-install  Skip dependency installation and only verify/run
  --allow-cpu     Continue when CUDA is unavailable
  -h, --help      Show this help
EOF
}

while (($#)); do
    case "$1" in
        --control)
            SENSOR_PROFILE="control"
            ;;
        --bev)
            SENSOR_PROFILE="bev"
            ;;
        --record)
            SENSOR_PROFILE="record"
            ;;
        --setup-only)
            SETUP_ONLY=1
            ;;
        --skip-install)
            SKIP_INSTALL=1
            ;;
        --allow-cpu)
            ALLOW_CPU=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac

    shift
done

# Script hangi klasörden çağrılırsa çağrılsın proje ana dizinine geçer.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Script yeni Conda environment oluşturmaz.
# Terminalde önceden aktif edilmiş environment'ı kullanır.
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "[ERROR] Aktif bir Conda environment bulunamadı." >&2
    echo "Önce environment'ı etkinleştir:" >&2
    echo "  conda activate carla_controller" >&2
    exit 1
fi

ACTIVE_ENVIRONMENT="${CONDA_DEFAULT_ENV:-unknown}"
PYTHON_PATH="${CONDA_PREFIX}/bin/python"

if [[ ! -x "$PYTHON_PATH" ]]; then
    echo "[ERROR] Aktif environment içinde Python bulunamadı:" >&2
    echo "  $PYTHON_PATH" >&2
    exit 1
fi

# Bütün pip işlemleri aktif Conda environment'ın Python'uyla yapılır.
PYTHON=("$PYTHON_PATH")

echo "[INFO] Aktif Conda environment: $ACTIVE_ENVIRONMENT"
echo "[INFO] Kullanılan Python: $PYTHON_PATH"
"${PYTHON[@]}" --version

CUDA_CHECK=(
    -c
    "import sys, torch; ok = torch.cuda.is_available() and ('sm_120' in torch.cuda.get_arch_list()); sys.exit(0 if ok else 1)"
)

if ((SKIP_INSTALL == 0)); then
    echo "[SETUP] Bağımlılıklar '$ACTIVE_ENVIRONMENT' ortamına kuruluyor."

    "${PYTHON[@]}" -m pip install --upgrade pip

    # Torch kurulu değilse veya RTX 50 mimarisini desteklemiyorsa yeniden kurar.
    if ! "${PYTHON[@]}" "${CUDA_CHECK[@]}"; then
        echo "[SETUP] RTX 50 serisi için CUDA 12.8 PyTorch kuruluyor."

        "${PYTHON[@]}" -m pip install --force-reinstall \
            torch \
            torchvision \
            --index-url https://download.pytorch.org/whl/cu128
    fi

    "${PYTHON[@]}" -m pip install -r requirements.txt
else
    echo "[SETUP] --skip-install seçildi; pip kurulumu atlanıyor."
fi

CUDA_READY=1
"${PYTHON[@]}" "${CUDA_CHECK[@]}" || CUDA_READY=0

if ((CUDA_READY == 0 && ALLOW_CPU == 0)); then
    echo "[ERROR] PyTorch RTX GPU'yu kullanamıyor." >&2
    echo "NVIDIA sürücüsünü kontrol et veya --allow-cpu kullan." >&2
    exit 1
fi

if ((CUDA_READY == 1)); then
    echo "[OK] PyTorch CUDA ve RTX 50 mimarisi hazır."
else
    echo "[WARN] CUDA kullanılamıyor; CPU ile devam ediliyor."
fi

"${PYTHON[@]}" scripts/check_setup.py

if ((SETUP_ONLY == 1)); then
    echo "[OK] Linux kurulumu hazır."
    exit 0
fi

export SENSOR_MODE="$SENSOR_PROFILE"
export VEHICLE_DEVICE="auto"
export CAMERA_WAIT_TIMEOUT_MS="10"
export PERCEPTION_EVERY_N_FRAMES="1"
export VEHICLE_IMAGE_SIZE="640"
export PYTHONUNBUFFERED="1"

echo "[RUN] Linux | environment=$ACTIVE_ENVIRONMENT | mode=$SENSOR_PROFILE"

exec "${PYTHON[@]}" -u main.py
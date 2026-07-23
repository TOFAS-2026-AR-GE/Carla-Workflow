#!/usr/bin/env bash

set -euo pipefail

SENSOR_PROFILE="control"
SETUP_ONLY=0
SKIP_INSTALL=0
ALLOW_CPU=0

usage() {
    cat <<'EOF'
Usage: bash run_linux.sh [options]

  --control       Full-sensor BEV-validated control (default)
  --bev           Full-sensor BEV-focused control
  --record        Full-sensor control with synchronized recording
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

    # Torch kurulu değilse veya RTX 50 mimarisini desteklemiyorsa uygun sürümü
    # kurar. --allow-cpu seçildiğinde çalışan bir CPU kurulumu yeniden indirilmez.
    if ! "${PYTHON[@]}" "${CUDA_CHECK[@]}"; then
        if ((ALLOW_CPU == 1)); then
            if "${PYTHON[@]}" -c "import torch, torchvision"; then
                echo "[SETUP] Çalışan CPU PyTorch kurulumu korunuyor."
            else
                echo "[SETUP] CPU PyTorch kuruluyor."
                "${PYTHON[@]}" -m pip install \
                    torch \
                    torchvision \
                    --index-url https://download.pytorch.org/whl/cpu
            fi
        else
            echo "[SETUP] RTX 50 serisi için CUDA 12.8 PyTorch kuruluyor."

            "${PYTHON[@]}" -m pip install --force-reinstall \
                torch \
                torchvision \
                --index-url https://download.pytorch.org/whl/cu128
        fi
    fi

    "${PYTHON[@]}" -m pip install -r requirements.txt

    if "${PYTHON[@]}" -c \
        "from carla_app.config import Settings; raise SystemExit(0 if Settings().enable_lane_detection else 1)"
    then
        echo "[SETUP] Etkin UFLD şerit modeli doğrulanıyor."
        "${PYTHON[@]}" scripts/download_lane_model.py
    fi
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
    DEFAULT_INFERENCE_DEVICE="auto"
else
    echo "[WARN] CUDA kullanılamıyor; varsayılan inference cihazı CPU."
    DEFAULT_INFERENCE_DEVICE="cpu"
fi

"${PYTHON[@]}" scripts/check_setup.py

if ((SETUP_ONLY == 1)); then
    echo "[OK] Linux kurulumu hazır."
    exit 0
fi

export SENSOR_MODE="$SENSOR_PROFILE"
export VEHICLE_DEVICE="${VEHICLE_DEVICE:-$DEFAULT_INFERENCE_DEVICE}"
export SIGN_DEVICE="${SIGN_DEVICE:-$DEFAULT_INFERENCE_DEVICE}"
export LANE_DEVICE="${LANE_DEVICE:-$DEFAULT_INFERENCE_DEVICE}"
export ENABLE_FP16_INFERENCE="${ENABLE_FP16_INFERENCE:-true}"
export ENABLE_SIGN_DETECTION="false"
export PYTHONUNBUFFERED="1"

echo "[RUN] Linux | environment=$ACTIVE_ENVIRONMENT | mode=$SENSOR_PROFILE"

exec "${PYTHON[@]}" -u main.py

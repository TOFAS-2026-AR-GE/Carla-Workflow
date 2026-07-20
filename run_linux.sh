#!/usr/bin/env bash

set -euo pipefail

SENSOR_PROFILE="control"
ENVIRONMENT_NAME="${CONDA_ENV_NAME:-carla}"
SETUP_ONLY=0
SKIP_INSTALL=0
ALLOW_CPU=0

usage() {
    cat <<'EOF'
Usage: bash run_linux.sh [options]

  --control       Low-latency camera/radar/LiDAR control mode (default)
  --bev           Enable the full 15-sensor BEV mode for the RTX 5090
  --record        Enable synchronized sensor recording
  --setup-only    Install and verify dependencies without starting the app
  --skip-install  Skip pip installation and only verify/run
  --allow-cpu     Continue when CUDA is unavailable
  --env NAME      Conda environment name (default: carla)
  -h, --help      Show this help
EOF
}

while (($#)); do
    case "$1" in
        --control) SENSOR_PROFILE="control" ;;
        --bev) SENSOR_PROFILE="bev" ;;
        --record) SENSOR_PROFILE="record" ;;
        --setup-only) SETUP_ONLY=1 ;;
        --skip-install) SKIP_INSTALL=1 ;;
        --allow-cpu) ALLOW_CPU=1 ;;
        --env)
            shift
            [[ $# -gt 0 ]] || { echo "--env requires a name" >&2; exit 2; }
            ENVIRONMENT_NAME="$1"
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if ! command -v conda >/dev/null 2>&1; then
    for profile in \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh"; do
        if [[ -f "$profile" ]]; then
            # shellcheck disable=SC1090
            source "$profile"
            break
        fi
    done
fi

command -v conda >/dev/null 2>&1 || {
    echo "Conda not found. Install Miniconda/Anaconda and reopen the shell." >&2
    exit 1
}

if ! conda run -n "$ENVIRONMENT_NAME" python --version >/dev/null 2>&1; then
    echo "[SETUP] Creating '$ENVIRONMENT_NAME' with Python 3.12."
    conda create -y -n "$ENVIRONMENT_NAME" python=3.12
fi

PYTHON=(conda run --no-capture-output -n "$ENVIRONMENT_NAME" python)
CUDA_CHECK=(
    -c
    "import sys,torch; ok=torch.cuda.is_available() and ('sm_120' in torch.cuda.get_arch_list()); sys.exit(0 if ok else 1)"
)

if ((SKIP_INSTALL == 0)); then
    "${PYTHON[@]}" -m pip install --upgrade pip

    if ! "${PYTHON[@]}" "${CUDA_CHECK[@]}"; then
        echo "[SETUP] Installing CUDA 12.8 PyTorch for the RTX 50 series."
        "${PYTHON[@]}" -m pip install --force-reinstall torch torchvision \
            --index-url https://download.pytorch.org/whl/cu128
    fi

    "${PYTHON[@]}" -m pip install -r requirements.txt
fi

CUDA_READY=1
"${PYTHON[@]}" "${CUDA_CHECK[@]}" || CUDA_READY=0
if ((CUDA_READY == 0 && ALLOW_CPU == 0)); then
    echo "RTX GPU is unavailable to PyTorch. Check the NVIDIA driver or use --allow-cpu." >&2
    exit 1
fi

"${PYTHON[@]}" scripts/check_setup.py

if ((SETUP_ONLY == 1)); then
    echo "[OK] Linux setup is ready."
    exit 0
fi

export SENSOR_MODE="$SENSOR_PROFILE"
export VEHICLE_DEVICE="auto"
export CAMERA_WAIT_TIMEOUT_MS="10"
export PERCEPTION_EVERY_N_FRAMES="1"
export VEHICLE_IMAGE_SIZE="640"
export PYTHONUNBUFFERED="1"

echo "[RUN] Linux | RTX 5090 profile | mode=$SENSOR_PROFILE"
exec "${PYTHON[@]}" -u main.py

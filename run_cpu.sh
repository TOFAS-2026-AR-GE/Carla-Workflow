#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=""
export VEHICLE_DEVICE="cpu"
export SIGN_DEVICE="cpu"

exec python -u main.py

#!/usr/bin/env bash

set -euo pipefail

unset CUDA_VISIBLE_DEVICES

exec python -u main.py
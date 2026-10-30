#!/usr/bin/env bash
# 在独立 uv 环境运行入口，默认选择本机 RTX 3070。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-acf43839-e475-deee-1606-de0fa5b8c26e}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec uv run --frozen "$@"

#!/usr/bin/env bash

# Launch DF-05-01 in the verified openvla310 cloud environment.
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/openvla_efficiency_project/scripts/Simulation/DF-05-01 Lora_Batch12"
OPENVLA_SOURCE_ROOT="/root/autodl-tmp/openvla_efficiency_project/scripts/DF-04-02 Lora_Batch12/src"
LIBERO_ROOT="/root/autodl-tmp/LIBERO"
CONFIG_PATH="${1:-$PROJECT_ROOT/config/simulation_config.yaml}"

# Activate the exact Python environment used for simulation.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate openvla310

# Configure headless rendering, local source imports, and offline model loading.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT:$OPENVLA_SOURCE_ROOT:$LIBERO_ROOT:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"
mkdir -p output/logs output/metrics output/videos

# Preserve launcher output while Python writes its structured simulation log.
python run_simulation.py --config "$CONFIG_PATH" 2>&1 \
  | tee -a output/logs/launcher.stdout.log

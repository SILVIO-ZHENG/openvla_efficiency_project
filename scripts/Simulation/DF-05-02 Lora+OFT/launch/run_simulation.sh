#!/usr/bin/env bash

# Launch DF-05-02 with the exact cloud interpreter and offline dependencies.
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/openvla_efficiency_project/scripts/Simulation/DF-05-02 Lora+OFT"
OPENVLA_SOURCE_ROOT="/root/autodl-tmp/openvla_efficiency_project/scripts/DF-04-03 Lora+OFT/src"
LIBERO_ROOT="/root/autodl-tmp/LIBERO"
PYTHON_BIN="/root/miniconda3/envs/openvla310/bin/python"
DEFAULT_CONFIG="$PROJECT_ROOT/config/simulation_config.yaml"
CONFIG_PATH="$DEFAULT_CONFIG"

# Consume a positional config path while allowing a flag as the first argument.
if [[ $# -gt 0 && "$1" != --* ]]; then
    CONFIG_PATH="$1"
    shift
fi

# Reject a false conda activation before loading any model files.
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Required interpreter is not executable: $PYTHON_BIN" >&2
    exit 1
fi

# Configure headless rendering, exact source imports, and offline model loading.
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

# Preserve the complete console stream while Python writes structured records.
"$PYTHON_BIN" -u run_simulation.py --config "$CONFIG_PATH" "$@" 2>&1 \
    | tee -a output/logs/full_console.log

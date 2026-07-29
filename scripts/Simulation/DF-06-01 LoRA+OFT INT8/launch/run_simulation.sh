#!/usr/bin/env bash

# Launch DF-06-01 with the exact cloud interpreter and INT8 dependencies.
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/openvla_efficiency_project/scripts/Simulation/DF-06-01 LoRA+OFT INT8"
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
unset DISPLAY
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT:$OPENVLA_SOURCE_ROOT:$LIBERO_ROOT:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"
mkdir -p output/logs output/metrics output/reports output/videos

# Prove that bitsandbytes can execute an INT8 linear kernel on the RTX 5090.
"$PYTHON_BIN" - <<'PY'
import importlib.metadata

import torch
from bitsandbytes.nn import Linear8bitLt

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")

layer = Linear8bitLt(
    16,
    8,
    bias=True,
    has_fp16_weights=False,
).to("cuda:0")
inputs = torch.randn(
    2,
    16,
    device="cuda:0",
    dtype=torch.bfloat16,
)
with torch.inference_mode():
    outputs = layer(inputs)
torch.cuda.synchronize()
if outputs.shape != (2, 8) or not torch.isfinite(outputs).all():
    raise RuntimeError("bitsandbytes INT8 kernel smoke test returned invalid output.")

print("DF-06-01 INT8 dependency check: OK")
print("gpu:", torch.cuda.get_device_name(0))
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("bitsandbytes:", importlib.metadata.version("bitsandbytes"))
print("accelerate:", importlib.metadata.version("accelerate"))
PY

# Preserve the complete console stream while Python writes structured records.
"$PYTHON_BIN" -u run_simulation.py --config "$CONFIG_PATH" "$@" 2>&1 \
    | tee -a output/logs/full_console.log

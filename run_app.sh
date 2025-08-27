#!/bin/bash

# Set ROCm environment variables for AMD GPU support
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HCC_AMDGPU_TARGET=gfx1030
export PYTORCH_ROCM_ARCH=gfx1030
export HIP_VISIBLE_DEVICES=0
export ROCM_PATH=/opt/rocm

# Set LiteLLM environment variables
export LITELLM_BASE_URL=http://localhost:4000
export LITELLM_API_KEY=sk-master-123

# Activate virtual environment
source .venv/bin/activate

# Optional: auto-install ROCm wheels if an AMD GPU is detected and CPU torch is present
# if command -v rocminfo >/dev/null 2>&1; then
#   PY_OUT=$(python - << 'PY'
# import torch
# print('CPU_ONLY' if not torch.cuda.is_available() else 'GPU_OK')
# PY
# )
#   if echo "$PY_OUT" | grep -q CPU_ONLY; then
#     echo "[run_app] ROCm detected but CPU-only torch installed. Installing ROCm wheels..."
#     pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
#     pip install --index-url https://download.pytorch.org/whl/rocm6.4 torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 || true
#   fi
# fi

# Run the Streamlit application
streamlit run penandpaper_streamlit.py 
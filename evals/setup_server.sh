#!/usr/bin/env bash
# One-time setup on the GPU server: install vLLM + lm-eval into a virtualenv and download the model.
# Works on vast.ai (reuses its /venv/main) and RunPod or other hosts (creates /workspace/venv).
# Safe to re-run; installs and the download skip what is already there.
#
# Usage: bash evals/setup_server.sh
#   then: source "$(cat /workspace/.venv_path)/bin/activate"
set -euo pipefail

MODEL_DIR=/workspace/models/Qwen3-30B-A3B

if [ -d /venv/main ]; then
  VENV=/venv/main
else
  VENV=/workspace/venv
  [ -d "$VENV" ] || uv venv --python 3.12 "$VENV"
fi
echo "$VENV" > /workspace/.venv_path
source "$VENV/bin/activate"

# FlashInfer compiles vLLM's B200 MoE kernels on first start and needs the cuBLAS/cuRAND headers. Some
# images (vast.ai) leave them out of the CUDA toolkit; install the dev packages matching its version.
if [ ! -f /usr/local/cuda/include/cublasLt.h ] || [ ! -f /usr/local/cuda/include/curand_kernel.h ]; then
  CUDA_VER=$(/usr/local/cuda/bin/nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | tr . -)
  apt-get install -y -q "libcublas-dev-$CUDA_VER" "libcurand-dev-$CUDA_VER"
fi

# vLLM pins its own torch; --torch-backend=auto picks the CUDA build that matches the driver
uv pip install vllm --torch-backend=auto
uv pip install "lm-eval[api]" evaluate hf_xet

python -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda, \
'| vllm', vllm.__version__, '|', torch.cuda.get_device_name(0))"

mkdir -p "$(dirname "$MODEL_DIR")"
hf download Qwen/Qwen3-30B-A3B --local-dir "$MODEL_DIR"
du -sh "$MODEL_DIR"

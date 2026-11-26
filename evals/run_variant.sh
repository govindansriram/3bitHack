#!/usr/bin/env bash
# Serve one checkpoint, run evals against it, then shut the server down (run on the GPU server, venv active).
#
# Usage: evals/run_variant.sh <model_dir> <run_name> [run_evals.py args...]
#   evals/run_variant.sh /workspace/models/Qwen3-30B-A3B-FP8 fp8_qwen \
#     --tasks wikitext mmlu arc_challenge gsm8k humaneval mbpp --full --thinking --allow-code-exec
#
# The server log goes to /workspace/logs/vllm_<run_name>.log. Stop any other vLLM server first: one
# server holds most of the GPU's memory.
set -euo pipefail

model_dir=${1:?model dir}
run_name=${2:?run name}
shift 2
here=$(cd "$(dirname "$0")" && pwd)
log_dir=/workspace/logs
mkdir -p "$log_dir"
server_log="$log_dir/vllm_$run_name.log"

"$here/serve_vllm.sh" "$model_dir" > "$server_log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true' EXIT

echo "waiting for vLLM to serve $model_dir (log: $server_log)"
until curl -sf "http://127.0.0.1:${PORT:-8000}/health" > /dev/null; do
  if ! kill -0 "$server" 2>/dev/null; then
    echo "vLLM exited during startup; see $server_log" >&2
    exit 1
  fi
  sleep 10
done

python "$here/run_evals.py" --run-name "$run_name" --base-url "http://127.0.0.1:${PORT:-8000}" "$@"

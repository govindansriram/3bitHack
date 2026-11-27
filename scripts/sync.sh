#!/usr/bin/env bash
# Copy code to the GPU server, or pull results back. Code lives in this repo; the server only runs it.
#
# Usage:
#   scripts/sync.sh push <port> <user@host>   # repo -> server:/workspace/3bithack (no models, venv or outputs)
#   scripts/sync.sh pull <port> <user@host>   # server:/workspace/3bithack/outputs/evals -> outputs/evals
#
# Set SSH_KEY to use a specific key, e.g. SSH_KEY=~/.ssh/id_ed25519
set -euo pipefail

cmd=${1:?push or pull}
port=${2:?ssh port}
host=${3:?user@host}
remote=/workspace/3bithack
repo=$(cd "$(dirname "$0")/.." && pwd)
ssh_cmd="ssh -p $port${SSH_KEY:+ -i $SSH_KEY}"

case "$cmd" in
  push)
    rsync -az --delete -e "$ssh_cmd" \
      --exclude .git --exclude .venv --exclude models --exclude outputs --exclude __pycache__ \
      "$repo/" "$host:$remote/"
    ;;
  pull)
    mkdir -p "$repo/outputs/evals"
    rsync -az -e "$ssh_cmd" "$host:$remote/outputs/evals/" "$repo/outputs/evals/"
    ;;
  *)
    echo "unknown command: $cmd (use push or pull)" >&2
    exit 1
    ;;
esac

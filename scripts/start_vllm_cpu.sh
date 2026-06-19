#!/usr/bin/env bash
#
# Start vLLM with your chosen configuration.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

set -euo pipefail

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

export VLLM_TARGET_DEVICE=cpu

uv run python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-0.6B-Instruct \
  --device cpu \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096

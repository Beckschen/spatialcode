#!/bin/bash
# Code → Answer: run on a few samples (inference only, no full eval metrics).
# Usage: ./scripts/run_code_to_answer.sh [optional: override QA_JSON or other vars]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$RL_ROOT"

# --- Config (override with env or first arg or edit) ---
# If SGLang runs on another machine (e.g. GPU node), set NODE to that hostname and PORT to scripts/sglang.sh's 3rd arg.
# Example: NODE=h08 PORT=22002 ./scripts/run_code_to_answer.sh /path/to/data.json
NODE="${NODE:-localhost}"
PORT="${PORT:-22002}"
BASE_URL="${BASE_URL:-http://${NODE}:${PORT}/v1}"
QA_JSON="${1:-${QA_JSON:-}}"  # Required: first arg or env QA_JSON
MODEL_NAME="${MODEL_NAME:-qwen3-4b-instruct}"
BACKEND="${BACKEND:-api}"
PROMPT_STYLE="${PROMPT_STYLE:-vsi}"                  # vsi | robospatial
LIMIT="${LIMIT:-1}"                                   # Number of samples to run

if [ -z "$QA_JSON" ]; then
    echo "Error: QA_JSON is required. Set it or pass as first argument."
    echo "Example: $0 /path/to/test/data.json"
    echo "Or:      QA_JSON=/path/to/test/data.json $0"
    exit 1
fi

ARGS=(
    --qa_json "$QA_JSON"
    --model_name "$MODEL_NAME"
    --backend "$BACKEND"
    --base_url "$BASE_URL"
    --prompt_style "$PROMPT_STYLE"
    --limit "$LIMIT"
)

echo "=========================================="
echo "Code → Answer (run samples)"
echo "  QA_JSON   = $QA_JSON"
echo "  MODEL     = $MODEL_NAME"
echo "  BACKEND   = $BACKEND"
echo "  BASE_URL  = $BASE_URL"
echo "  LIMIT     = $LIMIT"
echo "=========================================="
python scripts/run_code_to_answer.py "${ARGS[@]}"

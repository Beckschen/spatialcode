#!/bin/bash
# Code → Answer evaluation: accuracy + per-qa_type metrics.
# Usage: ./scripts/eval_code_to_answer.sh [optional: override QA_JSON or other vars]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$RL_ROOT"

# --- Config (override with env or first arg or edit) ---
# Default: PA (parse_anything_trl) compatible prompt + answer extraction are built-in; no PA_ROOT needed.
# Optional: set PA_ROOT=/path/to/parse_anything_trl to use that repo's prompt/extract instead.
# If SGLang runs on another machine (e.g. GPU node), set NODE and PORT.
# Example: NODE=h08 PORT=22002 ./scripts/eval_code_to_answer.sh /path/to/data.json
NODE="${NODE:-localhost}"
PORT="${PORT:-22002}"
BASE_URL="${BASE_URL:-http://${NODE}:${PORT}/v1}"
QA_JSON="${1:-${QA_JSON:-}}"  # Required: first arg or env QA_JSON
MODEL_NAME="${MODEL_NAME:-qwen3-4b-instruct}"
BACKEND="${BACKEND:-api}"
BATCH_SIZE="${BATCH_SIZE:-32}"                        # Concurrent API requests; larger = faster but server VRAM can grow (reduce if OOM)
MAX_TOKENS="${MAX_TOKENS:-30000}"                     # Match PA eval default; truncation drops "Final Answer" and hurts accuracy
TEMPERATURE="${TEMPERATURE:-0.0}"                     # Match PA eval
QA_TYPE_LIST="${QA_TYPE_LIST:-}"                      # Optional: space-separated qa_type(s); empty = eval all types
PA_ROOT="${PA_ROOT:-}"                                # Optional: path to parse_anything_trl repo (overrides built-in PA prompt/extract)
DATASET="${DATASET:-vsibench}"                         # vsibench | robospatial (used only when PA_ROOT is set)
LIMIT="${LIMIT:-}"                                    # Empty = full set; set to N for first N samples
RESULTS_DIR="${RESULTS_DIR:-./results}"               # Dir for saved results (created if missing)
# Default: save to results/ with timestamp; set OUTPUT="" and OUTPUT_JSON="" to disable
TS=$(date +%Y%m%d_%H%M%S)
_default_json="${RESULTS_DIR}/code_to_answer_${MODEL_NAME}_${TS}.json"
_default_jsonl="${RESULTS_DIR}/code_to_answer_${MODEL_NAME}_${TS}.jsonl"
OUTPUT="${OUTPUT:-$_default_jsonl}"                   # JSONL predictions (default: results/code_to_answer_<ts>.jsonl)
OUTPUT_JSON="${OUTPUT_JSON:-$_default_json}"          # Full results JSON (default: results/code_to_answer_<ts>.json)
DRY_RUN="${DRY_RUN:-}"                                # Set to 1 for dry run (no LLM calls)

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
    --batch_size "$BATCH_SIZE"
)
[ -n "$LIMIT" ] && ARGS+=(--limit "$LIMIT")
ARGS+=(--max_tokens "$MAX_TOKENS" --temperature "$TEMPERATURE")
if [ -n "$QA_TYPE_LIST" ]; then
    ARGS+=(--qa_type)
    for w in $QA_TYPE_LIST; do ARGS+=("$w"); done
fi
[ -n "$PA_ROOT" ] && ARGS+=(--pa_root "$PA_ROOT" --dataset "$DATASET")
[ -n "$OUTPUT" ] && ARGS+=(--output "$OUTPUT")
[ -n "$OUTPUT_JSON" ] && ARGS+=(--output_json "$OUTPUT_JSON")
[ "$DRY_RUN" = "1" ] && ARGS+=(--dry_run)

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Code → Answer evaluation"
echo "  QA_JSON   = $QA_JSON"
echo "  MODEL     = $MODEL_NAME"
echo "  BACKEND   = $BACKEND"
echo "  BASE_URL   = $BASE_URL"
echo "  BATCH_SIZE = $BATCH_SIZE"
[ -n "$QA_TYPE_LIST" ] && echo "  QA_TYPE(s)  = $QA_TYPE_LIST"
if [ -n "$PA_ROOT" ]; then
    echo "  PA_ROOT     = $PA_ROOT (external PA prompt+extract; default is built-in)"
else
    echo "  PA_ROOT     = (not set → using built-in PA-compatible prompt+extract)"
fi
echo "  OUTPUT      = $OUTPUT"
echo "  OUTPUT_JSON = $OUTPUT_JSON"
echo "=========================================="
python scripts/eval_code_to_answer.py "${ARGS[@]}"

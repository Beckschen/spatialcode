SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$RL_ROOT"

python scripts/eval_code_to_answer.py \
  --recompute_from results/code_to_answer_qwen3_4b.json \
  --vsi_bench_only

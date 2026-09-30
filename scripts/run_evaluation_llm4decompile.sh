#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python "${repo_dir}/evaluation/run_evaluation_llm4decompile.py" \
    --model_path "${MODEL_PATH:-llm4decompile-1.3b}" \
    --max_new_tokens 512 \
    --testset_path "${TESTSET_PATH:-${repo_dir}/legacy-test/decompile-eval.json}" \
    --repeat 1 --dtype bfloat16 --port 8080 \
    --max_input_len 8000 --max_total_tokens 8512 \
    --max_batch_prefill_tokens 36000 \
    --num_shards "${NUM_SHARDS:-1}" --num_workers "${NUM_WORKERS:-4}"

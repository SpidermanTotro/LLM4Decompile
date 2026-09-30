#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
root_dir="${ROOT_DIR:-${repo_dir}}"
factory_dir="${LLAMA_FACTORY_DIR:-${root_dir}/../LLaMA-Factory}"
: "${MODEL_PATH:?Set MODEL_PATH to an existing local checkpoint directory}"
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "MODEL_PATH has no config.json" >&2; exit 1; }
[[ -f "${factory_dir}/src/train.py" ]] || { echo "Install LLaMA-Factory or set LLAMA_FACTORY_DIR" >&2; exit 1; }
exp_id="${EXP_ID:-local-decompile-$(date -u +%Y%m%d-%H%M%S)}"
output_dir="${OUTPUT_DIR:-${root_dir}/output_models/${exp_id}}"
if [[ -d "${output_dir}" ]] && [[ -n "$(find "${output_dir}" -mindepth 1 -print -quit)" ]]; then
    echo "OUTPUT_DIR must be empty; use a fresh experiment directory" >&2
    exit 1
fi
mkdir -p "${output_dir}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
launcher=(deepspeed "--master_port=${DEEPSPEED_PORT:-11000}")
if [[ -n "${HOSTFILE:-}" ]]; then launcher+=("--hostfile=${HOSTFILE}"); fi
options=(
    "${factory_dir}/src/train.py"
    --deepspeed "${factory_dir}/examples/deepspeed/ds_z3_config.json"
    --stage sft --do_train --model_name_or_path "${MODEL_PATH}"
    --dataset "${DATASET:-llm4binary_v1}"
    --dataset_dir "${DATASET_DIR:-${root_dir}/train/llama_factory_llm4decompile/data}"
    --template empty --finetuning_type "${FINETUNING_TYPE:-full}"
    --output_dir "${output_dir}" --gradient_checkpointing
    --cutoff_len "${CUTOFF_LEN:-1024}" --max_grad_norm "${MAX_GRAD_NORM:-1.0}"
    --preprocessing_num_workers "${NUM_WORKERS:-4}"
    --per_device_train_batch_size "${BATCH_SIZE:-1}"
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS:-16}"
    --learning_rate "${LEARNING_RATE:-5e-6}" --lr_scheduler_type "${LR_SCHEDULER:-cosine}"
    --logging_steps "${LOGGING_STEPS:-1}" --warmup_ratio "${WARMUP_RATIO:-0.025}"
    --run_name "${exp_id}" --save_steps "${SAVE_STEPS:-20}"
    --save_total_limit "${SAVE_TOTAL_LIMIT:-10}" --flash_attn "${FLASH_ATTN:-auto}"
    --max_samples "${MAX_SAMPLES:-20000000}" --num_train_epochs "${NUM_EPOCHS:-1.0}"
    --report_to "${REPORT_TO:-none}"
)
if [[ "${BF16:-0}" == 1 ]]; then options+=(--bf16); fi
"${launcher[@]}" "${options[@]}" 2>"${output_dir}/train.err" | tee "${output_dir}/train.log"

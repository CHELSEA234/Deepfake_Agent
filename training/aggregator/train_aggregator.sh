#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to a LLaVA-1.5 7B HF ID/path}"
: "${DATA_PATH:?Set DATA_PATH to the training JSON}"
: "${IMAGE_FOLDER:?Set IMAGE_FOLDER to the image root}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the checkpoint destination}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_LLAVA_ROOT="${UPSTREAM_LLAVA_ROOT:-}"
if [[ -z "${UPSTREAM_LLAVA_ROOT}" ]]; then
  echo "UPSTREAM_LLAVA_ROOT must point to an upstream LLaVA checkout." >&2
  exit 2
fi

export PYTHONPATH="${SCRIPT_DIR}:${UPSTREAM_LLAVA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5}"
NUM_PROCESSES="${NUM_PROCESSES:-6}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29840}"
CACHE_DIR="${CACHE_DIR:-}"
REPORT_TO="${REPORT_TO:-none}"
# Optional SUBSTRING=METHOD_NUMBER pairs for tool_selection_reward, e.g. "face=2,editing=1,semantic=3".
DOMAIN_METHOD_MAP="${DOMAIN_METHOD_MAP:-}"

cache_args=()
[[ -n "${CACHE_DIR}" ]] && cache_args+=(--cache_dir "${CACHE_DIR}")

method_map_args=()
[[ -n "${DOMAIN_METHOD_MAP}" ]] && method_map_args+=(--domain_method_map "${DOMAIN_METHOD_MAP}")

accelerate launch \
  --gpu_ids "${GPU_IDS}" \
  --num_processes "${NUM_PROCESSES}" \
  --config_file "${SCRIPT_DIR}/deepspeed_zero3.yaml" \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  "${SCRIPT_DIR}/train.py" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --data_path "${DATA_PATH}" \
  --image_folder "${IMAGE_FOLDER}" \
  --output_dir "${OUTPUT_DIR}" \
  "${cache_args[@]}" \
  "${method_map_args[@]}" \
  --lora_enable true \
  --lora_r 128 \
  --lora_alpha 256 \
  --bf16 true \
  --fp16 false \
  --num_train_epochs 2 \
  --per_device_train_batch_size 6 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 3 \
  --learning_rate 2e-5 \
  --weight_decay 0 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --tf32 true \
  --model_max_length 512 \
  --gradient_checkpointing true \
  --dataloader_num_workers 4 \
  --group_by_modality_length false \
  --report_to "${REPORT_TO}" \
  --num_generations 4 \
  --beta 0.0 \
  --max_grad_norm 1.0

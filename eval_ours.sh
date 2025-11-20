#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=2

BASE_OUTPUT_PATH="./results/llada2"
MODEL_PATH="/scratch/prj/inf_rate/yizhen/LLaDA2-moe-mixEmbed/model"


export BASE_RATE=0.93
export RATE_FLEX=0.07
export COLD_START=2
export DEAN_TOPP=0.98
if_early_stop=1

length=512
task="humaneval"
OUTPUT_PATH="${BASE_OUTPUT_PATH}/${task}_${length}"
accelerate launch evaluation_script.py \
    -m dllm_eval \
    --model LLaDA2 \
    --tasks ${task} \
    --batch_size 1 \
    --model_args "pretrained=${MODEL_PATH},assistant_prefix=<reasoning> " \
    --gen_kwargs "block_length=32,gen_length=${length},steps=32,temperature=0,eos_early_stop=${if_early_stop}" \
    --num_fewshot 0 \
    --output_path "${OUTPUT_PATH}" \
    --log_samples \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --confirm_run_unsafe_code 

python metrics/humaneval.py \
    --model_path "${MODEL_PATH}" \
    --res_path "${OUTPUT_PATH}"

exit 0

export BASE_RATE=1
export RATE_FLEX=0
export COLD_START=0
export DEAN_TOPP=0
if_early_stop=1

task="gsm8k"
length=512
OUTPUT_PATH="${BASE_OUTPUT_PATH}/${task}_${length}"
accelerate launch evaluation_script.py \
    -m dllm_eval \
    --model LLaDA2 \
    --tasks ${task} \
    --batch_size 1 \
    --model_args "pretrained=${MODEL_PATH},assistant_prefix=<reasoning> " \
    --gen_kwargs "block_length=32,gen_length=${length},steps=32,temperature=0,eos_early_stop=False" \
    --num_fewshot 0 \
    --output_path "${OUTPUT_PATH}" \
    --log_samples \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --confirm_run_unsafe_code

python metrics/gsm8k.py \
    --model_path "${MODEL_PATH}" \
    --res_path "${OUTPUT_PATH}"


export BASE_RATE=1
export RATE_FLEX=0
export COLD_START=0
export DEAN_TOPP=0
if_early_stop=1

task="mbpp"
length=512
OUTPUT_PATH="${BASE_OUTPUT_PATH}/${task}_${length}"
accelerate launch evaluation_script.py \
    -m dllm_eval \
    --model LLaDA2 \
    --tasks ${task} \
    --batch_size 1 \
    --model_args "pretrained=${MODEL_PATH},assistant_prefix=<reasoning> " \
    --gen_kwargs "block_length=32,gen_length=${length},steps=32,temperature=0,eos_early_stop=False" \
    --num_fewshot 0 \
    --output_path "${OUTPUT_PATH}" \
    --log_samples \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --confirm_run_unsafe_code 

python metrics/mbpp.py \
    --model_path "${MODEL_PATH}" \
    --res_path "${OUTPUT_PATH}"


export BASE_RATE=1
export RATE_FLEX=0
export COLD_START=0
export DEAN_TOPP=0
if_early_stop=1

task="math500"
length=512
OUTPUT_PATH="${BASE_OUTPUT_PATH}/${task}_${length}"
accelerate launch evaluation_script.py \
    -m dllm_eval \
    --model LLaDA2 \
    --tasks ${task} \
    --batch_size 1 \
    --model_args "pretrained=${MODEL_PATH},assistant_prefix=<reasoning> " \
    --gen_kwargs "block_length=32,gen_length=${length},steps=32,temperature=0,eos_early_stop=False" \
    --num_fewshot 0 \
    --output_path "${OUTPUT_PATH}" \
    --log_samples \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --confirm_run_unsafe_code 

python metrics/math500.py \
    --model_path "${MODEL_PATH}" \
    --res_path "${OUTPUT_PATH}"
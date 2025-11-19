#!/bin/bash
set -e


# 下载模型（如果需要）
echo ">>> Checking and downloading model..."
SAS_KEY="?sv=2025-07-05&spr=https&st=2025-11-19T08%3A28%3A39Z&se=2025-11-26T08%3A28%3A00Z&skoid=a12cc5a8-0e98-40bd-9d94-db9a6a393450&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2025-11-19T08%3A28%3A39Z&ske=2025-11-26T08%3A28%3A00Z&sks=b&skv=2025-07-05&sr=c&sp=racwdxltf&sig=gczDZEWZvoaTC5rKcnuh%2FKG6mAB1a0B2oO8TDsYliyA%3D"

# 下载模型（如果需要）
if [ ! -d "LLaDA2.0-mini-preview" ]; then
    echo ">>> Downloading model files..."
    ./azcopy cp --recursive "https://shuwangmain.blob.core.windows.net/qinglinzhu/models/LLaDA2.0-mini-preview/${SAS_KEY}" ./
fi


BASE_OUTPUT_PATH="/mnt/blob-openpai-shuailu1-out/qinglin/iclr/LLADA2_512/"
MODEL_PATH="LLaDA2.0-mini-preview"
num_processes_num=8
length=512
task="humaneval"
OUTPUT_PATH="${BASE_OUTPUT_PATH}/${task}_${length}"
accelerate launch --num_processes=${num_processes_num} evaluation_script.py \
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

python metrics/humaneval.py \
    --model_path "${MODEL_PATH}" \
    --res_path "${OUTPUT_PATH}"

task="gsm8k"
length=512
OUTPUT_PATH="${BASE_OUTPUT_PATH}/${task}_${length}"
accelerate launch --num_processes=${num_processes_num} evaluation_script.py \
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


task="mbpp"
length=512
OUTPUT_PATH="${BASE_OUTPUT_PATH}/${task}_${length}"
accelerate launch --num_processes=${num_processes_num} evaluation_script.py \
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



task="math500"
length=512
OUTPUT_PATH="${BASE_OUTPUT_PATH}/${task}_${length}"
accelerate launch --num_processes=${num_processes_num} evaluation_script.py \
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
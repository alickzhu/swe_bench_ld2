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

# 批量运行LLaDA2评估脚本
echo ">>> Starting batch LLaDA2 evaluation with multiple parameter configurations..."


BASE_OUTPUT_PATH="/mnt/blob-openpai-shuailu1-out/qinglin/iclr/LLADA2_256/"
MODEL_PATH="LLaDA2.0-mini-preview"

# 创建输出目录
mkdir -p $BASE_OUTPUT_PATH

echo ">>> Environment setup completed. Starting evaluation runs..."

# 定义参数配置数组 - 参数搜索
# BASE_RATE: 0.95, 0.93
# RATE_FLEX: 0.07, 0.1
# COLD_START: 2, 6
# DEAN_TOPP: 0.9, 0.95
# if_early_stop: 1 (固定)
configs=(
    "0.93 0.07 2 0.9 1"
    "0.93 0.07 6 0.9 1"
    "0.9 0 6 0.9 1"
    "0.85 0 20 0.95 1"
)

# 定义任务列表
tasks=("humaneval" "gsm8k" "mbpp" "math500")

# 循环执行评估任务的函数
run_llada2_evaluation() {
    local BASE_RATE=$1
    local RATE_FLEX=$2
    local COLD_START=$3
    local DEAN_TOPP=$4
    local if_early_stop=$5

    # 设置当前运行的环境变量
    export BASE_RATE=$BASE_RATE
    export RATE_FLEX=$RATE_FLEX
    export COLD_START=$COLD_START
    export DEAN_TOPP=$DEAN_TOPP

    # 拼接参数标签
    param_tag="BR${BASE_RATE}_RF${RATE_FLEX}_CS${COLD_START}_TP${DEAN_TOPP}_ES${if_early_stop}"
    echo ">>> Running LLaDA2 evaluation with configuration: $param_tag"

    # 遍历所有任务
    for task_num in "${!tasks[@]}"; do
        task="${tasks[$task_num]}"
        task_display_num=$((task_num + 1))

        echo ">>> Task ${task_display_num}/${#tasks[@]}: ${task}"

        # 动态生成输出路径（包含参数标签）
        length=256
        OUTPUT_PATH="${BASE_OUTPUT_PATH}/${task}_${length}_${param_tag}"

        if [ -d "$OUTPUT_PATH" ]; then
            echo ">>> Skip ${task} (already exists: $OUTPUT_PATH)"
            continue
        fi

        echo ">>> Running ${task} evaluation..."

        # 根据任务设置eos_early_stop参数
        if [ "$task" = "humaneval" ]; then
            eos_param="eos_early_stop=${if_early_stop}"
        else
            eos_param="eos_early_stop=False"
        fi

        # 执行评估
        accelerate launch --num_processes=8 --main_process_port 12345 evaluation_script.py \
            -m dllm_eval \
            --model LLaDA2 \
            --tasks "${task}" \
            --batch_size 1 \
            --model_args "pretrained=${MODEL_PATH},assistant_prefix=<reasoning> " \
            --gen_kwargs "block_length=32,gen_length=${length},steps=32,temperature=0,${eos_param}" \
            --num_fewshot 0 \
            --output_path "${OUTPUT_PATH}" \
            --log_samples \
            --apply_chat_template \
            --fewshot_as_multiturn \
            --confirm_run_unsafe_code

        if [ $? -eq 0 ]; then
            echo ">>> ${task} evaluation completed successfully!"

            # 运行对应的metrics脚本
            echo ">>> Running metrics for ${task}..."
            python metrics/${task}.py \
                --model_path "${MODEL_PATH}" \
                --res_path "${OUTPUT_PATH}"

            if [ $? -eq 0 ]; then
                echo ">>> ${task} metrics completed successfully!"
            else
                echo ">>> ${task} metrics failed!"
                return 1
            fi
        else
            echo ">>> ${task} evaluation failed!"
            return 1
        fi

        # 在任务之间添加短暂延迟
        if [ $task_num -lt $((${#tasks[@]} - 1)) ]; then
            echo ">>> Waiting 10 seconds before next task..."
            sleep 10
        fi
    done

    echo ">>> All evaluation tasks completed for configuration: ${param_tag}"
    echo ">>> Results saved in: ${BASE_OUTPUT_PATH}/"
    return 0
}

# 开始批量运行
total_configs=${#configs[@]}
failed_configs=()

for i in "${!configs[@]}"; do
    config_num=$((i + 1))
    echo ">>> ===== Configuration ${config_num}/${total_configs} ====="

    # 解析参数
    read -ra params <<< "${configs[$i]}"
    BASE_RATE=${params[0]}
    RATE_FLEX=${params[1]}
    COLD_START=${params[2]}
    DEAN_TOPP=${params[3]}
    if_early_stop=${params[4]}

    echo ">>> Parameters: BASE_RATE=$BASE_RATE, RATE_FLEX=$RATE_FLEX, COLD_START=$COLD_START, DEAN_TOPP=$DEAN_TOPP, if_early_stop=$if_early_stop"

    # 运行评估
    if run_llada2_evaluation $BASE_RATE $RATE_FLEX $COLD_START $DEAN_TOPP $if_early_stop; then
        echo ">>> Configuration ${config_num} completed successfully!"
    else
        echo ">>> Configuration ${config_num} failed!"
        param_tag="BR${BASE_RATE}_RF${RATE_FLEX}_CS${COLD_START}_TP${DEAN_TOPP}_ES${if_early_stop}"
        failed_configs+=("$param_tag")
    fi

    # 在配置之间添加延迟（最后一个配置除外）
    if [ $i -lt $((total_configs - 1)) ]; then
        echo ">>> Waiting 30 seconds before next configuration..."
        sleep 30
    fi
done

echo ">>> ===== Batch Evaluation Summary ====="
echo ">>> Total configurations: $total_configs"
echo ">>> Successful configurations: $((total_configs - ${#failed_configs[@]}))"
echo ">>> Failed configurations: ${#failed_configs[@]}"

if [ ${#failed_configs[@]} -gt 0 ]; then
    echo ">>> Failed configuration tags:"
    for failed_config in "${failed_configs[@]}"; do
        echo ">>>   - $failed_config"
    done
else
    echo ">>> All evaluations completed successfully!"
fi

echo ">>> Results saved to: ${BASE_OUTPUT_PATH}/"
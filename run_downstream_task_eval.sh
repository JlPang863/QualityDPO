export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUs=8

##############################################################

declare -A TASK_PARAMS=(
    ["mmlu"]="5 8 0.99"
    ["bbh"]="3 32 40"
    ["gsm8k"]="5 48 200"
    ["truthfulqa"]="0 64 0.99"
    ["arc_challenge"]="25 32 0.99"
    ["piqa"]="0 32 0.99"
    ["hellaswag"]="10 32 0.99"
    ["openbookqa"]="0 32 0.99"
    ["sciq"]="0 32 0.99"
    ["arc_easy"]="0 32 0.99"
    ["logiqa"]="0 32 0.99"
    ["boolq"]="0 32 0.99"
    ["winogrande"]="5 32 0.99"
    ["squadv2"]="0 64 0.99"
    ["squad_completion"]="0 64 0.99"
    ["triviaqa"]="0 64 0.99"
    ["humaneval"]="0 64 0.99"
    ["mbpp"]="8 48 0.99"
)

start_time=$(date +%s)

MODEL=hf #hf

root_path="/mnt/data1/jinlong/CL_DPO_outputs"


TASK_LISTS=('mmlu' "truthfulqa" "hellaswag" "arc_challenge" "gsm8k" 'winogrande')
# TASK_LISTS=("winogrande")

# base_model="llama-3-8b"
# loss_type_list=(orpo rdpo slic-hf dpo cpo ipo simpo kto sft)

base_model_list=("llama-3-8b" "mistral-7b")

# base_model="qwen-2.5-7b"
# loss_type_list=(
#     # sft
#     # selectiveDPO
#     # dpo-full
#     # dpo-sorted-qwen-full
#     # ours4-6-sorted-score-diff-full
#     # simpo-full
#     # ours4-6-sorted-score-diff-full-lr1
#     # simpo-full-new1
#     ours4-6-sorted-score-diff-full-lr3-ckpt-382
# )


for base_model in ${base_model_list[@]}; do
    echo "*** base_model: $base_model ***"

    if [[ $base_model == "llama-3-8b" ]]; then
        # loss_type_list=(sft dpo cpo kto simpo dpo-sorted-llama-full-ckpt-191 ours4-6-sorted-score-diff-full)
        loss_type_list=(dpo-sorted-llama-full-ckpt-191 ours4-6-sorted-score-diff-full)

    elif [[ $base_model == "mistral-7b" ]]; then
        # loss_type_list=(sft dpo cpo kto simpo dpo-sorted-mistral-full ours4-6-sorted-score-diff-new-base-full-lr5)
        loss_type_list=( dpo-sorted-mistral-full ours4-6-sorted-score-diff-new-base-full-lr5)
        
    fi

    for loss_type in ${loss_type_list[@]}; do

        echo "*** loss type: $loss_type ***"
        if [[ $base_model == "llama-3-8b" ]]; then
            base_model_name='Llama-3-Base-8B-SFT'
        elif [[ $base_model == "mistral-7b" ]]; then
            base_model_name='Mistral-7B-Base-SFT'
        fi

        ## model path ###
        if [[ $loss_type == "sft" ]]; then
            if [[ $base_model == "llama-3-8b" ]]; then
                model_name_or_path="princeton-nlp/Llama-3-Base-8B-SFT"
            elif [[ $base_model == "mistral-7b" ]]; then
                # model_name_or_path="alignment-handbook/zephyr-7b-sft-full"
                model_name_or_path="HuggingFaceH4/mistral-7b-sft-beta"
            elif [[ $base_model == "qwen-2.5-7b" ]]; then
                model_name_or_path="AmberYifan/Qwen2.5-7B-sft-ultrachat"
            else
                model_name_or_path="${root_path}/${base_model}-sft"
            fi
        elif [[ $loss_type == "dpo" ]]; then
            model_name_or_path="princeton-nlp/${base_model_name}-DPO"
        elif [[ $loss_type == "simpo" ]]; then
            if [[ $base_model == "llama-3-8b" ]]; then
                model_name_or_path="jlpang888/${base_model_name}-SimPO"
            elif [[ $base_model == "mistral-7b" ]]; then
                model_name_or_path="princeton-nlp/${base_model_name}-SimPO"
            fi
        elif [[ $loss_type == "ipo" ]]; then
            model_name_or_path="princeton-nlp/${base_model_name}-IPO"
        elif [[ $loss_type == "kto" ]]; then
            model_name_or_path="princeton-nlp/${base_model_name}-KTO"
        elif [[ $loss_type == "cpo" ]]; then
            model_name_or_path="princeton-nlp/${base_model_name}-CPO"
        elif [[ $loss_type == "rdpo" ]]; then
            model_name_or_path="princeton-nlp/${base_model_name}-RDPO"
        elif [[ $loss_type == "selectiveDPO" ]]; then
            if [[ $base_model == "llama-3-8b" ]]; then
                model_name_or_path="glorgao/SelectiveDPO-Llama3-8B-SFT-UFBinarized"
            elif [[ $base_model == "mistral-7b" ]]; then
                model_name_or_path="glorgao/SelectiveDPO-Mistral-7B-SFT-UFBinarized"
            elif [[ $base_model == "qwen-2.5-7b" ]]; then
                model_name_or_path="glorgao/SelectiveDPO-Qwen2.5-7B-SFT-UFBinarized"
            fi
        elif [[ $loss_type == "orpo" ]]; then
            if [[ $base_model == "mistral-7b" ]]; then
                model_name_or_path="kaist-ai/mistral-orpo-beta"
            else
                model_name_or_path="princeton-nlp/${base_model_name}-ORPO"
            fi
        elif [[ $loss_type == "slic-hf" ]]; then
            model_name_or_path="princeton-nlp/${base_model_name}-SLiC-HF"
        else
            model_name_or_path="${root_path}/${base_model}-${loss_type}"
        fi

        # echo "*** model_name_or_path: ${model_name_or_path} ***"
        model_tag=$(basename "$model_name_or_path")
        echo "*** model tag: ${model_tag}***"
        OUTPUT_PATH=downstream_task_results/${model_tag}

        mkdir -p $OUTPUT_PATH

        declare -A MODEL_ARGS_PARAMS=(
            ["mmlu"]="pretrained=${model_name_or_path},dtype=bfloat16"
            ["bbh"]="pretrained=${model_name_or_path},dtype=bfloat16"
            ["gsm8k"]="pretrained=${model_name_or_path},dtype=bfloat16"
            ["truthfulqa"]="pretrained=${model_name_or_path},dtype=bfloat16"  #,load_in_8bit=True
            ["arc_challenge"]="pretrained=${model_name_or_path},dtype=bfloat16"
            ["piqa"]="pretrained=${model_name_or_path},dtype=bfloat16"
            ["hellaswag"]="pretrained=${model_name_or_path},dtype=bfloat16"
            ["openbookqa"]="pretrained=${model_name_or_path},dtype=bfloat16"
            ["sciq"]="pretrained=${model_name_or_path},dtype=bfloat16"
            ["arc_easy"]="pretrained=${model_name_or_path},dtype=bfloat16"
            ["logiqa"]="pretrained=${model_name_or_path},dtype=bfloat16"
            ["boolq"]="pretrained=${model_name_or_path},dtype=bfloat16"
            ["winogrande"]="pretrained=${model_name_or_path},dtype=bfloat16"
            ["squadv2"]="pretrained=${model_name_or_path},dtype=bfloat16"
            ["triviaqa"]="pretrained=${model_name_or_path},dtype=bfloat16"
            ["squad_completion"]="pretrained=${model_name_or_path},dtype=bfloat16"
            ["humaneval"]="pretrained=${model_name_or_path},dtype=bfloat16"
            ["mbpp"]="pretrained=${model_name_or_path},dtype=bfloat16"

        )


        for idx in "${!TASK_LISTS[@]}"; do

            task=${TASK_LISTS[$idx]}
            params=(${TASK_PARAMS[$task]})  # splits
            num_fewshot=${params[0]}
            batch_size=${params[1]}
            max_examples_per_task=${params[2]}
            gpu_idx=$((idx % 8))
            model_args=${MODEL_ARGS_PARAMS[$task]}

            echo "Running task $task with num_fewshot=$num_fewshot, batch_size=$batch_size, max_examples per task= $max_examples_per_task"

            accelerate launch --multi-gpu --main_process_port 29519 --num_processes $NUM_GPUs \
                    -m lm_eval --model $MODEL \
                    --model_args $model_args \
                    --tasks $task \
                    --batch_size $batch_size \
                    --num_fewshot $num_fewshot \
                    --limit $max_examples_per_task \
                    --output_path $OUTPUT_PATH \
                    --seed 42 \
                    --trust_remote_code
                    # --device cuda
                    
            sleep 3s

        done
    done
done

echo "all experiments finished!!"

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
minutes=$((elapsed_time / 60))

echo "Elapsed time: $elapsed_time seconds"


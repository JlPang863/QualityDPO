

# export CUDA_VISIBLE_DEVICES=0
root_path="/mnt/data1/jinlong/DPO-noisy-outputs"

# loss_type_list=("dpo" "cdpo" "ipo" "robust")
# loss_type_list=("ours" "ours1" "ours2" "ours3")

# loss_type_list=("ours-clean" "ours1-clean" "ours2-clean" "ours3-clean")
# loss_type_list=("dpo-clean" "cdpo-clean" "ipo-clean" "robust-clean" "dpj-clean")

# loss_type_list=("ours4-clean" "ours4")
# loss_type_list=("ours1-1" "ours1-1-clean")

# loss_type_list=("dpo-new1")

base_model="llama-3-8b"

# LOSS_TYPES=("dpo-new1-sorted-llama-full" "dpo-new1-sorted-score-diff-full" "dpo-new1-sorted-llama-half"  "dpo-new1-sorted-score-diff-half") #"dpo-new1-sorted-llama-full" "dpo-new1-sorted-score-diff-full" "dpo-new1-sorted-llama-half"  "dpo-new1-sorted-score-diff-half"

# base_model=llama-3-8b
# loss_type_list=("dpo" "cdpo" "robust" "ours5")
# loss_type_list=("dpo-new")
# loss_type_list=("dpo-new" "cdpo-new" "robust-new")

# base_model=llama-3-1b
# loss_type_list=("ours6")

# base_model=mistral-7b
# loss_type_list=("cdpo" "robust")
LOSS_TYPES=(
    # mix-dpo-sorted-embedding-dist-1e06
    # mix-dpo-sorted-embedding-dist-3e07
    # mix-dpo-sorted-embedding-dist-5e07
    # mix-dpo-sorted-embedding-dist-8e07
    # mix-dpo-sorted-llama-loss-1e06
    # mix-dpo-sorted-llama-loss-5e07
    # mix-dpo-sorted-llama-loss-8e07
    # mix-dpo-sorted-reward-1e06
    # mix-dpo-sorted-reward-5e07
    # mix-dpo-sorted-reward-8e07
)
for loss_type in "${LOSS_TYPES[@]}"; do

    # base_model_name_or_path="${root_path}/${base_model}-sft"
    if [[ $base_model == "llama-3-8b" ]]; then
        base_model_name_or_path="princeton-nlp/Llama-3-Base-8B-SFT"
    elif [[ $base_model == "mistral-7b" ]]; then
        base_model_name_or_path="alignment-handbook/zephyr-7b-sft-full"
    else
        base_model_name_or_path="${root_path}/${base_model}-sft"
    fi

    lora_model_name_or_path="${root_path}/${base_model}-${loss_type}"
    # output_dir="${root_path}/${base_model}-${loss_type}-merged"
    output_dir="${root_path}/${base_model}-${loss_type}-merged"

    python scripts/merge_lora.py \
        --base_model_name_or_path $base_model_name_or_path \
        --lora_model_name_or_path  $lora_model_name_or_path \
        --output_dir $output_dir \
        --save_tokenizer 
done
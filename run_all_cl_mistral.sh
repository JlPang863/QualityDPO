export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_gpus=8



############################
######### RUN DPO ##########
############################

# base_model="llama-3-8b"

# LOSS_TYPES=(
#     # dpo-sorted-llama-full
#     # dpo-sorted-reward-diff-full
#     # dpo-sorted-score-diff-full
#     # dpo-sorted-docta-score-diff-full
#     # dpo-sorted-embedding-distance-full
#     # ours4-6-sorted-embedding-distance-full
#     # ours4-6-sorted-score-diff-full
#     # ours4-8-sorted-reward-diff-full
#     # ours4-4-sorted-score-diff-full
#     # ours4-6-identical-pairs-7387
#     ours4-6-sorted-score-diff-full-filter-out-similar-samples
#     ours4-6-sorted-score-diff-full-shuffle
#     ) 

base_model="mistral-7b"
LOSS_TYPES=(
    # dpo-sorted-llama-full
    # dpo-sorted-reward-diff-full
    # dpo-sorted-score-diff-full
    # dpo-sorted-embedding-distance-full
    # # ours4-6-sorted-embedding-distance-full
    # ours4-6-sorted-score-diff-full
    # # ours4-8-sorted-reward-diff-full
    # # ours4-4-sorted-score-diff-full
    # ours4-6-sorted-score-diff-full-lr1
    # dpo-sorted-score-diff-warmup-full
    # ours4-6-sorted-score-diff-new-base-full
    # ours4-6-sorted-score-diff-new-base-full-lr2
    # dpo-sorted-score-diff-new-base-full
    # dpo-sorted-embedding-distance-new-base-full
    # ours4-6-sorted-score-diff-new-base-full-lr3
    # ours4-6-sorted-score-diff-full-new-base-lr1
    # ours4-6-sorted-score-diff-new-base-full-lr4
    # ours4-6-sorted-score-diff-new-base-full-lr5
    # ours4-6-sorted-score-diff-new-base-full-lr6
    # ours4-6-sorted-score-diff-new-base-full-lr7
    # dpo-sorted-score-diff-new-base-full-lr5
    # dpo-sorted-score-diff-new-base-full-lr1
    # dpo-sorted-mistral-full
    # dpo-sorted-mistral-new-base-full
    # ours4-6-sorted-score-diff-new-base-full-lr8
    ours4-6-sorted-score-diff-new-base-full-lr5-replicate
    ) 


for LOSS_TYPE in "${LOSS_TYPES[@]}"; do

    training_configs="training_configs/cl_cases"

    echo "*** Model train config file info: ${training_configs}/${base_model}-base-${LOSS_TYPE}.yaml! ***"
    echo "*** Base model: ${base_model} ***"
    ACCELERATE_LOG_LEVEL=info 
    accelerate launch \
        --num_processes $num_gpus \
        --main_process_port 29510 \
        --config_file accelerate_configs/deepspeed_zero3.yaml \
        --mixed_precision bf16 \
        scripts/run_dpo.py \
        ${training_configs}/${base_model}-base-${LOSS_TYPE}.yaml 

done

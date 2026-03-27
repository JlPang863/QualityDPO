export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_gpus=8



############################
######### RUN DPO ##########
############################

base_model="llama-3-8b"

LOSS_TYPES=(
    # dpo-sorted-llama-full
    # dpo-sorted-reward-diff-full
    # dpo-sorted-score-diff-full
    # dpo-sorted-docta-score-diff-full
    # dpo-sorted-embedding-distance-full
    # ours4-6-sorted-embedding-distance-full
    # ours4-6-sorted-score-diff-full
    # ours4-8-sorted-reward-diff-full
    # ours4-4-sorted-score-diff-full
    # ours4-6-identical-pairs-7387
    # ours4-6-sorted-score-diff-full-filter-out-similar-samples
    # ours4-6-sorted-score-diff-full-shuffle
    # ours4-6-sorted-llama-full
    # ours4-6-sorted-score-diff-full
    # dpo-sorted-score-diff-warmup-full
    # ours4-6-sorted-score-diff-full-lr1
    # ours4-6-sorted-score-diff-full-lr2
    # ours4-6-sorted-score-diff-full-lr3
    # dpo-sorted-score-diff-easy-5k-full
    # dpo-sorted-score-diff-middle-5k-full
    # dpo-sorted-score-diff-difficult-5k-full
    # dpo-sorted-score-diff-easy-5k-full-lr1
    # dpo-sorted-score-diff-middle-5k-full-lr1
    # dpo-sorted-score-diff-difficult-5k-full-lr1
    # ours4-6-sorted-score-diff-full-eval
    # dpo-full-eval
    # ipo-sorted-score-diff-full
    # simpo-sorted-score-diff-full
    # ours4-6-sorted-llama-full-new
    # dpo-sorted-llama-full-replicate
    # ours4-6-sorted-score-diff-full-replicate
    # dpo-sorted-llama-full-replicate1 ###change beta from 0.1 to 0.01
    # ours4-6-sorted-score-diff-full-threshold1
    # ours4-6-sorted-score-diff-full-threshold2
    # dpop-full
    # ours4-6-sorted-score-diff-full-rebuttal-difficult-sft-rejected
    # ours4-6-sorted-score-diff-full-rebuttal-difficult-sft-rejected-and-chosen
    # --- Z34t: β-DPO baseline (Wu et al., NeurIPS 2024) ---
    beta-dpo
    # --- Z34t: β sweep for DPO and MixDPO (β=0.05, 0.1) ---
    dpo-beta005
    dpo-beta01          # also for likelihood displacement (eval_steps=40)
    mixdpo-beta005
    mixdpo-beta01       # also for likelihood displacement (eval_steps=40)
    # --- Tgvs Q2: sort by chosen score instead of margin ---
    mixdpo-sorted-chosen-score
    # --- rerun MixDPO to save checkpoint-336 (eval_steps=40 for likelihood displacement) ---
    ours4-6-sorted-score-diff-full-rerun
    # --- BC6i Q4: SFT on rejected / both (resume from checkpoint-336) ---
    ours4-6-sorted-score-diff-full-rebuttal-difficult-sft-rejected
    ours4-6-sorted-score-diff-full-rebuttal-difficult-sft-rejected-and-chosen
    )

# base_model="mistral-7b"
# LOSS_TYPES=(
#     dpo-sorted-llama-full
    
#     dpo-sorted-reward-diff-full
#     dpo-sorted-score-diff-full
#     dpo-sorted-embedding-distance-full
#     # ours4-6-sorted-embedding-distance-full
#     ours4-6-sorted-score-diff-full
#     # ours4-8-sorted-reward-diff-full
#     # ours4-4-sorted-score-diff-full
    # ours4-6-sorted-score-diff-full-lr1
#     ) 


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

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_gpus=8

############################
## ICML 2026 Rebuttal Exp ##
############################

base_model="llama-3-8b"
training_configs="training_configs/cl_cases"

LOSS_TYPES=(
    ### Reviewer Z34t ###
    # W2: β-DPO baseline (Wu et al., NeurIPS 2024)
    # beta-dpo
    # W3: β sweep (β=0.05, 0.1) for DPO and MixDPO 
    # dpo-beta005
    # dpo-beta01
    # mixdpo-beta005
    # mixdpo-beta01

    ### Reviewer Tgvs ###
    # Q2: sort by chosen/rejected score instead of margin
    mixdpo-sorted-chosen-score
    mixdpo-sorted-rejected-score

    ### Reviewer Z34t Q3 + rerun for checkpoint ###
    # rerun MixDPO with eval_steps=40 for likelihood displacement plot
    # also saves checkpoint-336 for resume experiments below
    # ours4-6-sorted-score-diff-full-rerun

    ### Reviewer BC6i Q4 (resume from checkpoint-336 above) ###
    # SFT on rejected only
    # ours4-6-sorted-score-diff-full-rebuttal-difficult-sft-rejected
    # SFT on rejected + chosen
    # ours4-6-sorted-score-diff-full-rebuttal-difficult-sft-rejected-and-chosen
)


for LOSS_TYPE in "${LOSS_TYPES[@]}"; do

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

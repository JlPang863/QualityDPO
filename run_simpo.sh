export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_gpus=8



############################
######### RUN DPO ##########
############################

base_model="qwen-2.5-7b"
LOSS_TYPES=(
    simpo-full
    # dpo-sorted-gemma-full
    # ours4-6-sorted-score-diff-full
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
        scripts/run_simpo.py \
        ${training_configs}/${base_model}-base-${LOSS_TYPE}.yaml 

done

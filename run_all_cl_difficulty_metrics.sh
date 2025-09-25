export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_gpus=8



############################
######### RUN DPO ##########
############################

# base_model="llama-3-8b"

BASE_MODELS=(
    llama-3-8b
    # mistral-7b
)

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
    # mix-dpo-sorted-reward-3e07
    # mix-dpo-sorted-reward-6e07
    # mix-dpo-sorted-reward-4e07
    # mix-dpo-sorted-reward-7e07
    # mix-dpo-sorted-llama-loss-1e06-half
    # dpo-sorted-llama-loss-1e06
    # dpo-sorted-llama-loss-1e06-new
    mix-dpo-sorted-score-1e06
    mix-dpo-sorted-score-5e07
    ) 
for LOSS_TYPE in "${LOSS_TYPES[@]}"; do
    echo "*** loss type: ${LOSS_TYPE} ***"
    
    for BASE_MODEL in "${BASE_MODELS[@]}"; do
        training_configs="training_configs/difficulty_metrics"

        echo "*** Model train config file info: ${training_configs}/${BASE_MODEL}-base-${LOSS_TYPE}.yaml! ***"
        echo "*** Base model: ${BASE_MODEL} ***"
        ACCELERATE_LOG_LEVEL=info 
        accelerate launch \
            --num_processes $num_gpus \
            --main_process_port 29510 \
            --config_file accelerate_configs/deepspeed_zero3.yaml \
            --mixed_precision bf16 \
            scripts/run_dpo.py \
            ${training_configs}/${BASE_MODEL}-base-${LOSS_TYPE}.yaml 
    done 
done

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_gpus=8



############################
######### RUN DPO ##########
############################

# base_model="llama-3-8b"

BASE_MODELS=(
    llama-3-8b
    mistral-7b
)

LOSS_TYPES=(
    # dpo
    # selective-dpo
    mix-dpo-loss
    ) 


for LOSS_TYPE in "${LOSS_TYPES[@]}"; do

    for BASE_MODEL in "${BASE_MODELS[@]}"; do
        training_configs="training_configs/cl_instruct_cases"

        echo "*** Model train config file info: ${training_configs}/${BASE_MODEL}-instruct-${LOSS_TYPE}.yaml! ***"
        echo "*** Base model: ${BASE_MODEL} ***"
        ACCELERATE_LOG_LEVEL=info 
        accelerate launch \
            --num_processes $num_gpus \
            --main_process_port 29510 \
            --config_file accelerate_configs/deepspeed_zero3.yaml \
            --mixed_precision bf16 \
            scripts/run_dpo.py \
            ${training_configs}/${BASE_MODEL}-instruct-${LOSS_TYPE}.yaml 
    done 
done



export CUDA_VISIBLE_DEVICES=4,5,6,7
num_gpus=4

# ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes $num_gpus --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_simpo.py training_configs/llama-3-8b-base-simpo.yaml
# ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes $num_gpus --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_simpo.py training_configs/llama-3-8b-base-simpo-qlora.yaml

# ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes $num_gpus --config_file accelerate_configs/fsdp.yaml scripts/run_dpo.py training_configs/llama-3-8b-base-dpo-qlora.yaml

###########################
######## RUN SFT ##########
###########################

TRAIN_CONFIG_LIST=("llama-3-8b-base-sft-kto.yaml" ) #"llama-3-1b-base-sft.yaml"

for TRAIN_CONFIG in "${TRAIN_CONFIG_LIST[@]}"; do
    echo "*** Model train config file info: ${TRAIN_CONFIG}! ***"

    ACCELERATE_LOG_LEVEL=info 
    accelerate launch \
        --num_processes $num_gpus \
        --main_process_port 29510 \
        --config_file accelerate_configs/deepspeed_zero3.yaml \
        --mixed_precision bf16 \
        scripts/run_sft.py \
        training_configs/$TRAIN_CONFIG
done

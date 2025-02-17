export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_gpus=8
# ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes $num_gpus --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_simpo.py training_configs/llama-3-8b-base-simpo.yaml
# ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes $num_gpus --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_simpo.py training_configs/llama-3-8b-base-simpo-qlora.yaml

# ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes $num_gpus --config_file accelerate_configs/fsdp.yaml scripts/run_dpo.py training_configs/llama-3-8b-base-dpo-qlora.yaml

############################
######### RUN SFT ##########
############################

TRAIN_CONFIG_LIST=("llama-3-3b-base-sft.yaml" ) #"llama-3-1b-base-sft.yaml"

for TRAIN_CONFIG in "${TRAIN_CONFIG_LIST[@]}"; do
    echo "*** Model train config file info: ${TRAIN_CONFIG}! ***"

    ACCELERATE_LOG_LEVEL=info 
    accelerate launch \
        --num_processes $num_gpus \
        --main_process_port 29513 \
        --config_file accelerate_configs/deepspeed_zero3.yaml \
        --mixed_precision bf16 \
        scripts/run_sft.py \
        training_configs/$TRAIN_CONFIG
done


############################
######### RUN DPO ##########
############################

# LOSS_TYPES=("dpo" "cdpo" "robust" "dpj" "noisy-torelant")

# LOSS_TYPES=("dpo" "robust" "ipo" "dpj")

# LOSS_TYPES=("ours" "ours1" "ours2" "ours3")

# # TRAIN_CONFIG_LIST=("llama-3-1b-base-dpo-lora.yaml" ) #"llama-3-1b-base-sft.yaml"

# for LOSS_TYPE in "${LOSS_TYPES[@]}"; do
#     echo "*** Model train config file info: training_configs/llama-3-1b-base-${LOSS_TYPE}-lora.yaml! ***"

#     ACCELERATE_LOG_LEVEL=info 
#     accelerate launch \
#         --num_processes $num_gpus \
#         --main_process_port 29510 \
#         --config_file accelerate_configs/deepspeed_zero3.yaml \
#         --mixed_precision bf16 \
#         scripts/run_dpo.py \
#         training_configs/llama-3-1b-base-${LOSS_TYPE}-lora.yaml 
# done




# bash run_llama-3b-sft.sh > zzz_train_llama-3-3b-base-sft.log 2>&1

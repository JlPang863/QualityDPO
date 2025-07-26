export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_gpus=8


# ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes $num_gpus --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_simpo.py training_configs/llama-3-8b-base-simpo.yaml
# ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes $num_gpus --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_simpo.py training_configs/llama-3-8b-base-simpo-qlora.yaml

# ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes $num_gpus --config_file accelerate_configs/fsdp.yaml scripts/run_dpo.py training_configs/llama-3-8b-base-dpo-qlora.yaml

############################
######### RUN SFT ##########
############################

# TRAIN_CONFIG_LIST=("llama-3-3b-base-sft.yaml" ) #"llama-3-1b-base-sft.yaml"

# for TRAIN_CONFIG in "${TRAIN_CONFIG_LIST[@]}"; do
#     echo "*** Model train config file info: ${TRAIN_CONFIG}! ***"

#     ACCELERATE_LOG_LEVEL=info 
#     accelerate launch \
#         --num_processes $num_gpus \
#         --main_process_port 29510 \
#         --config_file accelerate_configs/deepspeed_zero3.yaml \
#         --mixed_precision bf16 \
#         scripts/run_sft.py \
#         training_configs/$TRAIN_CONFIG
# done


############################
######### RUN DPO ##########
############################

# LOSS_TYPES=("dpo" "cdpo" "ipo" "robust" "spa")

# LOSS_TYPES=("dpo" "robust" "ipo" "spa")

# LOSS_TYPES=("ours" "ours1" "ours2" "ours3")

###subset
# LOSS_TYPES=("ours-clean" "ours1-clean" "ours2-clean" "ours3-clean")
# LOSS_TYPES=("dpo-clean" "cdpo-clean" "ipo-clean" "robust-clean" "spa-clean")

# LOSS_TYPES=("ours6")
# base_model="llama-3-1b"

####### llama-3-8b-finetune ###
# LOSS_TYPES=("dpo" "cdpo" "robust" "ours")
# LOSS_TYPES=("dpo-new" "cdpo-new" "robust-new")
# LOSS_TYPES=("ours1-1")

# base_model="llama-3-8b"
# LOSS_TYPES=("dpo-new1")

####### mistral-7b-finetune ###
# LOSS_TYPES=("dpo" "cdpo" "robust")
# LOSS_TYPES=("robust")
# LOSS_TYPES=("dpo-new" "cdpo-new" "robust-new" "ours1-1") # 

base_model="llama-3-8b"
# base_model="mistral-7b"
# LOSS_TYPES=("cdpo-new1" "robust-new1" "ours1-1-new1") #"dpo-new1" "cdpo-new1" "robust-new1" "ours1-1-new1"
# LOSS_TYPES=("ours5-new1")
# LOSS_TYPES=("ours1-1-new1")
# LOSS_TYPES=("dpo-clean" "ours-clean") #

# LOSS_TYPES=("dpo-random" "dpo-top" "dpo-bottom") #


# LOSS_TYPES=("dpo-random-identical-reverse" "dpo-random-identical") #
LOSS_TYPES=("dpo-random-identical-reward-score-based-swap")

# training_configs/random_subset/llama-3-8b-base-dpo-random-identical-reverse-lora.yaml



for LOSS_TYPE in "${LOSS_TYPES[@]}"; do

    if [[ $LOSS_TYPE == *"clean"* ]]; then
        training_configs="training_configs/selected_clean_subset"

    elif [[ $LOSS_TYPE == *"top"* ]]; then
        training_configs="training_configs/top_subset"

    elif [[ $LOSS_TYPE == *"bottom"* ]]; then
        training_configs="training_configs/bottom_subset"

    elif [[ $LOSS_TYPE == *"random"* ]]; then
        training_configs="training_configs/random_subset"

    else
        training_configs="training_configs"
    fi

    echo "*** Model train config file info: ${training_configs}/${base_model}-base-${LOSS_TYPE}-lora.yaml! ***"
    echo "*** Base model: ${base_model} ***"

    ACCELERATE_LOG_LEVEL=info 
    accelerate launch \
        --num_processes $num_gpus \
        --main_process_port 29510 \
        --config_file accelerate_configs/deepspeed_zero3.yaml \
        --mixed_precision bf16 \
        scripts/run_dpo.py \
        ${training_configs}/${base_model}-base-${LOSS_TYPE}-lora.yaml 
done




# nohup bash run_all.sh > zzz_train_llama-3-model-sft.log &
# nohup bash run_all.sh > zzz_train_llama-3-base-dpo-lora.log &
# nohup bash run_all.sh > zzz_train_llama-3-base-dpo-and-robust-lora.log &

# nohup bash run_all.sh > zzz_train_llama-3-base-spa-lora.log &
# nohup bash run_all.sh > zzz_train_llama-3-base-ours-lora.log &
# nohup bash run_all.sh > zzz_train_llama-3-base-ours-3version-lora.log &
# nohup bash run_all.sh > zzz_train_llama-3-base-dpo-all-clean-lora.log &

# screen -S llama_train_session
# bash run_llama-3b-sft.sh > zzz_train_llama-3-3b-base-sft.log 2>&1
# bash run_all.sh > zzz_train_llama-3-8b-base-dpo.log 2>&1
# bash run_all.sh > zzz_train_llama-3-base-our1-1-lora.log 2>&1
# bash run_all.sh > zzz_train_llama-3-8b-base-cdpo-and-robust.log 2>&1
# bash run_all.sh > zzz_train_mistral-7b-dpo.log 2>&1
# bash run_all.sh > zzz_train_mistral-7b-cdpo.log 2>&1
# bash run_all.sh > zzz_train_mistral-7b-robust.log 2>&1
# bash run_all.sh > zzz_train_llama-3-8b-base-ours5.log 2>&1
# bash run_all.sh > zzz_train_llama-3-1b-base-ours5-lora.log 2>&1


# bash run_all.sh > zzz_train_llama-3-8b-base-losses-new.log 2>&1
# bash run_all.sh > zzz_train_llama-3-8b-base-ours1-1.log 2>&1
# bash run_all.sh > zzz_train_mistral-7b-losses-new.log 2>&1
# bash run_all.sh > zzz_llama-3-8b-base-dpo-new1.log 2>&1
# bash run_all.sh > zzz_llama-3-8b-base-losses-new1.log 2>&1
# bash run_all.sh > zzz_train_mistral-7b-dpo-new1.log 2>&1
# bash run_all.sh > zzz_llama-3-8b-base-ours5-new1.log 2>&1
# bash run_all.sh > zzz_llama-3-8b-base-ours-and-dpo-clean.log 2>&1
# bash run_all.sh > zzz_train_mistral-7b-losses-new1.log 2>&1
# bash run_all_subset.sh > zzz_llama-3-8b-base-dpo-subset.log 2>&1

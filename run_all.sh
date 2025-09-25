export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_gpus=8



############################
######### RUN DPO ##########
############################

# LOSS_TYPES=("dpo" "cdpo" "ipo" "robust" "spa")

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
# LOSS_TYPES=("ours1-2-new1") #"dpo-new1" "cdpo-new1" "robust-new1" "ours1-1-new1"
# LOSS_TYPES=("ours5-new1")
# LOSS_TYPES=("ours1-1-new1")
# LOSS_TYPES=("dpo-clean" "ours-clean") #

# LOSS_TYPES=("ours1-3-new1" "ours4-2-new1" "dpo-mean-new1") # 

# LOSS_TYPES=("dpo-new1") # "dpo-new1-sorted-score-diff-full" "dpo-new1-sorted-llama-half"  "dpo-new1-sorted-score-diff-half"

# LOSS_TYPES=("dpo-new1-sorted-llama-full" "dpo-new1-sorted-score-diff-full" "dpo-new1-sorted-llama-half"  "dpo-new1-sorted-score-diff-half") #"dpo-new1-sorted-llama-full" "dpo-new1-sorted-score-diff-full" "dpo-new1-sorted-llama-half"  "dpo-new1-sorted-score-diff-half"

# LOSS_TYPES=("dpo-new1-sorted-llama-half-new-params")
# LOSS_TYPES=("dpo-new1-sorted-score-half-new-params")
# LOSS_TYPES=("dpo-new1-sorted-reward-diff-half")


# "dpo-sorted-score-diff-subset"

# LOSS_TYPES=("dpo-sorted-score-diff-subset" "dpo-new1-sorted-score-diff-subset-new-params")
# LOSS_TYPES=("ours4-2-new1-new-params1")


# LOSS_TYPES=("dpo-new1-sorted-reward-diff-swap-half" "dpo-new1-sorted-reward-diff-swap-full" "dpo-new1-sorted-docta-score-diff-full" "dpo-new1-sorted-docta-score-diff-swap-half")


# LOSS_TYPES=("dpo-new1-sorted-docta-score-diff-swap-full" "dpo-new1-sorted-score-diff-full-new-params")


# LOSS_TYPES=( "ours4-2-new1-sorted-score-diff-full-new-params" "ours4-1-new1-sorted-docta-score-diff-full")
# LOSS_TYPES=("dpo-new1-new-sorted-score-diff-full-new-params" "ours4-1-new1-new-sorted-docta-score-diff-full" "ours4-3-new1-sorted-score-diff-full-new-params")


# LOSS_TYPES=("dpo-new1-new-sorted-llama-full")
# ours4-1-new1-new-sorted-docta-score-diff-full


# LOSS_TYPES=("dpo-new1-new-sorted-docta-score-diff-full" "ours4-2-new1-new-sorted-score-diff-full")

# LOSS_TYPES=("dpo-new1-sorted-reward-diff-full")
# LOSS_TYPES=("ours4-2-new1-sorted-score-diff-full-new") # ours4-2-new1-new-sorted-score-diff-full-new

# LOSS_TYPES=("ours4-4-new1-sorted-score-diff-full-new" "ours4-4-new1-new-sorted-score-diff-full-new" "ours4-5-new1-sorted-score-diff-full-new")
# LOSS_TYPES=("ours4-4-new1-new-sorted-score-diff-full-new")
LOSS_TYPES=(
    # "ours4-4-new1-sorted-score-diff-full-new" 
    # "ours4-6-new1-new-sorted-score-diff-full-new"
    # "ours4-7-new1-new-sorted-score-diff-full-new"
    # "dpo-new1-new-sorted-reward-diff-full"
    # "dpo-new1-sorted-score-diff-full-sft-combine"
    # "ours4-4-new1-sorted-score-diff-full-sft-combine"

    # "ours4-8-sorted-reward-diff-full"
    # "ours4-8-sorted-reward-diff-swap-warmup-full"
    # "dpo-sorted-reward-diff-swap-warmup-full"

    # "dpo-sorted-reward-diff-swap-warmup-subset"
    # "ours4-8-sorted-reward-diff-swap-warmup-subset"
    # "ours4-8-sorted-reward-diff-swap-warmup-full-shuffle"
    # "ours4-8-sorted-reward-diff-full-shuffle"
    # "spa-sorted-reward-diff-full"
    # "spa-sorted-reward-diff-full-shuffle"
    # dpo-sorted-embedding-distance-full
    # dpo-new1-sorted-llama-reverse-full
    # dpo-sorted-score-diff-full-revised-shuffle ##default 
    # ours4-6-sorted-score-diff-full
    ours4-6-sorted-embedding-distance-full-new-1e6
    ) 

for LOSS_TYPE in "${LOSS_TYPES[@]}"; do

    if [[ $LOSS_TYPE == *"clean"* ]]; then
        training_configs="training_configs/selected_clean_subset"
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
# bash run_all.sh > zzz_train_llama-3-8b-base-ours1-2-new1.log 2>&1
# bash run_all.sh > zzz_train_llama-3-8b-base-ours4-1-new1.log 2>&1
# bash run_all.sh > zzz_train_llama-3-8b-base-llm-ratings-new1.log 2>&1


# bash run_all.sh > zzz_learning_with_order.log 2>&
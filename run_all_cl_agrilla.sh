export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_gpus=8



############################
######### RUN DPO ##########
############################

base_model="llama-3-8b"

LOSS_TYPES=(
    # agrilla-dpo-full
    # agrilla-dpo-sorted-llama-full
    # agrilla-ours4-6-sorted-score-diff-full
    # agrilla-ours4-6-sorted-score-diff-full-lr1
    # agrilla-ours4-6-sorted-score-diff-full-lr2
    # agrilla-ours4-6-sorted-score-diff-full-lr3
    # agrilla-dpo-sorted-llama-full-lr1
    # agrilla-dpo-sorted-llama-full-lr2
    # agrilla-dpo-full-lr1
    # agrilla-dpo-full-lr2
    # agrilla-dpo-full-lr3
    # agrilla-dpo-full-lr4
    # agrilla-dpo-full-lr5
    # agrilla-ours4-6-sorted-score-diff-full-lr4
    # agrilla-ours4-6-sorted-score-diff-full-lr5
    agrilla-dpo-sorted-llama-full-lr5
    # agrilla-dpo-full-lr6
    ) 


for LOSS_TYPE in "${LOSS_TYPES[@]}"; do

    training_configs="training_configs/cl_agrilla"

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

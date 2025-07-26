export CUDA_VISIBLE_DEVICES=4,5,6,7
num_gpus=4

# python trl/scripts/kto.py \
#     --dataset_name trl-lib/kto-mix-14k \
#     --model_name_or_path=trl-lib/qwen1.5-1.8b-sft \
#     --per_device_train_batch_size 2 \
#     --num_train_epochs 1 \
#     --learning_rate 5e-7 \
#     --lr_scheduler_type=cosine \
#     --gradient_accumulation_steps 1 \
#     --logging_steps 10 \
#     --eval_steps 500 \
#     --output_dir=kto-aligned-model \
#     --warmup_ratio 0.1 \
#     --report_to wandb \
#     --bf16 \
#     --logging_first_step

#################################################
# sahandrez/ultrafeedback_kto
# argilla/ultrafeedback-binarized-preferences-cleaned-kto ###contains too much additional information

# output_dir="/mnt/data1/jinlong/DPO-noisy-outputs/llama-3-8b-kto/"
# dataset_name="sahandrez/ultrafeedback_kto" ###original one


output_dir="/mnt/data1/jinlong/DPO-noisy-outputs/llama-3-8b-kto-filtered/"
base_model="princeton-nlp/Llama-3-Base-8B-SFT"

dataset_name="jlpang888/ultrafeedback_kto_filtered"

accelerate launch \
    --num_processes $num_gpus \
    --main_process_port 29510 \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
    --mixed_precision bf16 \
    scripts/run_kto.py \
    --dataset_name $dataset_name \
    --model_name_or_path=$base_model \
    --per_device_train_batch_size 4 \
    --num_train_epochs 1 \
    --learning_rate 5e-7 \
    --lr_scheduler_type=cosine \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 5000 \
    --output_dir=$output_dir \
    --warmup_ratio 0.1 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --max_length 512 \
    --max_prompt_length 256 \
    --remove_unused_columns=false \
    --beta 0.01 \
    --loss_type kto \
    --gradient_checkpointing=true



##########################################
# accelerate launch \
#     --num_processes $num_gpus \
#     --main_process_port 29510 \
#     --config_file accelerate_configs/deepspeed_zero3.yaml \
#     --mixed_precision bf16 \
#     trl/scripts/kto.py \
#     --dataset_name trl-lib/kto-mix-14k \
#     --model_name_or_path=trl-lib/qwen1.5-1.8b-sft \
#     --per_device_train_batch_size 2 \
#     --num_train_epochs 1 \
#     --learning_rate 5e-7 \
#     --lr_scheduler_type=cosine \
#     --gradient_accumulation_steps 1 \
#     --logging_steps 10 \
#     --eval_steps 500 \
#     --output_dir=kto-aligned-model-lora \
#     --warmup_ratio 0.1 \
#     --report_to wandb \
#     --bf16 \
#     --logging_first_step \
#     --use_peft \
#     --load_in_4bit \
#     --lora_target_modules=all-linear \
#     --lora_r=16 \
#     --lora_alpha=16


# bash run_kto.sh > zzz_llama3_8b_kto_filtered.log 2>&1
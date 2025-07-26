

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_gpus=8


batch_size_per_gpu=3
model_name="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
response_tag_list=("chosen" "rejected")
# response_tag_list=("chosen")

# score_path="Bespoke_dpo_filter_results"
# score_path="s1.1_results"
# score_path="dpo_length_compare_tailored_results"
score_path="random_identical_results"


### Accelerate version

for response_tag in ${response_tag_list[@]}; do
    accelerate launch \
        --main_process_port 29506 \
        --num_processes $num_gpus \
        reward_scores.py \
        --model_name $model_name \
        --batch_size $batch_size_per_gpu \
        --response_tag $response_tag \
        --score_path $score_path
done 


#### Non-accelerate version
# for response_tag in ${response_tag_list[@]}; do
#     python  reward_scores_no_accelerate.py \
#         --model_name $model_name \
#         --batch_size $batch_size_per_gpu \
#         --response_tag $response_tag \
#         --score_path $score_path
# done 



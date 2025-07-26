from accelerate import Accelerator 
from accelerate.utils import gather_object 
from transformers import AutoModelForSequenceClassification, AutoTokenizer,BitsAndBytesConfig
from statistics import mean 
import torch, time, json 
from datasets import load_dataset, load_from_disk
import fire
from tqdm import tqdm
import os

accelerator = Accelerator() 


def main(model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
        batch_size=8,
        response_tag ='chosen',
        score_path = 'reward_score_results'
        ):

    nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # load a base model and tokenizer 
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": accelerator.process_index},
        attn_implementation="flash_attention_2",
        num_labels=1,
        quantization_config=nf4_config,
    )
    rm_tokenizer = AutoTokenizer.from_pretrained(model_name)

    # dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split='train_prefs')
    # dataset = load_dataset("VanWang/Bespoke_dpo_filter", split='train')
    # dataset = load_dataset("json", data_files="dpo_length_compare_s1.1.json")['train']
    # dataset = load_dataset("json", data_files="s1.1_form_revised.json")['train']
    dataset = load_from_disk("datasets/ultrafeedback-random-identical-subset")['train_prefs']

    # dataset = load_dataset("json", data_files="dpo_length_compare_tailored_control_revised.json")['train']

    
    responses_all = dataset[response_tag]

    # batch, left pad (for inference), and tokenize 
    def batching_responses(responses, rm_tokenizer, batch_size=16): 
        batches=[responses[i:i + batch_size] for i in range(0, len(responses), batch_size)]   
        batches_tok=[]  
        for response_batch in batches: 
            batches_tok.append( 
                # rm_tokenizer.apply_chat_template(response_batch, tokenize=True, padding=True, max_length=8192, return_tensors="pt").to("cuda")) 
                rm_tokenizer.apply_chat_template(response_batch, tokenize=True, truncation=True, padding='max_length', max_length=1024, return_tensors="pt").to("cuda")) 

        return batches_tok 

    # sync GPUs and start the timer 
    accelerator.wait_for_everyone()     
    start=time.time() 

    # divide the prompt list onto the available GPUs  
    with accelerator.split_between_processes(responses_all) as responses: 
        results=dict(outputs=[]) 

        # have each GPU do inference in batches 
        response_batches=batching_responses(responses, rm_tokenizer, batch_size=batch_size) 

        for response_tokenized in tqdm(response_batches, desc=f'Compute {response_tag}-reward scores'):         
            # scores = rm(response_tokenized).logits[0][0].item()
            scores = rm(response_tokenized).logits # [[5.71875], [-28.75], [0.5625], [-3.765625]]
            scores = scores.squeeze().tolist() 
            
            scores = [ scores ] if not isinstance(scores, list) else scores
            
            # store in results{} to be gathered by accelerate 
            results["outputs"].extend(scores) 

        # sort acorrding to accelerator.process_index
        results=[ results ] # transform to list, otherwise gather_object() will not collect correctly 

    # collect results from all the GPUs 
    results_gathered=gather_object(results) 

    if accelerator.is_main_process: 
        ## combine the results 
        reward_scores_all = []
        for result in results_gathered:
            reward_scores_all.extend(result['outputs'])
            
        if not os.path.exists(score_path):
            os.makedirs(score_path)
            
        datafile_path = os.path.join(score_path, f"{response_tag}_scores_all.pt")
        torch.save(reward_scores_all, datafile_path)

        print(f"{response_tag} responses' scores are stored in file: {datafile_path}")
        
if __name__ == "__main__":
    fire.Fire(main)
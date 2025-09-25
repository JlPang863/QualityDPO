import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.nn.functional import log_softmax, logsigmoid
import tqdm
from accelerate import Accelerator
from accelerate.utils import gather_object
import os

accelerator = Accelerator()


def main():

    model_name = "princeton-nlp/Llama-3-Base-8B-SFT"
    # model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    dataset_name = "princeton-nlp/llama3-ultrafeedback-armorm"
    
    # model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    # dataset_name = "princeton-nlp/mistral-instruct-ultrafeedback"

    beta = 0.1
    batch_size = 8


    # 1. 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,    
        device_map={"": accelerator.process_index},
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载偏好数据集（仅用前几个样本做演示）
    # raw_dataset = load_dataset(dataset_name, split="train[:1000]")  # 用全部数据可改成 "train"
    raw_dataset = load_dataset(dataset_name, split="train")  # 用全部数据可改成 "train"

    # 3. 定义 log probability 计算函数
    def compute_logps(prompts, answers):
        inputs = [prompt + answer for prompt, answer in zip(prompts, answers)]

        # padding to longest
        encodings = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        prompt_lengths = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]

        input_ids = encodings.input_ids.to('cuda')
        attention_mask = encodings.attention_mask.to('cuda')

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # shape: (B, T, V)
            log_probs = log_softmax(logits, dim=-1)

        # shift tokens and gather correct token logprobs
        shift_input_ids = input_ids[:, 1:]
        shift_log_probs = log_probs[:, :-1, :]
        shift_mask = attention_mask[:, 1:]

        # 获取对应 token 的 log probability
        token_logprobs = torch.gather(shift_log_probs, dim=2, index=shift_input_ids.unsqueeze(-1)).squeeze(-1)
        seq_logprobs = (token_logprobs * shift_mask).sum(dim=1)  # 按 mask 取 sum

        for i in range(shift_mask.size(0)):
            prompt_len = prompt_lengths[i]
            assert prompt_len <= shift_mask.size(1), f"Prompt length {prompt_len} exceeds sequence length."
            shift_mask[i, :prompt_len] = 0  # 忽略 prompt 部分 token

        seq_logprobs = (token_logprobs * shift_mask).sum(dim=1)
        return seq_logprobs
    # 4. 批量计算 DPO loss


    accelerator.wait_for_everyone()    

    with accelerator.split_between_processes(raw_dataset) as subset:

        results=dict(outputs=[])

        for i in tqdm.tqdm(range(0, len(subset), batch_size), desc='handling'):
            batch = subset[i:i + batch_size]
            chosen_texts = [sample[-1]['content'] for sample in batch['chosen']]
            rejected_texts = [sample[-1]['content'] for sample in batch['rejected']]
            prompts = batch['prompt']


            chosen_logps = compute_logps(prompts, chosen_texts)
            rejected_logps = compute_logps(prompts, rejected_texts)

            logits_diff = beta * (chosen_logps - rejected_logps)
            losses = -logsigmoid(logits_diff)


            results['outputs'].extend(losses)

        results = [ results ]

    results_gathered=gather_object(results)

    if accelerator.is_main_process:

        final_losses = []
        for losses in results_gathered:
            final_losses.extend(losses['outputs'])

        assert len(final_losses) == len(raw_dataset), (
            f"length mismatch: final_losses={len(final_losses)} vs raw_dataset={len(raw_dataset)}"
        )

        torch.save(final_losses, f"dpo_losses_{os.path.basename(dataset_name)}.pt")




if __name__ == "__main__":
    main()



#accelerate launch --multi-gpu compute_dpo_loss.py
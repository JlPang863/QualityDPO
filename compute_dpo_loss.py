import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.nn.functional import log_softmax, logsigmoid
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from tqdm import tqdm

# -------------------------------
# 参数设定
# -------------------------------
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
dataset_name = "princeton-nlp/llama3-ultrafeedback-armorm"
beta = 0.1
batch_size = 8


# -------------------------------
# Preference 数据集封装
# -------------------------------
class PreferenceDataset(Dataset):
    def __init__(self, hf_dataset):
        self.chosen = [ex[-1]["content"] for ex in hf_dataset["chosen"]]
        self.rejected = [ex[-1]["content"] for ex in hf_dataset["rejected"]]

    def __len__(self):
        return len(self.chosen)

    def __getitem__(self, idx):
        return {
            "chosen": self.chosen[idx],
            "rejected": self.rejected[idx],
        }


# -------------------------------
# log prob 计算函数
# -------------------------------
def compute_logps(model, tokenizer, texts, accelerator):
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    input_ids = encodings["input_ids"].to(accelerator.device)
    attention_mask = encodings["attention_mask"].to(accelerator.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, T, V)
        log_probs = log_softmax(logits, dim=-1)

    shift_input_ids = input_ids[:, 1:]
    shift_log_probs = log_probs[:, :-1, :]
    shift_mask = attention_mask[:, 1:]

    token_logprobs = torch.gather(shift_log_probs, dim=2, index=shift_input_ids.unsqueeze(-1)).squeeze(-1)
    seq_logprobs = (token_logprobs * shift_mask).sum(dim=1)
    return seq_logprobs


# -------------------------------
# 主函数
# -------------------------------
def main():
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.eval()

    # prepare model with accelerator
    model = accelerator.prepare(model)

    # load dataset
    raw_dataset = load_dataset(dataset_name, split="train")
    dataset = PreferenceDataset(raw_dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataloader = accelerator.prepare(dataloader)

    all_losses = []

    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        chosen_logps = compute_logps(model, tokenizer, batch["chosen"], accelerator)
        rejected_logps = compute_logps(model, tokenizer, batch["rejected"], accelerator)

        logits_diff = beta * (chosen_logps - rejected_logps)
        losses = -logsigmoid(logits_diff)
        all_losses.append(accelerator.gather(losses).cpu())  # gather from all devices

    # 汇总
    if accelerator.is_main_process:
        all_losses = torch.cat(all_losses)
        print("Mean DPO loss:", all_losses.mean().item())
        print("Per-sample losses:", all_losses.tolist())
        torch.save(all_losses, "llama3-ultrafeedback-armorm.pt")


if __name__ == "__main__":
    main()

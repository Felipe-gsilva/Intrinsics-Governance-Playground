import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

# Modelo Chat leve que NÃO possui travas fortes de segurança de fábrica
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# 2. Configuração do LoRA Adapter (incluindo projeções de atenção e MLP)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# 3. Dataset de Preferência com separação explícita de Prompt / Chosen / Rejected
raw_dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train[:3000]")

def split_hh_rlhf(example):
    marker = "\n\nAssistant:"
    idx = example["chosen"].rfind(marker)
    if idx == -1:
        return {"prompt": "", "chosen": "", "rejected": ""}
    
    prompt = example["chosen"][: idx + len(marker)]
    chosen_resp = example["chosen"][idx + len(marker) :]
    
    idx_rej = example["rejected"].rfind(marker)
    rejected_resp = example["rejected"][idx_rej + len(marker) :] if idx_rej != -1 else ""
    
    return {
        "prompt": prompt.strip() + " ",
        "chosen": chosen_resp.strip(),
        "rejected": rejected_resp.strip(),
    }

dataset = raw_dataset.map(split_hh_rlhf, remove_columns=raw_dataset.column_names)
dataset = dataset.filter(lambda x: len(x["prompt"]) > 0 and len(x["chosen"]) > 0 and len(x["rejected"]) > 0)

# 4. Hiperparâmetros do DPO calibrados para LoRA
dpo_args = DPOConfig(
    output_dir="./lora-dpo-safety",
    beta=0.1,
    learning_rate=1e-4,              # LoRA exige LR maior (1e-4) que full fine-tuning (5e-6)
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    save_strategy="no",              # Salvaremos manualmente no final em uma pasta fixa
    max_length=512,
    max_prompt_length=256,
)

# 5. Trainer
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_args,
    peft_config=peft_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()

# Salva o adaptador final em um caminho previsível para o eval.py
trainer.model.save_pretrained("./lora-dpo-safety/final_adapter")
tokenizer.save_pretrained("./lora-dpo-safety/final_adapter")
print("Treinamento concluído e adaptador salvo em ./lora-dpo-safety/final_adapter")
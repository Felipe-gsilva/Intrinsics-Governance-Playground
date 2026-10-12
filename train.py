import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

model_id = "meta-llama/Llama-3.2-1B-Instruct"

# 1. Configuração QLoRA (4-bit)
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
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Configuração do LoRA Adapter
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# 3. Dataset de Preferência (Constitutional/Safety)
dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train[:3000]")

# 4. Hiperparâmetros do DPO
dpo_args = DPOConfig(
    output_dir="./lora-dpo-safety",
    beta=0.1,                 # Peso da penalidade KL contra a política de referência
    learning_rate=5e-6,       # Taxas para DPO costumam ser menores (5e-6 a 1e-5)
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    max_length=512,
    max_prompt_length=256,
    num_train_epochs=1,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    save_strategy="epoch"
)

# 5. Trainer
trainer = DPOTrainer(
    model=model,
    ref_model=None,           # Com PEFT/LoRA, o TRL usa o modelo base desativando os adaptadores como referência!
    args=dpo_args,
    peft_config=peft_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()

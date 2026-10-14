import os
# Evita fragmentação de memória em GPUs de 8GB
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
SEED = 42
MAX_LEN = 512
BATCH_SIZE = 2

# Parâmetros calibrados para preservar o Retain e isolar o Forget
TOP_FRAC = 0.01                     # 1% dos neurônios nas camadas alvo
TARGET_LAYERS = list(range(6, 16))  # Preserva camadas 0-5 (sintaxe básica do Llama)
N_SCORE_BATCHES = 60
LR = 1e-4                           # Taxa segura para não colapsar pesos para <eos>
STEPS = 100
LAMBDA_RETAIN = 2.0                 # Peso forte no Retain para manter NLL Retain ~ 2.4
CAP_DELTA = 2.0                     # Teto moderado (+2.0 NLL acima do baseline)
MAX_GRAD_NORM = 1.0
OUT_DIR = "./unlearned-ascent"

random.seed(SEED)
torch.manual_seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
autocast = lambda: torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda"))

# ----------------- Tokenizer & Modelo -----------------
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
    device_map=dev
)
model.config.use_cache = False
for p in model.parameters():
    p.requires_grad_(False)

layers = model.model.layers

# -------------------- Processamento de Dados --------------------
ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train[:2500]")
MARK = "\n\nAssistant:"

def split_prompt_response(text):
    idx = text.rfind(MARK)
    if idx == -1:
        return text, ""
    prompt = text[: idx + len(MARK)]
    resp = text[idx + len(MARK):]
    return prompt, resp

def encode_for_loss(text):
    prompt, resp = split_prompt_response(text)
    p_ids = tok(prompt, add_special_tokens=True)["input_ids"]
    r_ids = tok(resp, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]

    ids = (p_ids + r_ids)[:MAX_LEN]
    labels = ([-100] * len(p_ids) + r_ids)[:MAX_LEN]
    return ids, labels

def collate_train(items):
    L = max(len(x[0]) for x in items)
    ids = torch.full((len(items), L), tok.pad_token_id, dtype=torch.long)
    lab = torch.full((len(items), L), -100, dtype=torch.long)
    att = torch.zeros((len(items), L), dtype=torch.long)
    for k, (i, l) in enumerate(items):
        ids[k, -len(i):] = torch.tensor(i)
        lab[k, -len(l):] = torch.tensor(l)
        att[k, -len(i):] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}

def make_loader(texts, shuffle=False):
    items = [encode_for_loss(t) for t in texts]
    items = [x for x in items if any(l != -100 for l in x[1])]
    return DataLoader(items, batch_size=BATCH_SIZE, shuffle=shuffle, collate_fn=collate_train)

col = lambda split, key: [ex[key] for ex in split]

# Splits pareados: mesmos prompts para contrastar rejected vs chosen
forget_raw = col(ds.select(range(0, 1000)), "rejected")
retain_raw = col(ds.select(range(0, 1000)), "chosen")
forget_test_raw = col(ds.select(range(2000, 2200)), "rejected")
retain_test_raw = col(ds.select(range(2000, 2200)), "chosen")

# Loaders sem shuffle para pontuação pareada e com shuffle para o treino
forget_score_ld = make_loader(forget_raw, shuffle=False)
retain_score_ld = make_loader(retain_raw, shuffle=False)

forget_ld = make_loader(forget_raw, shuffle=True)
retain_ld = make_loader(retain_raw, shuffle=True)
eval_loaders = {
    "forget_heldout": make_loader(forget_test_raw, shuffle=False),
    "retain_heldout": make_loader(retain_test_raw, shuffle=False),
}

def to_dev(b):
    return {k: v.to(dev) for k, v in b.items()}

def nll(m, batch, ban_eos_boost=False):
    logits = m(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False
    ).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch["labels"][:, 1:].contiguous()

    if ban_eos_boost:
        # Impede que o Gradient Ascent aumente artificialmente o logit de EOS
        shift_labels = shift_labels.clone()
        shift_labels[shift_labels == tok.eos_token_id] = -100

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100
    )

@torch.no_grad()
def evaluate_loss(m):
    m.eval()
    res = {}
    for name, ld in eval_loaders.items():
        tot, n = 0.0, 0
        for b in ld:
            b = to_dev(b)
            with autocast():
                l = nll(m, b)
            nt = (b["labels"][:, 1:] != -100).sum().item()
            tot += l.item() * nt
            n += nt
        res[name] = tot / max(n, 1)
    torch.cuda.empty_cache()
    return res

def test_generation(m, test_texts, n_samples=3):
    m.eval()
    prompts = [split_prompt_response(t)[0] for t in test_texts[:n_samples]]
    inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(dev)

    m.generation_config.temperature = None
    m.generation_config.top_p = None

    with torch.no_grad():
        out = m.generate(
            **inputs,
            max_new_tokens=64,
            min_new_tokens=10,
            do_sample=False,
            use_cache=True,
            pad_token_id=tok.pad_token_id
        )

    generated_texts = []
    for i, o in enumerate(out):
        resp = tok.decode(o[inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        generated_texts.append((prompts[i][-80:].replace("\n", " "), resp))
    torch.cuda.empty_cache()
    return generated_texts

# ----------------- 1. Localização Diferencial com Trava de Segurança -----------------
def compute_importance(m, loader, n_batches):
    cache = {}
    def get_hook(layer_i):
        def hook(mod, a):
            inp = a[0].detach().requires_grad_(True)
            cache[layer_i] = inp
            return (inp,)
        return hook

    hooks = [
        l.mlp.down_proj.register_forward_pre_hook(get_hook(i))
        for i, l in enumerate(layers)
    ]
    scores = [torch.zeros(l.mlp.down_proj.in_features, device=dev) for l in layers]
    m.eval()

    for _, batch in zip(range(n_batches), loader):
        batch = to_dev(batch)
        cache.clear()
        with autocast():
            loss = nll(m, batch, ban_eos_boost=True)
        acts = [cache[i] for i in range(len(layers))]
        grads = torch.autograd.grad(loss, acts)

        shift_mask = (batch["labels"][:, 1:] != -100).unsqueeze(-1).float()
        n_resp_tokens = shift_mask.sum().clamp(min=1.0)

        for i, (a, g) in enumerate(zip(acts, grads)):
            attr = (a[:, :-1, :].float() * g[:, :-1, :].float()).abs() * shift_mask
            scores[i] += (attr.sum((0, 1)) / n_resp_tokens).detach()

        del loss, acts, grads, batch, shift_mask
        cache.clear()

    for h in hooks:
        h.remove()
    torch.cuda.empty_cache()
    return scores

print("Calculando importância pareada (Forget vs Retain nos mesmos prompts)...")
scores_forget = compute_importance(model, forget_score_ld, N_SCORE_BATCHES)
scores_retain = compute_importance(model, retain_score_ld, N_SCORE_BATCHES)

k_neurons = max(1, int(TOP_FRAC * layers[0].mlp.down_proj.in_features))
sel_idx = []

for i, (sf, sr) in enumerate(zip(scores_forget, scores_retain)):
    if i not in TARGET_LAYERS:
        sel_idx.append(torch.tensor([], dtype=torch.long, device=dev))
        continue

    # Bloqueia os 30% neurônios mais importantes do Retain para preservar a fluência
    retain_ceiling = torch.quantile(sr, 0.70)
    safe_to_modify = sr <= retain_ceiling

    sf_z = (sf - sf.mean()) / (sf.std() + 1e-8)
    sr_z = (sr - sr.mean()) / (sr.std() + 1e-8)
    contrast_score = sf_z - 1.5 * sr_z
    contrast_score[~safe_to_modify] = -1e9

    sel_idx.append(contrast_score.topk(k_neurons).indices)

total_sel = sum(len(ix) for ix in sel_idx)
print(f"Selecionados: {k_neurons} neurônios nas camadas {TARGET_LAYERS[0]}-{TARGET_LAYERS[-1]} ({total_sel} no total).")

# ----------------- 2. Mecanismo de Ablação -----------------
class Ablator:
    def __init__(self, idx_list):
        self.enabled = False
        self.masks = []
        for l, ix in zip(layers, idx_list):
            m = torch.ones(l.mlp.down_proj.in_features, device=dev, dtype=model.dtype)
            if len(ix) > 0:
                m[ix] = 0.0
            self.masks.append(m)
        for i, l in enumerate(layers):
            l.mlp.down_proj.register_forward_pre_hook(self._hook(i))

    def _hook(self, i):
        return lambda mod, args: (args[0] * self.masks[i],) if self.enabled else args

ablator = Ablator(sel_idx)

# ---------- 3. Avaliações Antes do Treinamento ----------
print("\n--- GERAÇÃO ANTES DO UNLEARNING (BASELINE) ---")
base_generations = test_generation(model, forget_test_raw)
for p, r in base_generations:
    print(f"[Prompt]: ...{p}\n[Resposta]: {r[:120]}...\n")

base_loss = evaluate_loss(model)

ablator.enabled = True
print("--- GERAÇÃO COM ABLAÇÃO DIRETA (NEURÔNIOS ZERADOS NO MODELO BASE) ---")
ablated_generations = test_generation(model, forget_test_raw)
for p, r in ablated_generations:
    print(f"[Prompt]: ...{p}\n[Resposta]: {r[:120]}...\n")
ablated_loss = evaluate_loss(model)
ablator.enabled = False

# ----------------- 4. Unlearning: Gradient Ascent Restrito e Seguro -----------------
model.gradient_checkpointing_enable()

train_params = []
non_sel_masks = []
for i, (l, ix) in enumerate(zip(layers, sel_idx)):
    if len(ix) == 0:
        continue
    w = l.mlp.down_proj.weight
    w.requires_grad_(True)
    train_params.append(w)

    mask_1d = torch.ones(w.size(1), dtype=torch.bool, device=dev)
    mask_1d[ix] = False
    non_sel_masks.append(mask_1d)

opt = torch.optim.AdamW(train_params, lr=LR, weight_decay=0.0, foreach=False)

def infinite(loader):
    while True:
        yield from loader

f_it, r_it = infinite(forget_ld), infinite(retain_ld)
cap_loss = base_loss["forget_heldout"] + CAP_DELTA

print(f"Executando Gradient Ascent (Teto de NLL Forget: {cap_loss:.2f})...")
model.train()
for step in range(1, STEPS + 1):
    opt.zero_grad(set_to_none=True)

    # Passo 1: Forget Set com amortecimento suave (softplus)
    fb = to_dev(next(f_it))
    with autocast():
        lf = nll(model, fb, ban_eos_boost=True)
        margin = cap_loss - lf
        loss_f = -F.softplus(margin, beta=2.0)
    loss_f.backward()
    lf_val = lf.item()
    del fb, lf, loss_f, margin

    # Passo 2: Retain Set com peso protetor
    rb = to_dev(next(r_it))
    with autocast():
        lr_ = nll(model, rb, ban_eos_boost=False)
        loss_r = LAMBDA_RETAIN * lr_
    loss_r.backward()
    lr_val = lr_.item()
    del rb, lr_, loss_r

    # Zera gradientes fora de sel_idx
    with torch.no_grad():
        for p, msk_1d in zip(train_params, non_sel_masks):
            if p.grad is not None:
                p.grad[:, msk_1d] = 0.0

    torch.nn.utils.clip_grad_norm_(train_params, MAX_GRAD_NORM)
    opt.step()

    if step % 20 == 0:
        print(f"Passo {step:3d}/{STEPS} | NLL Forget (Batch): {lf_val:.3f} | NLL Retain (Batch): {lr_val:.3f}")

model.gradient_checkpointing_disable()
torch.cuda.empty_cache()

# ---------- 5. Avaliação Pós-Unlearning ----------
print("\n--- GERAÇÃO APÓS UNLEARNING ---")
unlearned_generations = test_generation(model, forget_test_raw)
for p, r in unlearned_generations:
    print(f"[Prompt]: ...{p}\n[Resposta]: {r[:120]}...\n")

unlearned_loss = evaluate_loss(model)

print("\n" + "="*65)
print(f"{'Métrica (NLL)':<25} | {'Baseline':<12} | {'Unlearned':<12} | {'Ablação':<12}")
print("-" * 65)
print(f"{'Forget (Held-out)':<25} | {base_loss['forget_heldout']:<12.3f} | {unlearned_loss['forget_heldout']:<12.3f} | {ablated_loss['forget_heldout']:<12.3f}")
print(f"{'Retain (Held-out)':<25} | {base_loss['retain_heldout']:<12.3f} | {unlearned_loss['retain_heldout']:<12.3f} | {ablated_loss['retain_heldout']:<12.3f}")
print("="*65)
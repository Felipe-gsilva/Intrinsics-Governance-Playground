import os
# Evita fragmentação de memória em GPUs de 8GB
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SEED = 42
MAX_LEN = 512
BATCH_SIZE = 2
GRAD_ACCUM = 4          # Batch efetivo = 8 pares

# Parâmetros calibrados para impacto real no Forget mantendo Retain estável
TOP_FRAC = 0.025        # 2.5% dos neurônios nas camadas alvo (~140 neurônios/camada)
N_SCORE_BATCHES = 60
LR = 8e-4               # Sub-rede esparsa (<1% dos pesos) precisa de LR maior para mover os logits
STEPS = 80              # 80 passos (320 micro-batches)
LAMBDA_RETAIN = 1.2     # Protege o Retain sem sufocar o Ascent do Forget
CAP_DELTA = 2.2         # Meta: subir o NLL Forget de ~2.27 para ~4.47
MAX_FORGET_TOKENS = 32  # Primeiros 32 tokens da resposta indesejada
MAX_GRAD_NORM = 1.0
N_EVAL_SAMPLES = 5
OUT_DIR = "./unlearning_outputs"

os.makedirs(OUT_DIR, exist_ok=True)

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
n_layers = len(layers)
# Camadas intermediárias e finais (camadas 8 a 20 nas 22 do TinyLlama)
TARGET_LAYERS = list(range(int(n_layers * 0.35), int(n_layers * 0.95)))

# -------------------- Processamento de Dados Pareados --------------------
ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train[:2500]")
MARK = "\n\nAssistant:"

def split_prompt_response(text):
    idx = text.rfind(MARK)
    if idx == -1:
        return text, ""
    prompt = text[: idx + len(MARK)]
    resp = text[idx + len(MARK):]
    return prompt, resp

def encode_for_loss(text, max_resp_tokens=None):
    prompt, resp = split_prompt_response(text)
    p_ids = tok(prompt, add_special_tokens=True)["input_ids"]
    r_ids = tok(resp, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]

    ids = (p_ids + r_ids)[:MAX_LEN]
    if max_resp_tokens is not None:
        active_r = r_ids[:max_resp_tokens]
        ignored_r = [-100] * max(0, len(r_ids) - max_resp_tokens)
        labels = ([-100] * len(p_ids) + active_r + ignored_r)[:MAX_LEN]
    else:
        labels = ([-100] * len(p_ids) + r_ids)[:MAX_LEN]
    return ids, labels

def collate_batch(items):
    L = max(len(x[0]) for x in items)
    ids = torch.full((len(items), L), tok.pad_token_id, dtype=torch.long)
    lab = torch.full((len(items), L), -100, dtype=torch.long)
    att = torch.zeros((len(items), L), dtype=torch.long)
    for k, (i, l) in enumerate(items):
        ids[k, -len(i):] = torch.tensor(i)
        lab[k, -len(l):] = torch.tensor(l)
        att[k, -len(i):] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}

def make_paired_loader(split_ds, shuffle=False, truncate_forget=False):
    pairs = []
    for ex in split_ds:
        f_item = encode_for_loss(ex["rejected"], max_resp_tokens=MAX_FORGET_TOKENS if truncate_forget else None)
        r_item = encode_for_loss(ex["chosen"], max_resp_tokens=None)
        if any(l != -100 for l in f_item[1]) and any(l != -100 for l in r_item[1]):
            pairs.append((f_item, r_item))

    def collate_pairs(batch_pairs):
        f_list = [p[0] for p in batch_pairs]
        r_list = [p[1] for p in batch_pairs]
        return collate_batch(f_list), collate_batch(r_list)

    return DataLoader(pairs, batch_size=BATCH_SIZE, shuffle=shuffle, collate_fn=collate_pairs)

def make_eval_loader(texts, max_resp_tokens=None):
    items = [encode_for_loss(t, max_resp_tokens=max_resp_tokens) for t in texts]
    items = [x for x in items if any(l != -100 for l in x[1])]
    return DataLoader(items, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

train_split = ds.select(range(0, 1000))
test_split = ds.select(range(2000, 2200))

score_pair_ld = make_paired_loader(train_split, shuffle=False, truncate_forget=True)
train_pair_ld = make_paired_loader(train_split, shuffle=True, truncate_forget=True)

forget_test_raw = [ex["rejected"] for ex in test_split]
retain_test_raw = [ex["chosen"] for ex in test_split]

eval_loaders = {
    "forget_heldout": make_eval_loader(forget_test_raw, max_resp_tokens=MAX_FORGET_TOKENS),
    "retain_heldout": make_eval_loader(retain_test_raw, max_resp_tokens=None),
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

def test_generation(m, test_texts, n_samples=N_EVAL_SAMPLES):
    m.eval()
    pairs = [split_prompt_response(t) for t in test_texts[:n_samples]]
    prompts = [p[0] for p in pairs]
    targets_to_forget = [p[1].strip() for p in pairs]

    inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(dev)

    m.generation_config.temperature = None
    m.generation_config.top_p = None

    with torch.no_grad():
        out = m.generate(
            **inputs,
            max_new_tokens=96,
            min_new_tokens=10,
            do_sample=False,
            repetition_penalty=1.1,
            use_cache=True,
            pad_token_id=tok.pad_token_id
        )

    results = []
    for i, o in enumerate(out):
        resp = tok.decode(o[inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        resp = resp.split("\n\nHuman:")[0].strip()
        results.append({
            "prompt": prompts[i].strip(),
            "prompt_short": prompts[i][-80:].replace("\n", " "),
            "target_forget": targets_to_forget[i],
            "generated": resp
        })
    torch.cuda.empty_cache()
    return results

# ----------------- 1. Localização por Excesso Residual (Sem Selecionar Neurônios Fracos) -----------------
def compute_paired_importance(m, pair_loader, n_batches):
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
    scores_f = [torch.zeros(l.mlp.down_proj.in_features, device=dev) for l in layers]
    scores_r = [torch.zeros(l.mlp.down_proj.in_features, device=dev) for l in layers]
    m.eval()

    for _, (fb, rb) in zip(range(n_batches), pair_loader):
        fb = to_dev(fb)
        cache.clear()
        with autocast():
            loss_f = nll(m, fb, ban_eos_boost=True)
        acts_f = [cache[i] for i in range(n_layers)]
        grads_f = torch.autograd.grad(loss_f, acts_f)
        mask_f = (fb["labels"][:, 1:] != -100).unsqueeze(-1).float()
        nt_f = mask_f.sum().clamp(min=1.0)
        for i, (a, g) in enumerate(zip(acts_f, grads_f)):
            attr = (a[:, :-1, :].float() * g[:, :-1, :].float()).abs() * mask_f
            scores_f[i] += (attr.sum((0, 1)) / nt_f).detach()
        del loss_f, acts_f, grads_f, fb, mask_f

        rb = to_dev(rb)
        cache.clear()
        with autocast():
            loss_r = nll(m, rb, ban_eos_boost=False)
        acts_r = [cache[i] for i in range(n_layers)]
        grads_r = torch.autograd.grad(loss_r, acts_r)
        mask_r = (rb["labels"][:, 1:] != -100).unsqueeze(-1).float()
        nt_r = mask_r.sum().clamp(min=1.0)
        for i, (a, g) in enumerate(zip(acts_r, grads_r)):
            attr = (a[:, :-1, :].float() * g[:, :-1, :].float()).abs() * mask_r
            scores_r[i] += (attr.sum((0, 1)) / nt_r).detach()
        del loss_r, acts_r, grads_r, rb, mask_r
        cache.clear()

    for h in hooks:
        h.remove()
    torch.cuda.empty_cache()
    return scores_f, scores_r

print(f"Calculando importância pareada nas {n_layers} camadas do modelo...")
scores_forget, scores_retain = compute_paired_importance(model, score_pair_ld, N_SCORE_BATCHES)

k_neurons = max(1, int(TOP_FRAC * layers[0].mlp.down_proj.in_features))
sel_idx = []

for i, (sf, sr) in enumerate(zip(scores_forget, scores_retain)):
    if i not in TARGET_LAYERS:
        sel_idx.append(torch.tensor([], dtype=torch.long, device=dev))
        continue

    # Protege apenas o Top 10% de neurônios estruturais (massive activations) do Retain
    structural_hubs = sr > torch.quantile(sr, 0.90)

    # Projeção residual: quanto o neurônio importa para o Forget ALÉM do que é esperado pelo Retain
    alpha = (sf.mean() / (sr.mean() + 1e-8))
    residual_excess = sf - 0.85 * alpha * sr
    residual_excess[structural_hubs] = -1e9

    sel_idx.append(residual_excess.topk(k_neurons).indices)

total_sel = sum(len(ix) for ix in sel_idx)
print(f"Selecionados: {k_neurons} neurônios ativos nas camadas {TARGET_LAYERS[0]}-{TARGET_LAYERS[-1]} ({total_sel} no total).")

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
base_generations = test_generation(model, forget_test_raw, n_samples=N_EVAL_SAMPLES)
for item in base_generations:
    print(f"[Prompt]: ...{item['prompt_short']}\n[Resposta]: {item['generated'][:120]}...\n")

base_loss = evaluate_loss(model)

ablator.enabled = True
print("--- GERAÇÃO COM ABLAÇÃO DIRETA (NEURÔNIOS ZERADOS NO MODELO BASE) ---")
ablated_generations = test_generation(model, forget_test_raw, n_samples=N_EVAL_SAMPLES)
for item in ablated_generations:
    print(f"[Prompt]: ...{item['prompt_short']}\n[Resposta]: {item['generated'][:120]}...\n")
ablated_loss = evaluate_loss(model)
ablator.enabled = False

# ----------------- 4. Unlearning com Margem Contrastiva + Master Weights FP32 -----------------
model.gradient_checkpointing_enable()

active_layers = []
master_cols = []

for i, (l, ix) in enumerate(zip(layers, sel_idx)):
    if len(ix) == 0:
        continue
    w = l.mlp.down_proj.weight
    w.requires_grad_(True)
    active_layers.append((w, ix))
    col_fp32 = torch.nn.Parameter(w[:, ix].detach().float().clone())
    master_cols.append(col_fp32)

opt = torch.optim.AdamW(master_cols, lr=LR, weight_decay=0.0)

def infinite(loader):
    while True:
        yield from loader

pair_it = infinite(train_pair_ld)
cap_loss = base_loss["forget_heldout"] + CAP_DELTA

print(f"Executando Unlearning (Meta NLL Forget: {cap_loss:.2f}, Retain Baseline: {base_loss['retain_heldout']:.2f})...")
for step in range(1, STEPS + 1):
    opt.zero_grad(set_to_none=True)
    for w, _ in active_layers:
        w.grad = None

    lf_accum, lr_accum = 0.0, 0.0

    for _ in range(GRAD_ACCUM):
        fb, rb = next(pair_it)
        fb, rb = to_dev(fb), to_dev(rb)

        model.train()
        # 1. Empurra o Forget para cima com força total enquanto estiver abaixo de cap_loss
        with autocast():
            lf = nll(model, fb, ban_eos_boost=True)
            # Soft-hinge quadrático/suave: gradiente forte abaixo de cap_loss, zero acima de cap_loss
            gap = F.relu(cap_loss - lf)
            loss_f = (0.5 * gap * gap / CAP_DELTA) / GRAD_ACCUM
        if loss_f.requires_grad:
            loss_f.backward()
        lf_accum += lf.item() / GRAD_ACCUM
        del fb, lf, loss_f, gap

        # 2. Mantém o Retain ancorado
        with autocast():
            lr_ = nll(model, rb, ban_eos_boost=False)
            loss_r = (LAMBDA_RETAIN * lr_) / GRAD_ACCUM
        loss_r.backward()
        lr_accum += lr_.item() / GRAD_ACCUM
        del rb, lr_, loss_r

    # Copia gradientes das colunas selecionadas para os Master Weights FP32
    with torch.no_grad():
        for (w, ix), col_p in zip(active_layers, master_cols):
            if w.grad is not None:
                col_p.grad = w.grad[:, ix].float()
                w.grad = None

    torch.nn.utils.clip_grad_norm_(master_cols, MAX_GRAD_NORM)
    opt.step()

    # Atualiza as colunas selecionadas no modelo em bfloat16
    with torch.no_grad():
        for (w, ix), col_p in zip(active_layers, master_cols):
            w[:, ix] = col_p.to(w.dtype)

    if step % 10 == 0:
        print(f"Passo {step:3d}/{STEPS} | NLL Forget (Batch): {lf_accum:.3f} | NLL Retain (Batch): {lr_accum:.3f}")

model.gradient_checkpointing_disable()
torch.cuda.empty_cache()

# ---------- 5. Avaliação Pós-Unlearning & Gravação dos Logs .txt ----------
print("\n--- GERAÇÃO APÓS UNLEARNING ---")
unlearned_generations = test_generation(model, forget_test_raw, n_samples=N_EVAL_SAMPLES)
for item in unlearned_generations:
    print(f"[Prompt]: ...{item['prompt_short']}\n[Resposta]: {item['generated'][:120]}...\n")

unlearned_loss = evaluate_loss(model)

for idx in range(N_EVAL_SAMPLES):
    b_item = base_generations[idx]
    a_item = ablated_generations[idx]
    u_item = unlearned_generations[idx]

    txt_content = (
        f"query {idx + 1}:\n{b_item['prompt']}\n\n"
        f"o que deve ser desaprendido (alvo rejected do dataset):\n{b_item['target_forget']}\n\n"
        f"antes do unlearning (baseline):\n{b_item['generated']}\n\n"
        f"com ablação direta (neurônios zerados):\n{a_item['generated']}\n\n"
        f"depois do unlearning:\n{u_item['generated']}\n"
    )

    file_path = os.path.join(OUT_DIR, f"query_{idx + 1}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(txt_content)

metrics_table = (
    "=" * 65 + "\n"
    f"{'Métrica (NLL)':<25} | {'Baseline':<12} | {'Unlearned':<12} | {'Ablação':<12}\n"
    + "-" * 65 + "\n"
    f"{'Forget (Held-out)':<25} | {base_loss['forget_heldout']:<12.3f} | {unlearned_loss['forget_heldout']:<12.3f} | {ablated_loss['forget_heldout']:<12.3f}\n"
    f"{'Retain (Held-out)':<25} | {base_loss['retain_heldout']:<12.3f} | {unlearned_loss['retain_heldout']:<12.3f} | {ablated_loss['retain_heldout']:<12.3f}\n"
    + "=" * 65 + "\n"
)

print("\n" + metrics_table)

metrics_path = os.path.join(OUT_DIR, "resumo_metricas.txt")
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write(metrics_table)

print(f"Arquivos de log salvos em: {OUT_DIR}/ (query_1.txt até query_{N_EVAL_SAMPLES}.txt + resumo_metricas.txt)")
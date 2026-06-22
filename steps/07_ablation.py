"""
STEP 7 — Ablation: how much does each piece (MoE, MLA) actually buy?
===================================================================

The headline experiment. To know what MoE and MLA each contribute, you can't just look at
the full model — you have to TURN EACH ON AND OFF and compare. So we train FOUR variants on
the exact same data, for the same steps, changing only two toggles:

    variant   attention   FFN        what it isolates
    -------   ---------   ---------  -------------------------------
    dense     GQA         SwiGLU     the baseline (modern-nanoGPT style)
    +MoE      GQA         MoE        what MoE adds on its own
    +MLA      MLA         SwiGLU     what MLA adds on its own
    both      MLA         MoE        nano-moe-mla (the full sparse model)

For each we report VAL LOSS (quality) plus the structural wins: total vs active params
(MoE's sparsity) and KV-cache floats/token (MLA's smaller cache).

Honest read at nano scale: the val-loss differences will be small and a bit noisy — MoE/MLA
pay off mostly at scale. The value here is the METHOD (isolate each feature) and the
structural numbers, which are real regardless of scale.

Run:  python steps/07_ablation.py
"""

import os
import math
import importlib.util
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m3 = _load("mola_model", "03_block_model.py")
m5 = _load("mola_data",  "05_multidomain.py")
MoeMlaGPT, MoeMlaConfig = m3.MoeMlaGPT, m3.MoeMlaConfig
CharMultiDomain, load_domains = m5.CharMultiDomain, m5.load_domains

device      = "cuda" if torch.cuda.is_available() else "cpu"
block_size  = 128
batch_size  = 32
train_iters = 1500          # per variant; 4 variants → a few minutes total
eval_iters  = 50
torch.manual_seed(1337)
print(f"[ablation] device = {device}")
data = CharMultiDomain(load_domains(), block_size, device)


@torch.no_grad()
def eval_val(model):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y, _ = data.get_batch("val", batch_size, domain=None)
        _, loss = model(x, y)
        losses[k] = loss.item()
    return losses.mean().item()


def run(use_moe, use_mla):
    """Train one variant and return (val_loss, total_params, active_params, kv_per_token)."""
    cfg = MoeMlaConfig(vocab_size=data.vocab_size, block_size=block_size, n_layer=4,
                     n_head=4, head_dim=16, n_embd=64,
                     n_experts=8, top_k=2, n_shared=1, n_kv_head=2, d_rope=8, d_latent=32,
                     use_moe=use_moe, use_mla=use_mla)         # ← the two toggles being ablated
    model = MoeMlaGPT(cfg).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    model.train()
    for _ in range(train_iters):
        x, y, _ = data.get_batch("train", batch_size, domain=None)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    val = eval_val(model)
    kv  = (cfg.d_latent + cfg.d_rope) if use_mla else (2 * cfg.n_kv_head * cfg.head_dim)
    return val, model.num_params(), model.active_params_per_token(), kv


# ----------------------------- run the 4 variants -----------------------------
variants = [("dense", False, False), ("+MoE", True, False), ("+MLA", False, True), ("both", True, True)]
rows = []
for name, use_moe, use_mla in variants:
    print(f"[ablation] training {name:6s} (MoE={use_moe}, MLA={use_mla})...")
    val, total, active, kv = run(use_moe, use_mla)
    rows.append((name, val, total, active, kv))
    print(f"           val {val:.4f}   total {total/1e3:.0f}K   active/tok {active/1e3:.0f}K   KV/tok {kv}")

# ----------------------------- the table -----------------------------
print("\n=== ablation (same data, same steps) ===")
print(f"{'variant':8s} {'val loss':>9s} {'total':>8s} {'active/tok':>11s} {'KV/tok':>7s}")
for name, val, total, active, kv in rows:
    print(f"{name:8s} {val:>9.4f} {total/1e3:>7.0f}K {active/1e3:>10.0f}K {kv:>7d}")

# ----------------------------- bar chart of val loss -----------------------------
names = [r[0] for r in rows]; vals = [r[1] for r in rows]
plt.figure(figsize=(6, 3.6))
bars = plt.bar(names, vals, color=["#888", "#1f6feb", "#1d9e75", "#8957e5"])
plt.ylabel("val loss (cross-entropy)"); plt.title("nano-moe-mla ablation — dense vs +MoE vs +MLA vs both")
plt.ylim(min(vals) - 0.05, max(vals) + 0.05)
for b, v in zip(bars, vals):
    plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
out = os.path.join(HERE, "..", "ablation.png")
plt.savefig(out, dpi=120)
print(f"\nchart saved to {out}")
print("OK — ablation done: each feature isolated on val loss + structural wins.")

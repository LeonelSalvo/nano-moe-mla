"""
STEP 6 — Routing probe: do the experts specialize by domain? (+ the balancing tradeoff)
=======================================================================================

The measurement of "expertise". Plan:
  1) train nano-moe-mla on the MIXED multi-domain corpus (step 5),
  2) feed it VAL text from ONE domain at a time and record, per token, which expert the
     router picks (top-1), across all MoE layers,
  3) build a domain × expert matrix → heatmap, and one number: the MUTUAL INFORMATION
     I(domain ; expert) in bits (0 = router ignores the domain; higher = it specializes).

THE TWIST we measure here: load balancing (#1, the DeepSeek bias trick) keeps experts
evenly used — which is good (no collapse) but FLATTENS the domain→expert structure (it
pushes routing toward uniform). So there's a real tradeoff: balance vs specialization.
This script trains TWICE — load-balancing ON and OFF — and compares the two MIs and
heatmaps, so you can SEE the tradeoff.

Honest expectation at nano scale: low MI overall (char-level, tiny model, top_k=2). The
OFF run usually shows a bit MORE specialization (higher MI) but risks expert collapse;
the ON run is flatter but balanced. The point is to measure the tradeoff, not to win.

Run:  python steps/06_routing_probe.py   (needs the multi-domain files from step 5)
"""

import os
import math
import importlib.util
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    """Import a module whose filename starts with a digit (can't be a normal import)."""
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
train_iters = 2000
torch.manual_seed(1337)
print(f"[probe] device = {device}")
data = CharMultiDomain(load_domains(), block_size, device)


def train_and_probe(load_balance):
    """Train a fresh model (balancing on/off), then measure domain→expert routing."""
    cfg = MoeMlaConfig(vocab_size=data.vocab_size, block_size=block_size, n_layer=4,
                     n_head=4, head_dim=16, n_embd=64,
                     n_experts=8, top_k=2, n_shared=1, d_rope=8, d_latent=32,
                     load_balance=load_balance)              # ← the toggle we're studying
    model = MoeMlaGPT(cfg).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    # 1) train on the MIXED corpus
    model.train()
    for it in range(1, train_iters + 1):
        x, y, _ = data.get_batch("train", batch_size, domain=None)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # 2) probe: per domain, count the router's top-1 expert for every token (all MoE layers).
    #    We grab each MoE's input with a forward pre-hook (no model changes needed).
    @torch.no_grad()
    def counts_for(domain, n_batches=20):
        counts, captured = torch.zeros(cfg.n_experts), []

        def hook(module, inp):
            xf = inp[0].reshape(-1, inp[0].shape[-1])
            top1 = F.softmax(module.router(xf), dim=-1).argmax(dim=-1)
            captured.append(top1)

        handles = [b.moe.register_forward_pre_hook(hook) for b in model.blocks]
        model.eval()
        for _ in range(n_batches):
            captured.clear()
            x, _, _ = data.get_batch("val", batch_size, domain=domain)
            model(x)
            for t in captured:
                counts += torch.bincount(t.cpu(), minlength=cfg.n_experts).float()
        for h in handles:
            h.remove()
        return counts

    rows = torch.stack([counts_for(name) for name in data.names])   # (domains, experts) counts
    frac = rows / rows.sum(dim=1, keepdim=True)                      # each row → distribution

    # 3) mutual information I(domain; expert), in bits
    joint = rows / rows.sum()
    p_dom = joint.sum(dim=1, keepdim=True)
    p_exp = joint.sum(dim=0, keepdim=True)
    mask  = joint > 0
    mi    = (joint[mask] * (joint[mask] / (p_dom * p_exp)[mask]).log2()).sum().item()
    return mi, frac


def heatmap(frac, mi, tag):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    im = ax.imshow(frac.numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(frac.shape[1])); ax.set_xticklabels([f"E{e}" for e in range(frac.shape[1])])
    ax.set_yticks(range(len(data.names))); ax.set_yticklabels(data.names)
    ax.set_xlabel("expert"); ax.set_title(f"domain → expert  ({tag},  MI = {mi:.2f} bits)")
    for i in range(frac.shape[0]):
        for j in range(frac.shape[1]):
            ax.text(j, i, f"{frac[i,j]*100:.0f}", ha="center", va="center",
                    color="white" if frac[i, j] < 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="fraction of tokens")
    fig.tight_layout()
    out = os.path.join(HERE, "..", f"routing_heatmap_{tag}.png")
    fig.savefig(out, dpi=120); print(f"  heatmap → {out}")


# ----------------------------- run the comparison -----------------------------
print("\n[probe] training WITH load balancing (on)...")
mi_on,  frac_on  = train_and_probe(load_balance=True)
print("[probe] training WITHOUT load balancing (off)...")
mi_off, frac_off = train_and_probe(load_balance=False)

print("\n=== specialization vs balancing ===")
print(f"  load-balancing ON  : I(domain;expert) = {mi_on:.3f} bits  (balanced experts, flatter routing)")
print(f"  load-balancing OFF : I(domain;expert) = {mi_off:.3f} bits  (more specialization, risk of collapse)")
print(f"  max possible       : {math.log2(len(data.names)):.3f} bits")
heatmap(frac_on,  mi_on,  "lb-on")
heatmap(frac_off, mi_off, "lb-off")
print("\nOK — routing tradeoff measured. On to step 7 (ablation: isolate each feature on val loss).")

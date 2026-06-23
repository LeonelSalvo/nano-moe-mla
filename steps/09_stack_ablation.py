"""
STEP 9 — Stack ablation: measure each ARCHITECTURE technique ON vs OFF
=====================================================================

Trains the SAME model on the multi-domain corpus (step 5) under different settings, flipping
ONE technique at a time, and prints a matrix:  setting → val loss (pure next-token CE) + MI.

Architecture / routing techniques covered: MoE, MLA, load-balancing (#1), router z-loss (#3),
QK-Norm (#5), sandwich-norm (#9), noisy top-k (#4), and top_k=1 vs 2.

(Cross-cutting techniques that are NOT specific to this architecture — the Muon optimizer, MTP,
and the from-scratch BPE — live in the companion repo `frontier-llm-techniques-2026-Q1`.)

Val loss reported is ALWAYS the plain next-token cross-entropy (the z-loss aux term is a training
signal, not a comparable loss), so every row is apples-to-apples.

EVERY run is SAVED under results/<run-tag>/: the metrics (CSV + JSON), the plots (val-CE and MI bar
charts with error bars, plus a routing heatmap per MoE setting), and the BASE model checkpoint — so a
long run is never lost. (Checkpoints are .pt → gitignored; the CSV/JSON/PNG are kept for the README.)

Scale (sized for a single 24 GB GPU, e.g. RTX 3090):
  SCALE=nano  (default) — fast smoke test, runs in a few minutes, metrics are tiny/noisy.
  SCALE=micro           — the real run: ~6 layers / 384 dim / 16 experts / block 512.
For meaningful MI, build a real BPE corpus first (data_prep.py) and use TOKENIZER=bpe SCALE=micro.

Run:  python steps/09_stack_ablation.py                       # nano, 1 seed (fast smoke)
      SEEDS=3 SCALE=micro TOKENIZER=bpe python steps/09_stack_ablation.py   # the real measurement

Knobs (env): SCALE=nano|micro · SEEDS=N (runs per setting → error bar) · ITERS · TOKENIZER=char|bpe · LR.
Reporting mean ± std over seeds is what separates a real effect from noise: if two settings'
error bars overlap, the difference isn't significant at this scale — and that's an honest result.
"""

import os
import re
import csv
import json
import math
import time
import statistics
import importlib.util
from dataclasses import asdict
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

m3 = _load("mola_model", "03_block_model.py")
m5 = _load("mola_data",  "05_multidomain.py")
MoeMlaGPT, MoeMlaConfig = m3.MoeMlaGPT, m3.MoeMlaConfig
CharMultiDomain, load_domains = m5.CharMultiDomain, m5.load_domains

device  = "cuda" if torch.cuda.is_available() else "cpu"
SCALE   = os.environ.get("SCALE", "nano")
SEEDS   = int(os.environ.get("SEEDS", "1"))     # runs per setting; >1 → report mean ± std
BASE_LR = float(os.environ.get("LR", "3e-4"))

# --- scale presets (both fit in 24 GB; micro is the meaningful one) ---
if SCALE == "micro":
    ARCH = dict(n_layer=6, n_head=6, head_dim=64, n_embd=384, n_experts=16, top_k=2,
                n_shared=1, d_rope=16, d_latent=64)
    BLOCK, BATCH, ITERS, EVAL = 512, 24, 5000, 200
else:
    ARCH = dict(n_layer=4, n_head=4, head_dim=16, n_embd=64, n_experts=8, top_k=2,
                n_shared=1, d_rope=8, d_latent=32)
    BLOCK, BATCH, ITERS, EVAL = 128, 32, 600, 100

# env overrides (for a ~1h preliminary run: e.g. SCALE=micro ITERS=1500)
ITERS = int(os.environ.get("ITERS", ITERS))
BATCH = int(os.environ.get("BATCH", BATCH))
TOKENIZER = os.environ.get("TOKENIZER", "char")     # "char" (verified) | "bpe" (real metrics)

print(f"[ablation] scale={SCALE}  tokenizer={TOKENIZER}  device={device}  iters={ITERS}  seeds={SEEDS}")
if TOKENIZER == "bpe":
    data = _load("bpe_data", "../bpe_data.py").BpeMultiDomain(BLOCK, device)
else:
    data = CharMultiDomain(load_domains(), BLOCK, device)

# results dir for THIS run (timestamped so nothing is ever overwritten)
RUN_TAG = f"{SCALE}_{TOKENIZER}_{ITERS}it_{SEEDS}s_{time.strftime('%Y%m%d-%H%M%S')}"
OUT = os.path.join(HERE, "..", "results", RUN_TAG)
os.makedirs(OUT, exist_ok=True)
RESULTS = []   # one dict per setting, filled by train_eval


def lr_at(it, warmup=max(20, ITERS // 50)):
    """Linear warmup → cosine decay to 10% of BASE_LR (so longer micro runs settle well)."""
    if it < warmup:
        return BASE_LR * (it + 1) / warmup
    r = (it - warmup) / max(1, ITERS - warmup)
    return BASE_LR * 0.1 + 0.5 * BASE_LR * 0.9 * (1 + math.cos(math.pi * r))


def make_cfg(**over):
    base = dict(vocab_size=data.vocab_size, block_size=BLOCK, **ARCH)
    base.update(over)
    return MoeMlaConfig(**base)


@torch.no_grad()
def val_ce(model):
    """Pure next-token cross-entropy on val — the comparable number (ignores aux terms)."""
    model.eval()
    tot = 0.0
    n = EVAL // 10 or 5
    for _ in range(n):
        x, y, _ = data.get_batch("val", BATCH, domain=None)
        logits, _ = model(x)
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
    model.train()
    return tot / n


@torch.no_grad()
def measure_mi(model, cfg):
    """I(domain; expert) in bits + the domain×expert fraction matrix (for the heatmap)."""
    if not cfg.use_moe:
        return None, None
    rows = []
    for name in data.names:
        counts, captured = torch.zeros(cfg.n_experts), []
        def hook(mod, inp):
            xf = inp[0].reshape(-1, inp[0].shape[-1])
            captured.append(F.softmax(mod.router(xf), dim=-1).argmax(dim=-1))
        handles = [b.moe.register_forward_pre_hook(hook) for b in model.blocks]
        model.eval()
        for _ in range(15):
            captured.clear()
            x, _, _ = data.get_batch("val", BATCH, domain=name)
            model(x)
            for t in captured:
                counts += torch.bincount(t.cpu(), minlength=cfg.n_experts).float()
        for h in handles:
            h.remove()
        rows.append(counts)
    rows = torch.stack(rows)
    frac = (rows / rows.sum(dim=1, keepdim=True)).cpu().numpy()       # each domain → its expert distribution
    joint = rows / rows.sum()
    p_dom, p_exp = joint.sum(1, keepdim=True), joint.sum(0, keepdim=True)
    mask = joint > 0
    mi = (joint[mask] * (joint[mask] / (p_dom * p_exp)[mask]).log2()).sum().item()
    return mi, frac


def run_once(cfg, seed):
    """One full train + eval for a given seed. Returns (val CE, MI, frac, model)."""
    torch.manual_seed(seed)
    model = MoeMlaGPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=(0.9, 0.95), weight_decay=0.1)
    model.train()
    for it in range(ITERS):
        for g in opt.param_groups:                       # warmup → cosine schedule
            g["lr"] = lr_at(it)
        x, y, _ = data.get_batch("train", BATCH, domain=None)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    ce = val_ce(model)
    mi, frac = measure_mi(model, cfg)
    return ce, mi, frac, model


def _fmt(xs):
    if not xs:
        return "   —   "
    m = statistics.mean(xs)
    if len(xs) == 1:
        return f"{m:.3f}      "
    return f"{m:.3f}±{statistics.pstdev(xs):.3f}"


def train_eval(label, over, keep_model=False):
    """Run SEEDS times, report mean ± std, and record everything into RESULTS."""
    cfg = make_cfg(**over)
    ces, mis, last_frac, kept = [], [], None, None
    for s in range(SEEDS):
        ce, mi, frac, model = run_once(cfg, 1337 + s)
        ces.append(ce)
        if mi is not None:
            mis.append(mi)
            last_frac = frac
        if keep_model:
            kept = (model, cfg)
    print(f"  {label:24s}  val CE {_fmt(ces)}   MI {_fmt(mis)}")
    RESULTS.append(dict(label=label, ces=ces, mis=mis, frac=last_frac))
    if kept is not None:
        _save_checkpoint(*kept)
    return statistics.mean(ces), (statistics.mean(mis) if mis else None)


# ----------------------------- artifact saving -----------------------------
def _slug(s):
    return re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_")


def _save_checkpoint(model, cfg):
    path = os.path.join(OUT, "ckpt_BASE.pt")
    torch.save({"model": model.state_dict(), "cfg": asdict(cfg),
                "tokenizer": TOKENIZER, "domains": data.names}, path)
    print(f"  [saved] BASE checkpoint → {path}")


def _stats(xs):
    if not xs:
        return None, None
    return statistics.mean(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def save_all():
    # 1) run config
    with open(os.path.join(OUT, "config.json"), "w") as f:
        json.dump(dict(scale=SCALE, tokenizer=TOKENIZER, iters=ITERS, seeds=SEEDS, lr=BASE_LR,
                       arch=ARCH, block_size=BLOCK, batch=BATCH, vocab_size=data.vocab_size,
                       domains=data.names), f, indent=2)
    # 2) the PLOTS first (the visual artifacts — written before anything that could fail)
    _bar("ce", "val cross-entropy (lower = better)", os.path.join(OUT, "val_ce.png"))
    _bar("mi", "I(domain; expert) bits (higher = more specialization)", os.path.join(OUT, "mi.png"))
    for r in RESULTS:
        if r["frac"] is not None:
            _heatmap(r["frac"], r["label"], os.path.join(OUT, f"heatmap_{_slug(r['label'])}.png"))
    # 3) metrics CSV (means, stds, and every per-seed value)
    with open(os.path.join(OUT, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "ce_mean", "ce_std", "mi_mean", "mi_std", "ces", "mis"])
        for r in RESULTS:
            cem, ces_sd = _stats(r["ces"]); mim, mis_sd = _stats(r["mis"])
            w.writerow([r["label"], cem, ces_sd, mim, mis_sd,
                        ";".join(f"{x:.4f}" for x in r["ces"]),
                        ";".join(f"{x:.4f}" for x in r["mis"])])
    # 4) metrics JSON — convert the frac ndarray to a list so it's JSON-serializable
    serializable = [{**r, "frac": (r["frac"].tolist() if r["frac"] is not None else None)}
                    for r in RESULTS]
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n[saved] all artifacts → {OUT}")
    print("        metrics.csv · metrics.json · config.json · val_ce.png · mi.png · heatmap_*.png · ckpt_BASE.pt")


def _bar(key, ylabel, path):
    labels, means, stds = [], [], []
    for r in RESULTS:
        xs = r["ces"] if key == "ce" else r["mis"]
        m, sd = _stats(xs)
        if m is None:
            continue
        labels.append(r["label"]); means.append(m); stds.append(sd)
    if not means:
        return
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.9), 4.2))
    ax.bar(range(len(means)), means, yerr=stds, capsize=4, color="#4878a8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(f"stack ablation ({SCALE}, {TOKENIZER}, {ITERS} it, {SEEDS} seeds)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  [saved] {os.path.basename(path)}")


def _heatmap(frac, label, path):
    fig, ax = plt.subplots(figsize=(max(6, frac.shape[1] * 0.5), 2.8))
    im = ax.imshow(frac, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(data.names))); ax.set_yticklabels(data.names)
    ax.set_xticks(range(frac.shape[1])); ax.set_xticklabels([f"E{e}" for e in range(frac.shape[1])], fontsize=7)
    ax.set_xlabel("expert"); ax.set_title(f"domain → expert  [{label}]", fontsize=9)
    fig.colorbar(im, ax=ax, label="fraction of tokens")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ----------------------------- the ablation matrix -----------------------------
if __name__ == "__main__":
    print(f"\n=== stack ablation (one technique flipped at a time, {SEEDS} seed(s)) ===")
    print("  setting                   val CE              MI(domain;expert)")
    base = train_eval("BASE (full stack)", {}, keep_model=True)
    train_eval("− MoE (dense FFN)", dict(use_moe=False))
    train_eval("− MLA (GQA attn)",  dict(use_mla=False))
    train_eval("− load-balancing",  dict(load_balance=False))
    train_eval("+ z-loss (#3)",     dict(z_loss_gamma=1e-3))
    train_eval("− QK-Norm",         dict(qk_norm=False))
    train_eval("− sandwich-norm",   dict(post_norm=False))
    train_eval("+ noisy top-k (#4)", dict(noisy_topk=True))
    train_eval("top_k=1 (Switch)",  dict(top_k=1))

    save_all()
    assert base[0] < math.log(data.vocab_size), "BASE didn't learn — check the setup"
    print("\nOK — stack matrix measured + saved. Compare each row vs BASE; trust gaps bigger than the ± std.")
    if SEEDS == 1:
        print("    (1 seed = no error bar → noisy. Re-run with SEEDS=3 SCALE=micro TOKENIZER=bpe for real numbers.)")
    else:
        print("    (overlapping error bars between two rows = the difference is within noise at this scale.)")

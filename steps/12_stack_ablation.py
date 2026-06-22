"""
STEP 12 — Stack ablation: measure each technique ON vs OFF
==========================================================

Trains the SAME model on the multi-domain corpus (step 5) under different settings, flipping
ONE technique at a time, and prints a matrix:  setting → val loss (pure next-token CE) + MI.

Techniques covered: MoE, MLA, load-balancing (#1), router z-loss (#3), QK-Norm (#5),
sandwich-norm (#9), noisy top-k (#4), top_k=1 vs 2, MTP (#13), and AdamW vs Muon (#14).

Val loss reported is ALWAYS the plain next-token cross-entropy (the aux terms — z-loss, MTP —
are training signals, not comparable losses), so every row is apples-to-apples.

Scale (sized for a single 24 GB GPU, e.g. RTX 3090):
  SCALE=nano  (default) — fast smoke test, runs in a few minutes, metrics are tiny/noisy.
  SCALE=micro           — the real run: ~6 layers / 384 dim / 16 experts / block 512.
For meaningful MI, download real domain files first (see steps/05_multidomain.py) and SCALE=micro.

Run:  python steps/12_stack_ablation.py                  # nano, 1 seed (fast smoke)
      SEEDS=3 SCALE=micro python steps/12_stack_ablation.py   # the real measurement, mean ± std

Knobs (env): SCALE=nano|micro · SEEDS=N (runs per setting → error bar) · LR=3e-4.
Reporting mean ± std over seeds is what separates a real effect from noise: if two settings'
error bars overlap, the difference isn't significant at this scale — and that's an honest result.
"""

import os
import math
import statistics
import importlib.util
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

m3 = _load("mola_model", "03_block_model.py")
m5 = _load("mola_data",  "05_multidomain.py")
mMuon = _load("muon_mod", "11_muon.py")
MoeMlaGPT, MoeMlaConfig = m3.MoeMlaGPT, m3.MoeMlaConfig
CharMultiDomain, load_domains = m5.CharMultiDomain, m5.load_domains
Muon = mMuon.Muon

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


def build_optimizers(model, lr, use_muon):
    """AdamW (default) or Muon (hidden 2D matrices) + AdamW (embeddings/tied head/norms)."""
    if not use_muon:
        return [torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)]
    emb_ids, seen, muon_p, adamw_p = {id(model.tok_emb.weight)}, set(), [], []
    for _n, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (muon_p if (p.ndim == 2 and id(p) not in emb_ids) else adamw_p).append(p)
    return [Muon(muon_p, lr=lr, momentum=0.95),
            torch.optim.AdamW(adamw_p, lr=lr, betas=(0.9, 0.95), weight_decay=0.1)]


@torch.no_grad()
def val_ce(model):
    """Pure next-token cross-entropy on val — the comparable number (ignores aux terms)."""
    model.eval()
    tot = 0.0
    for _ in range(EVAL // 10 or 5):
        x, y, _ = data.get_batch("val", BATCH, domain=None)
        logits, _ = model(x)
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
    model.train()
    return tot / (EVAL // 10 or 5)


@torch.no_grad()
def measure_mi(model, cfg):
    """I(domain; expert) in bits via the router's top-1 choice per token (all MoE layers)."""
    if not cfg.use_moe:
        return None
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
    joint = rows / rows.sum()
    p_dom, p_exp = joint.sum(1, keepdim=True), joint.sum(0, keepdim=True)
    mask = joint > 0
    return (joint[mask] * (joint[mask] / (p_dom * p_exp)[mask]).log2()).sum().item()


def run_once(cfg, use_muon, seed):
    """One full train + eval for a given seed. Returns (val CE, MI or None)."""
    torch.manual_seed(seed)
    model = MoeMlaGPT(cfg).to(device)
    opts = build_optimizers(model, lr=BASE_LR, use_muon=use_muon)
    want_t2 = cfg.mtp
    model.train()
    for it in range(ITERS):
        lr = lr_at(it)                                   # warmup → cosine schedule
        for o in opts:
            for g in o.param_groups:
                g["lr"] = lr
        if want_t2:
            x, y, t2, _ = data.get_batch("train", BATCH, domain=None, want_t2=True)
            _, loss = model(x, y, t2)
        else:
            x, y, _ = data.get_batch("train", BATCH, domain=None)
            _, loss = model(x, y)
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
    return val_ce(model), measure_mi(model, cfg)


def _fmt(xs):
    if not xs:
        return "   —   "
    m = statistics.mean(xs)
    if len(xs) == 1:
        return f"{m:.3f}      "
    return f"{m:.3f}±{statistics.pstdev(xs):.3f}"


def train_eval(label, over, use_muon=False):
    """Run SEEDS times and report mean ± std — the error bar is what tells signal from noise."""
    cfg = make_cfg(**over)
    ces, mis = [], []
    for s in range(SEEDS):
        ce, mi = run_once(cfg, use_muon, 1337 + s)
        ces.append(ce)
        if mi is not None:
            mis.append(mi)
    print(f"  {label:24s}  val CE {_fmt(ces)}   MI {_fmt(mis)}")
    return statistics.mean(ces), (statistics.mean(mis) if mis else None)


# ----------------------------- the ablation matrix -----------------------------
if __name__ == "__main__":
    print(f"\n=== stack ablation (one technique flipped at a time, {SEEDS} seed(s)) ===")
    print("  setting                   val CE              MI(domain;expert)")
    base = train_eval("BASE (full stack)", {})
    train_eval("− MoE (dense FFN)",      dict(use_moe=False))
    train_eval("− MLA (GQA attn)",       dict(use_mla=False))
    train_eval("− load-balancing",       dict(load_balance=False))
    train_eval("+ z-loss (#3)",          dict(z_loss_gamma=1e-3))
    train_eval("− QK-Norm",              dict(qk_norm=False))
    train_eval("− sandwich-norm",        dict(post_norm=False))
    train_eval("+ noisy top-k (#4)",     dict(noisy_topk=True))
    train_eval("top_k=1 (Switch)",       dict(top_k=1))
    train_eval("+ MTP (#13)",            dict(mtp=True))
    train_eval("Muon optimizer (#14)",   {}, use_muon=True)

    assert base[0] < math.log(data.vocab_size), "BASE didn't learn — check the setup"
    print("\nOK — stack matrix measured. Compare each row vs BASE, but ONLY trust gaps bigger than the ± std.")
    if SEEDS == 1:
        print("    (1 seed = no error bar → noisy. Re-run with SEEDS=3 SCALE=micro for a real measurement.)")
    else:
        print("    (overlapping error bars between two rows = the difference is within noise at this scale.)")

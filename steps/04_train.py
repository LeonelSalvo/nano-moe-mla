"""
STEP 4 — Train it, and put the numbers next to the dense baseline
=================================================================

The pieces are built and verified (MoE, MLA, the sparse model). Now we TRAIN, the same
3-step loop as always:

    repeat:
        1) FORWARD : run a batch through the model → loss (how badly it predicts)
        2) BACKWARD: loss.backward() → every weight gets its gradient
        3) UPDATE  : the optimizer nudges each weight against its gradient
        (+ every so often: evaluate on train/val, log, keep the best checkpoint)

If everything is wired right and gradients flow, THE LOSS DROPS — that's the real check
that MoE + MLA are connected correctly. Then we print the SPARSE STORY in numbers:
  • total params  ≫  active params per token   (the MoE win — capacity for cheap compute)
  • KV cached per token  ≪  full attention      (the MLA win — cheap long context)
The dense LOSS reference is modern-nanoGPT (val ≈ 1.48 on this same TinyShakespeare).

Run:  python steps/04_train.py
  (optional real data:)
  mkdir -p data && curl -o data/input.txt \
    https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
"""

import os
import csv
import math
import time
import importlib.util
from dataclasses import asdict
import torch

# --- import the model we assembled in step 3 (its filename starts with a digit, so we
#     load it by path instead of a normal "import") ---
HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("mola_model", os.path.join(HERE, "03_block_model.py"))
mola  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mola)
MoeMlaGPT, MoeMlaConfig = mola.MoeMlaGPT, mola.MoeMlaConfig


# ----------------------------- data: text → tokens -----------------------------
_FALLBACK = (
    "The mixture of experts routes each token to a few specialists, while latent "
    "attention keeps only a small compressed memory of the past. Sparse where it can "
    "be, dense where it must.\n"
) * 400


def load_text():
    """Use data/input.txt (e.g. TinyShakespeare) if present; else a small embedded text
    so the script runs with no download."""
    path = os.path.join(HERE, "..", "data", "input.txt")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    print("[data] no data/input.txt → using a tiny embedded text. "
          "curl TinyShakespeare into data/input.txt for a real run.")
    return _FALLBACK


class CharData:
    """Character-level dataset: maps each distinct character to an integer id, splits into
    train/val, and serves random (input, next-token) batches."""
    def __init__(self, text, block_size, device, split=0.9):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(len(data) * split)
        self.train, self.val = data[:n], data[n:]
        self.block_size, self.device = block_size, device

    def get_batch(self, split, batch_size):
        d = self.train if split == "train" else self.val
        ix = torch.randint(len(d) - self.block_size - 1, (batch_size,))           # random start points
        x = torch.stack([d[i:i + self.block_size] for i in ix])                   # the inputs
        y = torch.stack([d[i + 1:i + 1 + self.block_size] for i in ix])           # shifted by 1 = the targets
        return x.to(self.device), y.to(self.device)

    def decode(self, t):
        return "".join(self.itos[int(i)] for i in t)


# ----------------------------- hyperparameters -----------------------------
block_size    = 128       # context length (T)
batch_size    = 32
max_iters     = 3000
eval_interval = 250
eval_iters    = 50        # batches averaged per eval (a stable loss number)
lr            = 3e-4      # max learning rate
min_lr        = 3e-5
warmup_iters  = 100
grad_clip     = 1.0

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(1337)
print(f"[train] device = {device}")

# --- data + model ---
data = CharData(load_text(), block_size, device)
cfg  = MoeMlaConfig(vocab_size=data.vocab_size, block_size=block_size, n_layer=4,
                  n_head=4, head_dim=16, n_embd=64,
                  n_experts=8, top_k=2, n_shared=1, d_rope=8, d_latent=32)
model = MoeMlaGPT(cfg).to(device)
print(f"[train] vocab={cfg.vocab_size}  "
      f"total={model.num_params()/1e3:.1f}K  active/token={model.active_params_per_token()/1e3:.1f}K")

# AdamW: Adam (per-weight adaptive step) + weight decay (regularization). betas are its
# internal moving averages.
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)


def get_lr(it):
    """Learning-rate schedule: warm up linearly from ~0, then cosine-decay to min_lr.
    Start gentle (avoid early instability), end fine (settle into a good minimum)."""
    if it < warmup_iters:
        return lr * (it + 1) / warmup_iters
    ratio = (it - warmup_iters) / max(1, max_iters - warmup_iters)               # 0 → 1
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * ratio))         # cosine 1 → 0


@torch.no_grad()
def estimate_loss():
    """Average loss over several batches for train and val. Val is the one that matters:
    it says whether the model GENERALIZES or just memorizes."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = data.get_batch(split, batch_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# loss-curve log (one row per eval) to plot later
os.makedirs(os.path.join(HERE, "..", "data"), exist_ok=True)
log = open(os.path.join(HERE, "..", "data", "train_log.csv"), "w")
log.write("iter,train,val,lr\n")

# ===================== THE TRAINING LOOP =====================
best_val, t0 = float("inf"), time.time()
for it in range(max_iters + 1):
    for g in optimizer.param_groups:                 # set this step's lr from the schedule
        g["lr"] = get_lr(it)

    if it % eval_interval == 0:                       # evaluate + log + keep best checkpoint
        l = estimate_loss()
        print(f"  iter {it:5d}  train {l['train']:.4f}  val {l['val']:.4f}  lr {get_lr(it):.2e}")
        log.write(f"{it},{l['train']:.4f},{l['val']:.4f},{get_lr(it):.6f}\n"); log.flush()
        if l["val"] < best_val:
            best_val = l["val"]
            # save cfg as a plain dict (asdict) — pickling the MoeMlaConfig CLASS fails because
            # it lives in a module we loaded by path, not a normally-importable one.
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg),
                        "stoi": data.stoi, "itos": data.itos},
                       os.path.join(HERE, "..", "ckpt.pt"))

    x, y = data.get_batch("train", batch_size)        # 1) a batch
    _, loss = model(x, y)                             #    FORWARD: predict + measure loss
    optimizer.zero_grad(set_to_none=True)             #    clear old gradients
    loss.backward()                                   # 2) BACKWARD: gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)   # cut giant gradients
    optimizer.step()                                  # 3) UPDATE: move the weights

log.close()
print(f"[train] done in {(time.time()-t0)/60:.1f} min. Best val loss: {best_val:.4f}")

# ----------------------------- the sparse story, in numbers -----------------------------
# MoE: total params is large, but only top_k (+shared) experts run per token.
# MLA: the KV cache per token is the compressed latent + one shared rope key — far less
#      than caching full K and V per head (MHA).
kv_mla = cfg.d_latent + cfg.d_rope
kv_mha = 2 * cfg.n_head * cfg.head_dim
print("\n--- dense vs sparse (this model) ---")
print(f"params:    total {model.num_params()/1e3:.1f}K   |   active/token {model.active_params_per_token()/1e3:.1f}K   "
      f"({100*model.active_params_per_token()//model.num_params()}% active)   ← MoE")
print(f"KV cache:  MLA {kv_mla} floats/token   vs   full-attention (MHA) {kv_mha}   "
      f"({100*kv_mla//kv_mha}%)   ← MLA")
print("dense LOSS reference: modern-nanoGPT reached val ~1.48 on this same TinyShakespeare.")

# ----------------------------- a sample from the trained model -----------------------------
ckpt = torch.load(os.path.join(HERE, "..", "ckpt.pt"), map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"]); model.eval()
start  = torch.zeros((1, 1), dtype=torch.long, device=device)
tokens = model.generate(start, max_new_tokens=400, temperature=0.8, top_k=40)[0].tolist()
print("\n--- sample ---\n" + data.decode(tokens))

assert best_val < math.log(cfg.vocab_size) - 0.3, "the loss barely moved — check for a bug"
print("\nOK — nano-moe-mla trains: the loss dropped, and it's sparse (MoE) with a small KV cache (MLA).")

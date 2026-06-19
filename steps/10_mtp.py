"""
STEP 10 — Multi-Token Prediction (MTP)  (feature #13)
=====================================================

A normal LM head predicts the NEXT token (position t+1). MTP adds a SECOND head that, from
the same hidden state, predicts the token TWO ahead (t+2). Training on both gives a denser
learning signal (DeepSeek-V3 used it in pretraining), and at inference it lets you DRAFT two
tokens per step (speculative-style speedups).

    loss = CE(head1, x[t+1])  +  lambda * CE(head2, x[t+2])

We build a tiny transformer with two heads, train it on a small text, and check that BOTH
losses start near ln(vocab) and drop together.

Run:  python steps/10_mtp.py
"""

import os
import math
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("mola_model", os.path.join(HERE, "03_block_model.py"))
m3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(m3)
RMSNorm, GQA, Expert = m3.RMSNorm, m3.GQA, m3.Expert
build_rope_cache, apply_rope = m3.build_rope_cache, m3.apply_rope


class Block(nn.Module):
    """A plain transformer block (GQA + SwiGLU), reused from step 3's pieces."""
    def __init__(self, n_embd, n_head, n_kv_head, head_dim):
        super().__init__()
        self.norm1 = RMSNorm(n_embd); self.attn = GQA(n_embd, n_head, n_kv_head, head_dim)
        self.norm2 = RMSNorm(n_embd); self.ffn  = Expert(n_embd)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class TinyMTP(nn.Module):
    def __init__(self, vocab, n_embd=64, n_head=4, n_kv_head=2, head_dim=16, n_layer=2,
                 block_size=128, mtp_weight=0.5):
        super().__init__()
        self.mtp_weight = mtp_weight
        self.tok = nn.Embedding(vocab, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, n_kv_head, head_dim) for _ in range(n_layer)])
        self.norm = RMSNorm(n_embd)
        self.head1 = nn.Linear(n_embd, vocab, bias=False)   # predicts t+1 (the normal head)
        self.head2 = nn.Linear(n_embd, vocab, bias=False)   # predicts t+2 (the MTP head)
        cos, sin = build_rope_cache(head_dim, block_size)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx, t1=None, t2=None):
        B, T = idx.shape
        x = self.tok(idx)
        for blk in self.blocks:
            x = blk(x, self.cos[:T], self.sin[:T])
        x = self.norm(x)
        l1, l2 = self.head1(x), self.head2(x)               # both from the SAME hidden state
        loss = None
        if t1 is not None:
            loss1 = F.cross_entropy(l1.reshape(B * T, -1), t1.reshape(B * T))
            loss2 = F.cross_entropy(l2.reshape(B * T, -1), t2.reshape(B * T))
            loss = loss1 + self.mtp_weight * loss2
            return l1, l2, loss, (loss1.item(), loss2.item())
        return l1, l2, loss, None


# ----------------------------- TEST (self-checking) -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = ("the mixture of experts routes each token; latent attention compresses the past. "
            "sparse where it can be, dense where it must. ") * 300
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    vocab = len(chars)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=device)
    T = 64

    def batch(bs=32):
        ix = torch.randint(len(data) - T - 2, (bs,))
        x  = torch.stack([data[i:i + T]         for i in ix])     # input
        t1 = torch.stack([data[i + 1:i + 1 + T] for i in ix])     # target +1
        t2 = torch.stack([data[i + 2:i + 2 + T] for i in ix])     # target +2 (MTP)
        return x, t1, t2

    model = TinyMTP(vocab).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print("=== Step 10: Multi-Token Prediction ===")
    x, t1, t2 = batch()
    _, _, _, (l1_0, l2_0) = model(x, t1, t2)
    print(f"initial: loss(+1) {l1_0:.3f}   loss(+2) {l2_0:.3f}   ln(vocab) {math.log(vocab):.3f}")

    for it in range(1, 801):
        x, t1, t2 = batch()
        _, _, loss, _ = model(x, t1, t2)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    x, t1, t2 = batch()
    _, _, _, (l1_1, l2_1) = model(x, t1, t2)
    print(f"trained: loss(+1) {l1_1:.3f}   loss(+2) {l2_1:.3f}")
    # the +2 task is genuinely harder → its loss stays higher, but BOTH must drop
    assert l1_1 < l1_0 - 0.3 and l2_1 < l2_0 - 0.2, "MTP losses didn't drop"
    print("\nOK — MTP works: two heads (t+1 and t+2), both losses drop. The +2 head adds a denser signal.")

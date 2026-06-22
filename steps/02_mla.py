"""
STEP 2 — MLA (Multi-head Latent Attention; replaces GQA/MHA).

Zoom out: GQA shrank the KV cache by sharing K/V heads. MLA (DeepSeek-V2) goes further:
it COMPRESSES K and V into one small LATENT vector per token, caches only that latent,
and up-projects back to per-head K and V at compute time. Smaller cache than GQA, and
DeepSeek reports better quality too.

The RoPE snag (and the fix): RoPE must rotate the keys by position, but a key rebuilt
from a compressed latent can't carry a clean per-position rotation. DeepSeek's fix is
DECOUPLED RoPE: split q and k into two parts —
  • a CONTENT part (no RoPE) that comes from the compressed latent, and
  • a small ROPE part (carries the position) computed separately; the rope KEY is SHARED
    across heads (one per token).
The attention score is just the sum of the two dot products: q·k = q_content·k_content +
q_rope·k_rope. Content gives "what", rope gives "where (relative)".

Cached per token = the latent (d_latent) + the shared rope key (d_rope) — that's the win.

Test: shape preserved, causal (future doesn't leak), and the cached size is smaller than
plain MHA's (and than GQA's).

Run:  python steps/02_mla.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rope_cache(dim, seq_len, base=10000.0, device=None):
    idx      = torch.arange(0, dim, 2, device=device).float()
    inv_freq = 1.0 / (base ** (idx / dim))
    t        = torch.arange(seq_len, device=device).float()
    freqs    = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):                                 # x: (..., T, dim), cos/sin: (T, dim/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack((rx1, rx2), dim=-1).flatten(-2)


class MLA(nn.Module):
    def __init__(self, n_embd, n_head, head_dim=None, d_rope=8, d_latent=32):
        super().__init__()
        self.nh = n_head
        self.hd = head_dim or n_embd // n_head               # content dim per head
        self.dr = d_rope                                     # rope dim (per query head; shared for the key)
        self.dc = d_latent                                   # KV compression latent (the cached size)
        # queries: content (hd) + rope (dr) per head
        self.w_q   = nn.Linear(n_embd, n_head * (self.hd + self.dr), bias=False)
        # KV: down-project to the latent (CACHED), then up-project to per-head K-content and V
        self.w_dkv = nn.Linear(n_embd, self.dc, bias=False)
        self.w_uk  = nn.Linear(self.dc, n_head * self.hd, bias=False)
        self.w_uv  = nn.Linear(self.dc, n_head * self.hd, bias=False)
        # shared rope key: one per token, shared across heads (CACHED)
        self.w_kr  = nn.Linear(n_embd, self.dr, bias=False)
        self.w_o   = nn.Linear(n_head * self.hd, n_embd, bias=False)

    def forward(self, x, cos, sin, return_scores=False):     # x: (B, T, C)
        B, T, C = x.shape
        nh, hd, dr, dc = self.nh, self.hd, self.dr, self.dc

        # queries → split into content and rope parts
        q = self.w_q(x).view(B, T, nh, hd + dr).transpose(1, 2)   # (B, nh, T, hd+dr)
        q_c, q_r = q[..., :hd], q[..., hd:]                       # content / rope

        # latent KV (this is what a real inference cache would store)
        c_kv = self.w_dkv(x)                                      # (B, T, dc)  <-- cached
        k_c  = self.w_uk(c_kv).view(B, T, nh, hd).transpose(1, 2) # (B, nh, T, hd)
        v    = self.w_uv(c_kv).view(B, T, nh, hd).transpose(1, 2) # (B, nh, T, hd)

        # shared rope key (cached) + RoPE on both rope parts
        k_r = self.w_kr(x).view(B, 1, T, dr)                      # (B, 1, T, dr)  <-- cached, shared over heads
        q_r = apply_rope(q_r, cos, sin)
        k_r = apply_rope(k_r, cos, sin)

        # decoupled score = content dot + rope dot, then scale
        scale  = 1.0 / math.sqrt(hd + dr)
        scores = (q_c @ k_c.transpose(-2, -1) + q_r @ k_r.transpose(-2, -1)) * scale  # (B, nh, T, T)

        # causal mask + softmax + weighted sum of V
        future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        masked = scores.masked_fill(future, float("-inf"))
        attn   = torch.softmax(masked, dim=-1)
        out = attn @ v                                           # (B, nh, T, hd)
        out = out.transpose(1, 2).reshape(B, T, nh * hd)
        out = self.w_o(out)
        if return_scores:
            return out, scores, attn                             # scores = pre-mask, pre-softmax
        return out


# ----------------------------- TEST (self-checking) -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T = 1, 6
    n_embd, n_head, head_dim, d_rope, d_latent = 64, 4, 16, 8, 32
    mla = MLA(n_embd, n_head, head_dim, d_rope, d_latent)
    cos, sin = build_rope_cache(d_rope, T)

    print("=== Step 2: MLA ===")

    # (a) shape preserved
    x = torch.randn(B, T, n_embd)
    y = mla(x, cos, sin)
    print("input shape:", tuple(x.shape), " -> output shape:", tuple(y.shape))
    assert y.shape == x.shape, "the shape changed, something is wrong"

    # (b) causality: changing the last token must not affect earlier outputs
    x2 = x.clone()
    x2[:, -1, :] = torch.randn(n_embd)
    y2 = mla(x2, cos, sin)
    prev_unchanged = torch.allclose(y[:, :-1], y2[:, :-1], atol=1e-6)
    print("previous outputs intact after changing the future?:", prev_unchanged)
    assert prev_unchanged, "CAUSAL FAILURE: an earlier token saw the future"

    # (c) the KV-cache win: floats cached per token (latent + shared rope key) vs MHA / GQA
    mla_cache = d_latent + d_rope                    # MLA caches the latent + 1 shared rope key
    mha_cache = 2 * n_head * head_dim                # MHA caches full K and V per head
    gqa_cache = 2 * 2 * head_dim                     # GQA with n_kv_head=2
    print(f"KV cached / token  →  MLA: {mla_cache}   GQA(2): {gqa_cache}   MHA: {mha_cache}")
    assert mla_cache < mha_cache, "MLA should cache less than MHA"

    # (d) CORRECTNESS — attention is a valid probability distribution (rows ≥ 0 and sum to 1)
    _, _, attn = mla(x, cos, sin, return_scores=True)
    rows = attn.sum(dim=-1)
    print("attention rows sum to 1?:", torch.allclose(rows, torch.ones_like(rows), atol=1e-5),
          " | all weights ≥ 0?:", bool((attn >= 0).all()))
    assert (attn >= 0).all() and torch.allclose(rows, torch.ones_like(rows), atol=1e-5), \
        "attention is not a valid distribution"

    # (e) CORRECTNESS — decoupled RoPE encodes RELATIVE position: shifting ALL positions by k
    #     leaves the (pre-mask) score matrix unchanged, i.e. score(i,j) depends only on i−j.
    shift = 3
    cos_l, sin_l = build_rope_cache(d_rope, T + shift)
    _, s_base,  _ = mla(x, cos_l[:T],            sin_l[:T],            return_scores=True)
    _, s_shift, _ = mla(x, cos_l[shift:shift+T], sin_l[shift:shift+T], return_scores=True)
    rel_ok = torch.allclose(s_base, s_shift, atol=1e-5)
    print("scores invariant to a global position shift (relative RoPE)?:", rel_ok)
    assert rel_ok, "RoPE is not encoding RELATIVE position — score(i,j) changed under a global shift"

    print("\nOK — MLA works AND is correct: latent-compressed KV + decoupled RoPE, causal, smaller cache,\n"
          "valid attention distribution, and relative-position scores. On to step 3 (block + model).")

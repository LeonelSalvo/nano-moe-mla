"""
STEP 8 — KV-cache for MLA at inference  (feature #11)
====================================================

Why: when generating, a transformer emits ONE token at a time. With no cache it re-runs
attention over the WHOLE prefix at every step → O(T^2) work to produce T tokens. A KV-cache
stores, per past token, just what attention needs, so each new step is O(T). For MLA the
cached thing is tiny — only the compressed latent c_kv (d_latent) + the shared rope key
(d_rope) per token. That small cache IS the whole point of MLA.

Here MLA runs two ways and we PROVE they agree:
  • parallel : the full sequence at once (training / prefill),
  • cached    : tokens one by one, growing a cache, attending over it.
The last-token output must be identical both ways — that's the correctness test.

Run:  python steps/08_kv_cache.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rope_cache(dim, seq_len, base=10000.0):
    idx = torch.arange(0, dim, 2).float()
    inv = 1.0 / (base ** (idx / dim))
    t   = torch.arange(seq_len).float()
    fr  = torch.outer(t, inv)
    return torch.cos(fr), torch.sin(fr)


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class CachedMLA(nn.Module):
    def __init__(self, n_embd, n_head, head_dim, d_rope, d_latent):
        super().__init__()
        self.nh, self.hd, self.dr, self.dc = n_head, head_dim, d_rope, d_latent
        self.w_q   = nn.Linear(n_embd, n_head * (head_dim + d_rope), bias=False)
        self.w_dkv = nn.Linear(n_embd, d_latent, bias=False)     # → the cached latent
        self.w_uk  = nn.Linear(d_latent, n_head * head_dim, bias=False)
        self.w_uv  = nn.Linear(d_latent, n_head * head_dim, bias=False)
        self.w_kr  = nn.Linear(n_embd, d_rope, bias=False)       # → the cached rope key
        self.w_o   = nn.Linear(n_head * head_dim, n_embd, bias=False)

    def _project(self, x):
        """Shared front-end: queries, the latent c_kv (cached), the rope key k_r (cached)."""
        B, T, _ = x.shape
        q = self.w_q(x).view(B, T, self.nh, self.hd + self.dr).transpose(1, 2)  # (B,nh,T,hd+dr)
        c_kv = self.w_dkv(x)                                     # (B,T,dc)
        k_r  = self.w_kr(x)                                      # (B,T,dr)
        return q, c_kv, k_r

    def parallel(self, x, cos, sin):
        """Full-sequence forward (training / prefill)."""
        B, T, _ = x.shape; nh, hd, dr = self.nh, self.hd, self.dr
        q, c_kv, k_r = self._project(x)
        q_c, q_r = q[..., :hd], q[..., hd:]
        k_c = self.w_uk(c_kv).view(B, T, nh, hd).transpose(1, 2)
        v   = self.w_uv(c_kv).view(B, T, nh, hd).transpose(1, 2)
        q_r = apply_rope(q_r, cos[:T], sin[:T])
        k_r = apply_rope(k_r.view(B, 1, T, dr), cos[:T], sin[:T])
        scale = 1.0 / math.sqrt(hd + dr)
        scores = (q_c @ k_c.transpose(-2, -1) + q_r @ k_r.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        out = torch.softmax(scores, dim=-1) @ v
        return self.w_o(out.transpose(1, 2).reshape(B, T, nh * hd))

    @torch.no_grad()
    def step(self, x_t, pos, cache, cos, sin):
        """One new token x_t=(B,1,C) at position `pos`, using and GROWING the cache."""
        B = x_t.shape[0]; nh, hd, dr = self.nh, self.hd, self.dr
        q, c_kv, k_r = self._project(x_t)                       # T = 1
        # append this token's latent + rope key to the cache
        cache["ckv"] = c_kv if cache["ckv"] is None else torch.cat([cache["ckv"], c_kv], dim=1)
        cache["kr"]  = k_r  if cache["kr"]  is None else torch.cat([cache["kr"],  k_r],  dim=1)
        Tc = cache["ckv"].shape[1]                              # how many tokens cached so far
        k_c = self.w_uk(cache["ckv"]).view(B, Tc, nh, hd).transpose(1, 2)   # rebuild K/V from the latent
        v   = self.w_uv(cache["ckv"]).view(B, Tc, nh, hd).transpose(1, 2)
        q_c, q_r = q[..., :hd], q[..., hd:]
        q_r = apply_rope(q_r, cos[pos:pos + 1], sin[pos:pos + 1])
        k_r = apply_rope(cache["kr"].view(B, 1, Tc, dr), cos[:Tc], sin[:Tc])
        scale = 1.0 / math.sqrt(hd + dr)
        scores = (q_c @ k_c.transpose(-2, -1) + q_r @ k_r.transpose(-2, -1)) * scale  # (B,nh,1,Tc), all past → no mask
        out = torch.softmax(scores, dim=-1) @ v
        return self.w_o(out.transpose(1, 2).reshape(B, 1, nh * hd))


# ----------------------------- TEST (self-checking) -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, C, nh, hd, dr, dc = 1, 8, 64, 4, 16, 8, 32
    mla = CachedMLA(C, nh, hd, dr, dc).eval()
    cos, sin = build_rope_cache(dr, T)
    x = torch.randn(B, T, C)

    print("=== Step 8: KV-cache for MLA ===")
    # parallel: the last token's output computed over the whole sequence at once
    out_parallel = mla.parallel(x, cos, sin)[:, -1]
    # cached: feed tokens one by one, growing the cache
    cache, last = {"ckv": None, "kr": None}, None
    for t in range(T):
        last = mla.step(x[:, t:t + 1], t, cache, cos, sin)
    out_cached = last[:, -1]

    diff = (out_parallel - out_cached).abs().max().item()
    print(f"max |parallel - cached| = {diff:.2e}   (should be ~0 → the cache is exact)")
    assert torch.allclose(out_parallel, out_cached, atol=1e-4), "cache mismatch — bug"
    print(f"cache per token: {dc + dr} floats (latent {dc} + rope key {dr})  "
          f"vs full MHA {2 * nh * hd}")
    print("\nOK — incremental KV-cache matches the parallel forward. Generation is O(T), not O(T^2).")

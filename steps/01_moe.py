"""
STEP 1 — MoE block (Mixture-of-Experts; replaces the single dense FFN).

Zoom out: in modern-nanoGPT every block had ONE SwiGLU FFN, and all its params ran
for every token (dense). Here that FFN becomes N expert FFNs + a router that sends
each token to only its top-k experts (sparse) — plus a shared expert that's always on.
Result: huge TOTAL params, few ACTIVE params per token. This is the #1 frontier swap
(DeepSeek-V3, Qwen3-MoE, GLM, Kimi K2).

Zoom in:
  router  = Linear(C → n_experts): a score per expert per token → softmax → pick top-k.
  experts = N independent SwiGLU FFNs; a token only runs its top-k of them.
  shared  = expert(s) that run for EVERY token (capture common knowledge so the routed
            ones can specialize — the DeepSeekMoE idea).
  output  = Σ (router weight · expert(x)) over the top-k  +  shared expert(s).

Test: shape preserved, each token uses exactly top_k routed experts, and the param
count shows the sparse win (total ≫ active-per-token).

Run:  python steps/01_moe.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """One expert = a SwiGLU FFN (same as the dense block's MLP)."""
    def __init__(self, n_embd):
        super().__init__()
        hidden = 64 * ((int(2 / 3 * 4 * n_embd) + 63) // 64)
        self.w_gate = nn.Linear(n_embd, hidden, bias=False)
        self.w_up   = nn.Linear(n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoE(nn.Module):
    def __init__(self, n_embd, n_experts=8, top_k=2, n_shared=1):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.router  = nn.Linear(n_embd, n_experts, bias=False)      # score per expert
        self.experts = nn.ModuleList([Expert(n_embd) for _ in range(n_experts)])
        self.shared  = nn.ModuleList([Expert(n_embd) for _ in range(n_shared)])  # always-on

    def forward(self, x):                                            # x: (B, T, C)
        B, T, C = x.shape
        xf = x.reshape(-1, C)                                        # (N, C), N = B*T tokens

        # 1) router → probabilities over experts, then keep the top-k per token
        probs = F.softmax(self.router(xf), dim=-1)                   # (N, n_experts)
        topk_w, topk_i = probs.topk(self.top_k, dim=-1)             # (N, top_k) each
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)          # renormalize to sum 1

        # 2) routed experts: for each expert, run only the tokens that chose it
        out = torch.zeros_like(xf)
        for e in range(self.n_experts):
            sel = (topk_i == e)                                     # (N, top_k) bool
            tok = sel.any(dim=-1)                                   # (N,) tokens using expert e
            if tok.any():
                w = (topk_w * sel).sum(dim=-1)[tok]                # routing weight for those tokens
                out[tok] += w.unsqueeze(-1) * self.experts[e](xf[tok])

        # 3) shared expert(s): always on, full weight
        for sh in self.shared:
            out += sh(xf)

        return out.reshape(B, T, C)


# ----------------------------- TEST (self-checking) -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, n_embd = 2, 16, 64
    n_experts, top_k, n_shared = 8, 2, 1
    moe = MoE(n_embd, n_experts, top_k, n_shared)

    print("=== Step 1: MoE block ===")

    # (a) shape preserved
    x = torch.randn(B, T, n_embd)
    y = moe(x)
    print("input shape:", tuple(x.shape), " -> output shape:", tuple(y.shape))
    assert y.shape == x.shape, "the shape changed, something is wrong"

    # (b) each token routes to exactly top_k experts
    probs = F.softmax(moe.router(x.reshape(-1, n_embd)), dim=-1)
    _, idx = probs.topk(top_k, dim=-1)
    used = idx.unique(dim=-1).shape[-1] if idx.numel() else 0
    print(f"each token uses top_k = {idx.shape[-1]} routed experts (+ {n_shared} shared, always on)")
    assert idx.shape[-1] == top_k

    # (c) expert load: how many tokens each expert got (should be spread, not all on one)
    counts = torch.bincount(idx.flatten(), minlength=n_experts).tolist()
    print("tokens per routed expert:", counts)

    # (d) the SPARSE win: total params vs active-per-token params
    def params(m): return sum(p.numel() for p in m.parameters())
    total      = params(moe)
    per_expert = params(moe.experts[0])
    active     = params(moe.router) + (top_k + n_shared) * per_expert   # what actually runs per token
    print(f"total params: {total/1e3:.1f}K   vs active per token: {active/1e3:.1f}K   "
          f"({100*active//total}% active)")
    assert active < total, "MoE should activate only a fraction of its params per token"

    print("\nOK — MoE works: N experts, top-k routing + shared expert, sparse activation. On to step 2 (MLA).")

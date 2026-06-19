"""
STEP 11 — Muon optimizer from scratch  (feature #14)
====================================================

AdamW adapts a per-weight step size from running gradient statistics. Muon (the optimizer
Moonshot used to train Kimi K2 — the first trillion-param model trained without AdamW) does
something different for 2D WEIGHT MATRICES: it takes the momentum and then ORTHOGONALIZES it
before stepping. "Orthogonalize" here = push the update's singular values toward 1 via a few
Newton-Schulz iterations (a matrix polynomial that needs no SVD). Intuition: it spreads the
update evenly across directions instead of letting a few dominate → faster, more stable
training. Non-matrix params (norms, embeddings, biases) fall back to plain momentum/AdamW.

We implement Muon and show it drives a tiny regression loss down.

Run:  python steps/11_muon.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def zeropower_newtonschulz(G, steps=5):
    """Orthogonalize G (2D): return a matrix with ~the same column space but singular
    values ≈ 1, using `steps` Newton-Schulz iterations (no SVD). Coeffs from Keller Jordan."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + 1e-7)                       # normalize so the iteration is stable
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T                                     # work on the shorter side
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X                           # the quintic NS step
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            for p in grp["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                buf = st.get("buf")
                if buf is None:
                    buf = st["buf"] = torch.zeros_like(p)
                buf.mul_(grp["momentum"]).add_(p.grad)      # momentum
                if p.ndim == 2:                             # MATRIX → orthogonalize the update
                    upd   = zeropower_newtonschulz(buf, grp["ns_steps"])
                    scale = max(p.size(0), p.size(1)) ** 0.5   # keep update magnitude sane
                    p.add_(upd, alpha=-grp["lr"] * scale)
                else:                                       # vector/scalar → plain momentum step
                    p.add_(buf, alpha=-grp["lr"])


# ----------------------------- TEST (self-checking) -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    d = 32
    W_true = torch.randn(d, d)
    X = torch.randn(256, d)
    Y = X @ W_true                                  # the target: a linear map to recover

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(d, d, bias=False)   # 2D weights → handled by Muon's orthogonalization
            self.l2 = nn.Linear(d, d, bias=False)
        def forward(self, x):
            return self.l2(torch.relu(self.l1(x)))

    model = MLP()
    opt   = Muon(model.parameters(), lr=0.02)

    print("=== Step 11: Muon optimizer ===")
    loss0 = F.mse_loss(model(X), Y).item()
    for it in range(1, 301):
        loss = F.mse_loss(model(X), Y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if it % 100 == 0:
            print(f"  iter {it:3d}  loss {loss.item():.4f}")
    lossN = F.mse_loss(model(X), Y).item()

    print(f"loss: {loss0:.4f} → {lossN:.4f}")
    assert lossN < loss0 * 0.5, "Muon didn't reduce the loss enough"
    print("\nOK — Muon works: orthogonalized-momentum updates drive the loss down (no AdamW).")

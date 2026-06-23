"""
STEP 3 — The sparse Block + the full Model (assemble everything)
================================================================

Now I glue steps 1 (MoE) and 2 (MLA) onto the same skeleton I used in the dense
baseline (modern-nanoGPT). The skeleton never changed — only the two sub-layers did.

THE BLOCK (pre-norm + residual):
    x = x + MLA(  RMSNorm(x) )     ← sub-layer 1: attention (latent-compressed KV, step 2)
    x = x + MoE(  RMSNorm(x) )     ← sub-layer 2: the FFN, now N experts + router (step 1)
  • "pre-norm": normalize BEFORE each sub-layer (stable).
  • "residual" (the x + ...): ADD the sub-layer's output to its input, don't replace it.
    That gives the gradient a clean path backward → you can stack many blocks.

THE MODEL (MoeMlaGPT), end to end:
    token ids
      → tok_emb        each id → a vector of n_embd numbers      (NO position added here!)
      → N × SparseBlock   each = MLA (attention) + MoE (FFN)
      → norm_f         a final RMSNorm
      → lm_head        vector → one score (logit) per vocab token (TIED to tok_emb)
      → logits

WHY "NO POSITION TABLE":
  GPT-2 had TWO tables: tok_emb ("which token") AND pos_emb ("which position", added on).
  RoPE removes the position table. Position is injected differently — not by adding a
  vector at the entrance, but by ROTATING the query/key vectors by their position, and
  that rotation happens INSIDE the attention (MLA), every layer. So tok_emb only says
  "which token"; the "where" is baked into the attention comparison via RoPE.

WHY "RoPE LIVES INSIDE MLA":
  The only thing the model carries is the cos/sin "rotation table" (fixed trig, NOT learned
  params → it's a register_buffer). The model passes cos/sin down to each block, and the
  actual apply_rope() call happens inside MLA.forward, on the rope-part of q and k. So
  position enters deep inside attention, in every layer — not once at the start.

Test: forward gives logits (B,T,vocab) + loss; initial loss ≈ ln(vocab) (random guessing);
weight tying; the model-level sparse win (total params ≫ active per token); generate works.

Run:  python steps/03_block_model.py
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== reused dense piece: RMSNorm (from modern-nanoGPT) =====
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        msq = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(msq + self.eps))


def build_rope_cache(dim, seq_len, base=10000.0, device=None):
    idx      = torch.arange(0, dim, 2, device=device).float()
    inv_freq = 1.0 / (base ** (idx / dim))
    t        = torch.arange(seq_len, device=device).float()
    freqs    = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack((rx1, rx2), dim=-1).flatten(-2)


# ===== step 1: MoE =====
class Expert(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        hidden = 64 * ((int(2 / 3 * 4 * n_embd) + 63) // 64)
        self.w_gate = nn.Linear(n_embd, hidden, bias=False)
        self.w_up   = nn.Linear(n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoE(nn.Module):
    def __init__(self, n_embd, n_experts, top_k, n_shared, load_balance=True, lb_gamma=1e-3,
                 z_loss_gamma=0.0, noisy_topk=False, noise_std=1.0):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.load_balance, self.lb_gamma = load_balance, lb_gamma
        self.z_loss_gamma = z_loss_gamma                         # #3 router z-loss weight (0 = off)
        self.noisy_topk, self.noise_std = noisy_topk, noise_std  # #4 exploration noise on routing
        self.aux_z = None                                        # last batch's z-loss term (read by the model)
        self.router  = nn.Linear(n_embd, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(n_embd) for _ in range(n_experts)])
        self.shared  = nn.ModuleList([Expert(n_embd) for _ in range(n_shared)])
        # #1 the load-balancing bias: NOT a learned parameter — a running value nudged by load
        self.register_buffer("expert_bias", torch.zeros(n_experts))

    def forward(self, x):
        B, T, C = x.shape
        xf = x.reshape(-1, C)
        logits   = self.router(xf)                               # (N, n_experts): pre-softmax router scores
        affinity = F.softmax(logits, dim=-1)                     # (N, n_experts): the routing weights
        # #3 router z-loss (ST-MoE): penalize large router logits → numerical stability at scale.
        # Off by default (gamma 0). Stored as a tensor WITH grad; the model adds gamma * aux_z to the loss.
        self.aux_z = (torch.logsumexp(logits, dim=-1) ** 2).mean() if self.z_loss_gamma > 0 else None

        # #1 LOAD BALANCING (aux-loss-free, DeepSeek). The per-expert bias is added ONLY to decide
        # WHICH experts win the top-k — never to the weight used to combine outputs. So it steers
        # routing toward under-used experts without adding any loss term or distorting the math.
        sel_score = affinity + self.expert_bias if self.load_balance else affinity
        # #4 noisy top-k: jitter the SELECTION score while training so the router explores experts
        # it would otherwise never try. Noise hits selection only, never the combine weights.
        if self.noisy_topk and self.training:
            sel_score = sel_score + torch.randn_like(sel_score) * (self.noise_std / self.n_experts)
        _, topk_i = sel_score.topk(self.top_k, dim=-1)          # selection (biased toward balance)
        topk_w = torch.gather(affinity, 1, topk_i)              # combine weights = CLEAN affinity of the chosen
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(xf)
        for e in range(self.n_experts):
            sel = (topk_i == e)
            tok = sel.any(dim=-1)
            if tok.any():
                w = (topk_w * sel).sum(dim=-1)[tok]
                out[tok] += w.unsqueeze(-1) * self.experts[e](xf[tok])
        for sh in self.shared:
            out += sh(xf)

        # nudge the bias from THIS batch's load (no gradient; only while training):
        # under-used experts → bias up (picked more next time), over-used → bias down.
        if self.load_balance and self.training:
            with torch.no_grad():
                load = torch.bincount(topk_i.flatten(), minlength=self.n_experts).float()
                load = load / load.sum()
                self.expert_bias += self.lb_gamma * (1.0 / self.n_experts - load)

        return out.reshape(B, T, C)


# ===== step 2: MLA =====
class MLA(nn.Module):
    def __init__(self, n_embd, n_head, head_dim, d_rope, d_latent, qk_norm=True):
        super().__init__()
        self.nh, self.hd, self.dr, self.dc = n_head, head_dim, d_rope, d_latent
        self.w_q   = nn.Linear(n_embd, n_head * (head_dim + d_rope), bias=False)
        self.w_dkv = nn.Linear(n_embd, d_latent, bias=False)
        self.w_uk  = nn.Linear(d_latent, n_head * head_dim, bias=False)
        self.w_uv  = nn.Linear(d_latent, n_head * head_dim, bias=False)
        self.w_kr  = nn.Linear(n_embd, d_rope, bias=False)
        self.w_o   = nn.Linear(n_head * head_dim, n_embd, bias=False)
        # #5 QK-Norm: a small RMSNorm on the per-head CONTENT q and k before the dot product.
        # Keeps the attention-logit scale under control → fewer loss spikes (OLMo2/Gemma/Qwen3).
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        nh, hd, dr, dc = self.nh, self.hd, self.dr, self.dc
        q = self.w_q(x).view(B, T, nh, hd + dr).transpose(1, 2)
        q_c, q_r = q[..., :hd], q[..., hd:]
        c_kv = self.w_dkv(x)
        k_c  = self.w_uk(c_kv).view(B, T, nh, hd).transpose(1, 2)
        v    = self.w_uv(c_kv).view(B, T, nh, hd).transpose(1, 2)
        if self.qk_norm:                                        # #5: normalize content q/k per head
            q_c, k_c = self.q_norm(q_c), self.k_norm(k_c)
        k_r  = self.w_kr(x).view(B, 1, T, dr)
        q_r  = apply_rope(q_r, cos, sin)
        k_r  = apply_rope(k_r, cos, sin)
        scale  = 1.0 / math.sqrt(hd + dr)
        scores = (q_c @ k_c.transpose(-2, -1) + q_r @ k_r.transpose(-2, -1)) * scale
        future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
        out = torch.softmax(scores, dim=-1) @ v
        out = out.transpose(1, 2).reshape(B, T, nh * hd)
        return self.w_o(out)


# GQA — the dense baseline's attention (used when use_mla=False), reused from modern-nanoGPT.
# Unlike MLA, it caches full K/V per head and applies RoPE to the WHOLE head_dim (not decoupled).
class GQA(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, head_dim, qk_norm=True):
        super().__init__()
        self.nh, self.nkv, self.hd = n_head, n_kv_head, head_dim
        self.rep = n_head // n_kv_head
        self.w_q = nn.Linear(n_embd, n_head    * head_dim, bias=False)
        self.w_k = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.w_v = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.w_o = nn.Linear(n_head * head_dim, n_embd, bias=False)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.w_q(x).view(B, T, self.nh,  self.hd).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)     # RoPE on the full head_dim
        k = k.repeat_interleave(self.rep, dim=1)                    # GQA: share K/V across query heads
        v = v.repeat_interleave(self.rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, self.nh * self.hd)
        return self.w_o(out)


# ===================== what's NEW today: Block and Model =====================
@dataclass
class MoeMlaConfig:
    vocab_size: int = 65
    block_size: int = 128
    n_layer:    int = 4
    n_head:     int = 4
    head_dim:   int = 16
    n_embd:     int = 64
    d_rope:     int = 8       # MLA decoupled-rope dim
    d_latent:   int = 32      # MLA KV compression
    n_experts:  int = 8
    top_k:      int = 2
    n_shared:   int = 1
    n_kv_head:  int = 2            # GQA key/value heads (used only when use_mla=False)
    rope_base:  float = 10000.0
    # architecture toggles (for the dense-vs-sparse ablation):
    use_moe:    bool = True        # True = MoE FFN (N experts + router); False = one dense SwiGLU
    use_mla:    bool = True        # True = MLA attention; False = GQA attention (the dense baseline)
    # frontier add-ons (toggleable, so the ablation can turn each on/off):
    qk_norm:      bool  = True     # #5: RMSNorm on per-head q/k before attention (stability)
    post_norm:    bool  = True     # #9: sandwich norm — also normalize each sub-layer's OUTPUT
    load_balance: bool  = True     # #1: aux-loss-free load balancing (DeepSeek "bias trick")
    lb_gamma:     float = 1e-3      # how fast the load-balancing bias adapts to expert load
    z_loss_gamma: float = 0.0      # #3: router z-loss weight (0 = off). Keeps router logits small/stable at scale.
    noisy_topk:   bool  = False    # #4: add Gaussian noise to the router selection scores while training (exploration)
    noise_std:    float = 1.0      # std of that noise, scaled by 1/n_experts (only if noisy_topk)


class SparseBlock(nn.Module):
    """One transformer block. Same two-sub-layer shape as the dense baseline, but the
    attention is MLA (step 2) and the FFN is MoE (step 1). Each sub-layer is pre-normalized
    (RMSNorm before it) and wrapped in a residual (x + ...)."""
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)                          # norm BEFORE attention
        # attention: MLA (sparse-cache) or GQA (dense baseline), per the use_mla toggle
        self.attn  = (MLA(cfg.n_embd, cfg.n_head, cfg.head_dim, cfg.d_rope, cfg.d_latent, qk_norm=cfg.qk_norm)
                      if cfg.use_mla else
                      GQA(cfg.n_embd, cfg.n_head, cfg.n_kv_head, cfg.head_dim, qk_norm=cfg.qk_norm))
        self.norm2 = RMSNorm(cfg.n_embd)                          # norm BEFORE the FFN
        # FFN: MoE (sparse experts) or a single dense SwiGLU (the dense baseline), per use_moe
        self.moe   = (MoE(cfg.n_embd, cfg.n_experts, cfg.top_k, cfg.n_shared,
                          load_balance=cfg.load_balance, lb_gamma=cfg.lb_gamma,
                          z_loss_gamma=cfg.z_loss_gamma,
                          noisy_topk=cfg.noisy_topk, noise_std=cfg.noise_std)
                      if cfg.use_moe else
                      Expert(cfg.n_embd))                          # Expert == a SwiGLU FFN
        # #9 sandwich norm: also RMSNorm each sub-layer's OUTPUT (before the residual add), not
        # just its input. Pre-norm = "tidy the input"; sandwich = "tidy input AND output" — extra
        # stability (Gemma 2 / OLMo 2). nn.Identity() = a no-op when the flag is off.
        self.post1 = RMSNorm(cfg.n_embd) if cfg.post_norm else nn.Identity()
        self.post2 = RMSNorm(cfg.n_embd) if cfg.post_norm else nn.Identity()

    def forward(self, x, cos, sin):
        # sub-layer 1 — attention: tokens "talk to each other" (RoPE enters here, in MLA).
        # pre-norm on the input, #9 post-norm on the output, then the residual add.
        x = x + self.post1(self.attn(self.norm1(x), cos, sin))
        # sub-layer 2 — MoE FFN: each token "thinks" through its top-k experts (+ the shared one).
        x = x + self.post2(self.moe(self.norm2(x)))
        return x


class MoeMlaGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # token embedding: turns each token id into a vector of n_embd numbers.
        # There is NO positional embedding table — position is handled by RoPE inside MLA.
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # the stack of N sparse blocks (the depth of the model)
        self.blocks  = nn.ModuleList([SparseBlock(cfg) for _ in range(cfg.n_layer)])
        self.norm_f  = RMSNorm(cfg.n_embd)                       # final norm before the head
        # lm_head: projects each token's vector to a score (logit) per vocab token
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # WEIGHT TYING: the input table (text→vector) and the output matrix (vector→logits)
        # are the SAME tensor — one shared "dictionary" for reading and writing. Saves params.
        self.lm_head.weight = self.tok_emb.weight
        # the RoPE "rotation table" (cos/sin for the rope-part dimension). It's fixed trig,
        # NOT learned → register_buffer (travels with the model but isn't a parameter).
        # MLA rotates only the small rope-part (d_rope); GQA rotates the full head_dim
        rope_dim = cfg.d_rope if cfg.use_mla else cfg.head_dim
        cos, sin = build_rope_cache(rope_dim, cfg.block_size, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)                                   # GPT-style weight init (std 0.02)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        # tok_emb and lm_head share the tied matrix → count it once
        return sum(p.numel() for p in self.parameters()) - self.lm_head.weight.numel()

    def active_params_per_token(self):
        """The sparse win: how many params actually RUN for one token. It's everything
        except the experts the router did NOT pick (top_k of n_experts run; the rest idle)."""
        cfg = self.cfg
        total = self.num_params()
        if not cfg.use_moe:                                       # dense FFN → everything runs
            return total
        per_expert = sum(p.numel() for p in self.blocks[0].moe.experts[0].parameters())
        idle = (cfg.n_experts - cfg.top_k) * per_expert * cfg.n_layer   # skipped experts × layers
        return total - idle

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)                                    # (B, T, n_embd) — tokens, NO position yet
        # hand the rotation table (sliced to this sequence length) down to every block;
        # the actual rotation by position happens inside MLA.forward.
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for block in self.blocks:                                # pass through the N sparse blocks
            x = block(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)                                 # (B, T, vocab): a score per token
        loss = None
        if targets is not None:
            # cross-entropy: how far the predicted distribution is from the true next token
            loss = F.cross_entropy(logits.reshape(B * T, -1), targets.reshape(B * T))
            # #3 add the router z-loss from every MoE block (only if enabled; gamma 0 → no-op).
            if self.cfg.use_moe and self.cfg.z_loss_gamma > 0:
                aux_z = sum(b.moe.aux_z for b in self.blocks if b.moe.aux_z is not None)
                if not isinstance(aux_z, float):                 # at least one block contributed
                    loss = loss + self.cfg.z_loss_gamma * aux_z
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx


# ----------------------------- TEST (self-checking) -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = MoeMlaConfig(vocab_size=65, block_size=32, n_layer=4)
    model = MoeMlaGPT(cfg)

    print("=== Step 3: sparse Block + MoeMlaGPT ===")
    print(f"flags: qk_norm={cfg.qk_norm}  post_norm={cfg.post_norm}  load_balance={cfg.load_balance}  (toggleable for the ablation)")
    print(f"total params: {model.num_params()/1e3:.1f}K   "
          f"active/token: {model.active_params_per_token()/1e3:.1f}K   "
          f"({100*model.active_params_per_token()//model.num_params()}% active)")

    # (a) forward → logits + loss
    B, T = 2, 32
    idx     = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = model(idx, targets)
    print("logits shape:", tuple(logits.shape), " (expected", (B, T, cfg.vocab_size), ")")
    assert logits.shape == (B, T, cfg.vocab_size)

    # (b) initial loss ≈ ln(vocab)
    expected = math.log(cfg.vocab_size)
    print(f"initial loss: {loss.item():.3f}   ln(vocab) = {expected:.3f}")
    assert abs(loss.item() - expected) < 0.6, "initial loss looks off"

    # (c) weight tying
    print("tok_emb and lm_head share the matrix?:", model.tok_emb.weight is model.lm_head.weight)
    assert model.tok_emb.weight is model.lm_head.weight

    # (d) generate
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=20)
    print("generate:", out.shape[1], "tokens (1 + 20 new)")
    assert out.shape == (1, 21)

    print("\nOK — the sparse model runs end to end (MoE + MLA). On to step 4 (train + dense-vs-sparse).")

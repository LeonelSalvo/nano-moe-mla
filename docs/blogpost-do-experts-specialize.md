# Do MoE experts actually specialize? Measuring it from scratch (MoE + MLA, PyTorch)

*Draft for dev.to / Substack — links back to [nano-moe-mla](https://github.com/LeonelSalvo/nano-moe-mla).*

Mixture-of-Experts is the default for current open LLMs (DeepSeek-V3, Kimi K2, Qwen3-MoE). The usual
story is intuitive: a router sends each token to a few specialist experts, so "one expert handles code,
another handles prose." But is that story *true*? Recent papers disagree — and the disagreement is the
interesting part.

So I built the smallest honest testbed I could to look at it directly: a from-scratch **MoE + MLA**
nano-model in pure PyTorch, with the measurement tooling to ask "do experts specialize?" and a clean
ablation to ask "what does each piece actually buy?". No `nn.Transformer`, no HuggingFace — every piece
written out and self-checked.

## The setup

- A trainable model that combines the two components that define current sparse LLMs: **MoE** (sparse
  expert FFNs + top-k router with a shared expert) and **MLA** (Multi-head Latent Attention — the KV
  cache compressed to a latent vector, with decoupled RoPE).
- A **labeled 3-domain corpus**: English drama · Python code · Spanish prose.
- A **router-specialization probe**: how strongly does the router send each domain to different experts?
  Measured with mutual information `I(domain ; expert)`.
- A **2×2 factorial ablation**: dense / +MoE / +MLA / both — to separate each component's effect.

## Finding 1 — the two pieces do different jobs

| variant | attention | FFN | val loss | total | active/tok | KV/tok |
|---|---|---|---|---|---|---|
| dense | GQA | SwiGLU | 1.665 | 198K | 198K | 64 |
| +MoE | GQA | MoE | **1.496** | 1380K | 495K | 64 |
| +MLA | MLA | SwiGLU | 1.670 | 216K | 216K | **40** |
| both | MLA | MoE | 1.602 | 1398K | 513K | **40** |

- **MoE is the quality lever** — the biggest val-loss drop, by adding capacity (huge total params, only
  ~36% active per token).
- **MLA is the memory lever** — on its own it's roughly neutral on loss but shrinks the KV cache to
  **40 vs 64 floats/token (−37%)**. Its win is the cache, not the loss.

This is the expected, honest result: the two components are *orthogonal* — one buys quality, the other
buys cheap context.

## Finding 2 — the balancing ↔ specialization tradeoff

The router probe measures `I(domain ; expert)` with load balancing on vs off:

| load balancing | I(domain ; expert) |
|---|---|
| **OFF** | **0.075 bits** — more specialization, but risks expert collapse |
| **ON**  | **0.072 bits** — experts evenly used, but flatter routing |

Load balancing buys stability at a small cost in specialization. **Honest caveat:** at nano scale the gap
is small and noisy, and MI stays far below the 1.585-bit ceiling. The *direction* is reproducible; the
magnitude is not a publishable number. The point is the mechanism and the measurement, not the digits.

## Where this sits in the research conversation

Whether experts specialize *semantically* is an open question right now:

- **Mixtral** (arXiv:2401.04088) and **POS-routing** (arXiv:2412.16971) find routing is **token /
  syntactic** — driven by surface features like indentation and repeated tokens, *not* topic.
- **Intel — "Probing Semantic Routing"** (arXiv:2502.10928) finds **clear evidence of semantic routing**
  in large (>100B) MoE models.
- A **"MoE Routing Testbed"** (arXiv:2604.07030) studies the same specialization question at small scale.

So the field genuinely hasn't settled it. A nano-scale probe doesn't resolve the debate — but it lets you
*see the question* on hardware you own, and it sits in the same conversation as the papers above.

## Reproduce it

```bash
git clone https://github.com/LeonelSalvo/nano-moe-mla
cd nano-moe-mla && pip install -r requirements.txt
bash run_all.sh      # every step + regenerates the figures (~15 min on one GPU)
```

Each component is a self-checking script (`steps/01_moe.py` … `09_stack_ablation.py`). The MLA step now also
verifies correctness, not just that it runs: the attention weights form a valid distribution, and the
decoupled RoPE encodes *relative* position (scores are invariant to a global position shift).

The value isn't the val loss — it's being able to implement the sparse template down to the tensor and
then *measure* what each part contributes. If you've read about MoE and MLA but never watched a router
specialize (or fail to), this is a place to see it.

---

*Built from scratch in the spirit of Karpathy's nanoGPT. Dense baseline:
[modern-nanoGPT](https://github.com/LeonelSalvo/modern-nanoGPT).*

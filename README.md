<div align="center">

# nano-moe-mla

**A from-scratch PyTorch implementation of the two components that define current sparse open LLMs — Mixture-of-Experts (`MoE`) and Multi-head Latent Attention (`MLA`) — in a single trainable nano-scale model, with tooling to measure what each one does.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Overview

[`modern-nanoGPT`](https://github.com/LeonelSalvo/modern-nanoGPT) is a **dense** Llama-style transformer — the backbone shared by most open LLMs. Current sparse open models (DeepSeek-V3, Kimi K2, GLM, Qwen3-MoE) converge on one template:

> **MoE** (sparse experts instead of one dense FFN) **+ MLA** (latent-compressed attention instead of plain MHA/GQA).

This repo implements both from scratch in the same trainable model, and includes the measurement tooling that makes the result legible: a labeled multi-domain corpus, a router-specialization probe, and a 2×2 feature ablation.

**How this differs from related from-scratch work.** [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) isolates MLA; [cameronrwolfe/nanoMoE](https://github.com/cameronrwolfe/nanoMoE) isolates MoE (with classic auxiliary-loss balancing). Here MoE and MLA live in **one model**, with a factorial dense / +MoE / +MLA / both ablation that separates their effects, plus a router-specialization probe over a labeled corpus.

## Dense → sparse: the two swaps

| dense (modern-nanoGPT) | sparse (this repo) | what changes |
|---|---|---|
| one SwiGLU FFN per block | **MoE** — N expert FFNs + a top-k router (+ a shared expert) | huge total params, few **active** per token |
| GQA attention | **MLA** (Multi-head Latent Attention) | KV cache compressed to a latent vector, with decoupled RoPE |

Everything else (RMSNorm, RoPE, pre-norm + residual, tied/no-bias) is reused from the dense baseline.

## Optional features (toggleable flags)

On top of MoE + MLA, a few small recent features, each a flag on `MoeMlaConfig` so the ablation can turn them on/off — full record (applied / queued / dropped + why) in [`FEATURES.md`](FEATURES.md):

- **#1 aux-loss-free load balancing** (DeepSeek bias trick) — keeps experts evenly used without an extra loss.
- **#5 QK-Norm** — RMSNorm on q/k before attention (stability; replaced Gemma's logit soft-capping).
- **#9 sandwich norm** — normalize each sub-layer's input *and* output (Gemma 2 / OLMo 2).

## Build it step by step

Same approach as modern-nanoGPT: one piece at a time, each a self-checking script.

```
steps/
├── 01_moe.py            # MoE block: router (top-k) + experts + shared expert
├── 02_mla.py            # Multi-head Latent Attention: KV compression + decoupled RoPE
├── 03_block_model.py    # assemble the sparse block + full model (+ the toggleable flags)
├── 04_train.py          # train + the sparse story in numbers (active params, KV cache)
├── 05_multidomain.py    # a LABELED multi-domain corpus (drama / code / Spanish)
├── 06_routing_probe.py  # do experts specialize? domain→expert heatmap + the balancing tradeoff
├── 07_ablation.py       # isolate each piece: dense / +MoE / +MLA / both
├── 08_kv_cache.py       # real KV-cache for MLA → O(T) generation (cached == parallel)
└── 09_stack_ablation.py # measure each architecture/routing technique ON vs OFF (val CE + MI matrix)
```

> Three cross-cutting techniques (the **Muon** optimizer, **Multi-Token Prediction**, and a
> **from-scratch BPE** tokenizer) live in the companion repo
> [`frontier-llm-techniques-2026-Q1`](https://github.com/LeonelSalvo/frontier-llm-techniques-2026-Q1).

```bash
python -m venv .venv && source .venv/bin/activate   # on Debian/Ubuntu use python3
pip install -r requirements.txt
bash run_all.sh            # runs every step + regenerates the result images (~15 min)
# or one at a time: python steps/01_moe.py  → 02 → 03 → ...
```

## Results

Trained from scratch on a single RTX 3090.

**The model (char-level, TinyShakespeare).** ~1.4M params, **36% active per token** (MoE), and MLA caches **40 floats/token vs 128 for full attention** (31%). Best val loss ≈ **1.57** (with QK-Norm + sandwich-norm + load balancing on) — close to the dense baseline (modern-nanoGPT, val ≈ 1.48 at ~9M params), which is the honest, expected result: at nano scale a small sparse model trades a little quality for the structural wins; the point is the **mechanism**, not beating dense.

![loss curve](loss_curve.png)

### Finding 1 — the balancing ↔ specialization tradeoff

On a labeled 3-domain corpus (English drama · Python code · Spanish prose), the probe measures how strongly the router sends each domain to different experts, via the mutual information `I(domain ; expert)`:

| load balancing | I(domain ; expert) |
|---|---|
| **OFF** | **0.075 bits** — more specialization, but risks expert collapse |
| **ON** | **0.072 bits** — experts evenly used, but flatter routing |

Across runs, **OFF consistently shows equal-or-higher specialization than ON** — load balancing buys stability at the cost of specialization. At nano scale the gap is small and a bit noisy (MI stays far below the 1.585-bit ceiling), but the direction is reproducible. Heatmaps: `routing_heatmap_lb-on.png`, `routing_heatmap_lb-off.png`.

### Finding 2 — ablation: what each piece buys

Four variants, same multi-domain data, same steps (`python steps/07_ablation.py`):

| variant | attention | FFN | val loss | total | active/tok | KV/tok |
|---|---|---|---|---|---|---|
| dense | GQA | SwiGLU | 1.665 | 198K | 198K | 64 |
| +MoE | GQA | MoE | **1.496** | 1380K | 495K | 64 |
| +MLA | MLA | SwiGLU | 1.670 | 216K | 216K | **40** |
| both | MLA | MoE | 1.602 | 1398K | 513K | **40** |

![ablation](ablation.png)

The two pieces do **different jobs**, and the ablation isolates each cleanly:

- **MoE is the quality lever** — biggest val-loss drop (1.674 → 1.496) by adding capacity (huge total params, only ~36% active per token).
- **MLA is the memory lever** — on its own it's ~neutral on loss (1.674 → 1.691) but shrinks the KV cache to **40 vs 64 floats/token (−37%)**. Its win is the cache, not the loss.
- **both** = the full model: keeps most of MoE's quality gain *and* MLA's smaller cache, with a small quality give-back from compressing attention — the expected tradeoff.

<sub>At nano scale these gaps are modest, but the directions are clean and the structural numbers (active params, KV cache) hold at any scale.</sub>

### Finding 3 — micro scale, BPE, seed-averaged (the honest version)

The nano findings above are single-seed and char-level, so the MI is noisy. Scaling to a **balanced
BPE corpus** (English / real Python / Spanish, ~5M chars each, vocab 16k), a **micro** model
(6 layers · 384 dim · 16 experts), and **averaging 3 seeds** turns the noise into error bars you can
read. The rule: a gap is real **only if the `± std` bars don't overlap**. Architecture/routing matrix
(`python steps/09_stack_ablation.py`, `TOKENIZER=bpe SCALE=micro SEEDS=3`), vs `BASE` = CE 4.898±0.020,
MI 0.117±0.016:

| flip vs BASE | val CE | MI (domain;expert) | verdict |
|---|---|---|---|
| **− load-balancing** | 4.855±0.022 | **0.158±0.021** | **MI up, bars clear** → the tradeoff, confirmed |
| **top_k=1** | 4.931±0.022 | **0.025±0.002** | **MI craters** → top-1 kills domain specialization |
| − QK-Norm | 4.906±0.011 | 0.092±0.004 | small MI drop (marginal) |
| + z-loss / − sandwich / + noisy top-k | ≈ BASE | ≈ BASE | **within noise** — no claim at this budget |

The headline: with error bars, the **balancing ↔ specialization tradeoff (Finding 1) holds** — removing
load balancing raises MI beyond the noise — and **top_k=1 collapses** domain specialization. The other
knobs are *within noise* at this budget, which is itself an honest result, not a failure. (1500 iters is
underfit, so MoE's quality edge from Finding 2 is muted here; a longer run sharpens the marginals.)

<sub>Run <code>TOKENIZER=bpe SCALE=micro SEEDS=3 python steps/09_stack_ablation.py</code> to reproduce: it
saves bar charts with ± std, a routing heatmap per setting, and the BASE checkpoint under
<code>results/&lt;run&gt;/</code>.</sub>

> On method: the mutual-information probe is one clean informational angle on specialization. The MoE literature more commonly measures it by routing distribution per category, load, ablation, or predictive probing — MI here is a complement, not the field-standard method. Reporting **mean ± std over seeds** is what separates a real effect from noise.

## Scope

This is an educational and reference implementation, not a model intended for deployment. It has two goals: implement the sparse template (MoE + MLA) from scratch, down to the tensor, and provide the instruments to study it — a labeled multi-domain corpus, a router-specialization probe, and a feature ablation. At nano scale the validation loss is not the objective; the value is understanding each component in full and being able to measure what each one contributes (see the two findings above).

## How to scale beyond nano

Three directions, roughly by effort:

**Make it bigger (stop being nano).** Use the BPE data pipeline (`data_prep.py` + `bpe_data.py`, `TOKENIZER=bpe`) on a balanced multi-domain corpus and bump `n_layer / n_embd / block_size`. Add bf16 + `torch.compile`, gradient accumulation, and a **batched MoE kernel** — the Python expert loop is the real bottleneck, and a Triton fused kernel is the natural next exercise. (Cross-cutting training tricks — the Muon optimizer and Multi-Token Prediction — live in the companion repo and can be wired in from there.)

**Push the architecture toward the real frontier.** DeepSeek Sparse Attention (DSA) or a linear-attention / gated-DeltaNet hybrid (Qwen3-Next) for cheap long context; a Mamba/SSM block for a transformer-vs-SSM head-to-head; fine-grained experts, expert-parallel routing, capacity factors.

**Keep measuring.** Re-run the routing probe at larger scale or with `top_k=1` and see if specialization (MI) rises. Ablate the *stabilizers* (`qk_norm`, `post_norm`, `load_balance`) on val loss, not just the big pieces. Measure the KV-cache memory and tokens/sec gap (MLA vs GQA vs MHA) **as context grows** — that's where MLA actually pays off. And an SFT step (→ DPO/GRPO) turns the base model into a chat-y one (the nanochat direction).

## Credits

Architecture from the DeepSeek-V2/V3 papers (MLA + DeepSeekMoE) and the Shazeer/Switch line of MoE work; built from scratch in the spirit of Karpathy's nanoGPT. Dense baseline: [modern-nanoGPT](https://github.com/LeonelSalvo/modern-nanoGPT).

## License

MIT — see [LICENSE](LICENSE). Built by [Leonel Salvo](https://github.com/LeonelSalvo).

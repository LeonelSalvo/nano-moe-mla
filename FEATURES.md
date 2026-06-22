# nano-moe-mla — feature registry

Record of which 2026-frontier features are in the model, queued, or deliberately left out.
Applied features are **toggleable flags** on `MoeMlaConfig` (steps/03_block_model.py) so the
ablation can turn each on/off.

## ✅ Applied (flags in MoeMlaConfig)

| # | feature | flag | what it does |
|---|---|---|---|
| 1 | Aux-loss-free load balancing (DeepSeek "bias trick") | `load_balance` | per-expert bias steers top-k toward under-used experts, no extra loss |
| 5 | QK-Norm | `qk_norm` | RMSNorm on per-head content q/k before the dot product (stability) |
| 9 | Sandwich / post-norm (Gemma 2 / OLMo 2) | `post_norm` | also normalize each sub-layer's output, not just its input |
| 7 | Top-1 routing (Switch-style) | `top_k=1` | already a config knob; with top-1, load balancing matters more |

## ✅ Demonstrated as standalone steps (from scratch, each with a self-checking test)

| # | feature | step | what it shows |
|---|---|---|---|
| 11 | Real KV-cache at inference for MLA | `steps/08_kv_cache.py` | incremental cache matches the parallel forward → O(T) generation |
| 16 | BPE tokenizer | `steps/09_bpe.py` | learn merges, exact round-trip, shorter sequences than char-level |
| 13 | Multi-Token Prediction (MTP) | `steps/10_mtp.py` | a 2nd head predicts t+2; both losses drop together |
| 14 | Muon optimizer | `steps/11_muon.py` | orthogonalized-momentum (Newton-Schulz) drives loss down, no AdamW |

<sub>These are demonstrated in isolation to keep the main ablation model (steps 03–07) stable and char-level for a clean comparison. Wiring them into the trained model is the natural next extension.</sub>

## 🔜 Queued

| # | feature | note |
|---|---|---|
| 3 | Router z-loss | regularize router logits; needs aux-loss plumbing into the train loop |
| 4 | Noisy top-k routing | Gaussian noise on router scores while training (exploration) |

## ❌ Dropped (with reason)

| # | feature | why not |
|---|---|---|
| 2 | Aux-loss load balancing | superseded by #1 (the modern, loss-free version). Lives as an ablation row, not a 2nd mechanism |
| 8 | Logit soft-capping (Gemma 2) | replaced by #5 (QK-Norm) — Gemma 3 dropped it. Redundant stabilizer |
| 17 | RoPE scaling (YaRN/NTK) | invisible at block_size=128; only pays off with long context |
| 18 | bf16 + torch.compile | clashes with the dynamic-shape Python MoE loop; needs the batched MoE first |
| 20 | Triton fused MoE kernel | you chose to skip; revisit when optimizing the MoE for real |
| 21–26 | DSA, linear/DeltaNet hybrid, Mamba layer, DDP, FP8, post-training (SFT/RLHF/GRPO) | need scale / multi-GPU / a separate phase — out of nano scope |

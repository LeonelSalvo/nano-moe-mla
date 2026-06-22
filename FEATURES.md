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
| 3 | Router z-loss (ST-MoE) | `z_loss_gamma` | penalizes large router logits → numerical stability at scale. Opt-in (0 = off) |
| 4 | Noisy top-k routing | `noisy_topk` / `noise_std` | jitter the selection score while training so the router explores. Opt-in |
| 13 | Multi-Token Prediction (MTP) | `mtp` / `mtp_weight` | a 2nd head predicts t+2 (denser training signal). Opt-in; built into the model |

## ✅ Demonstrated as standalone steps (from scratch, each with a self-checking test)

| # | feature | step | what it shows |
|---|---|---|---|
| 11 | Real KV-cache at inference for MLA | `steps/08_kv_cache.py` | incremental cache matches the parallel forward → O(T) generation |
| 16 | BPE tokenizer | `steps/09_bpe.py` | learn merges, exact round-trip, shorter sequences than char-level |
| 13 | Multi-Token Prediction (MTP) | `steps/10_mtp.py` | a 2nd head predicts t+2; both losses drop together |
| 14 | Muon optimizer | `steps/11_muon.py` | orthogonalized-momentum (Newton-Schulz) drives loss down, no AdamW. **Also wireable into training:** `USE_MUON=1 python steps/04_train.py` (Muon for hidden matrices + AdamW for embeddings/head/norms) |

<sub>KV-cache, BPE and MTP are demonstrated in isolation to keep the main ablation model (steps 03–07) stable and char-level for a clean comparison. Wiring BPE + MTP into the trained model is the next extension (see the wiki escalado note).</sub>

## 🧪 Stack ablation

`steps/12_stack_ablation.py` trains the model flipping ONE technique at a time and prints a matrix
of **val CE + MI** per setting (MoE, MLA, load-balancing, z-loss, QK-Norm, sandwich-norm, noisy
top-k, top_k=1, MTP, AdamW-vs-Muon). `SCALE=nano` (fast smoke) or `SCALE=micro` (24 GB GPU).

## 🧰 Real-metrics data pipeline (BPE + balanced multi-domain)

For meaningful numbers (not char-level memorization), `data_prep.py` downloads a **bigger, balanced**
3-domain corpus (English / real Python from CPython / Spanish, capped to equal size) and `bpe_data.py`
tokenizes it with BPE. Run the ablation on it with `TOKENIZER=bpe`.

> **Abstraction receipt:** the from-scratch BPE is `steps/09_bpe.py` (that's the mechanism). The data
> pipeline uses Hugging Face `tokenizers` (the same byte-level BPE, in Rust) because the naive
> O(merges×corpus) Python version would take hours on tens of MB. From-scratch to learn it; the fast
> version to use it — exactly the "name what the wrapper wraps" rule.

## ❌ Dropped (with reason)

| # | feature | why not |
|---|---|---|
| 2 | Aux-loss load balancing | superseded by #1 (the modern, loss-free version). Lives as an ablation row, not a 2nd mechanism |
| 8 | Logit soft-capping (Gemma 2) | replaced by #5 (QK-Norm) — Gemma 3 dropped it. Redundant stabilizer |
| 17 | RoPE scaling (YaRN/NTK) | invisible at block_size=128; only pays off with long context |
| 18 | bf16 + torch.compile | clashes with the dynamic-shape Python MoE loop; needs the batched MoE first |
| 20 | Triton fused MoE kernel | you chose to skip; revisit when optimizing the MoE for real |
| 21–26 | DSA, linear/DeltaNet hybrid, Mamba layer, DDP, FP8, post-training (SFT/RLHF/GRPO) | need scale / multi-GPU / a separate phase — out of nano scope |

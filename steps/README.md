# Building nano-moe-mla step by step

Same idea as modern-nanoGPT: one component at a time, each a self-checking script
(zoom out → zoom in → implementation → test). Here the two new pieces are the
sparse/frontier swaps. RMSNorm, RoPE, pre-norm + residual come from the dense baseline.

## Roadmap

**Build the sparse model**
1. **MoE block** — router (top-k) + N expert FFNs + a shared expert. The dense FFN goes sparse.
2. **MLA** — Multi-head Latent Attention: compress the KV cache into a latent + decoupled RoPE.
3. **Block + model** — assemble RMSNorm + MLA + MoE into the sparse block + model (+ toggleable flags).

**Train + measure**
4. **Train** — train, and put the numbers next to the dense baseline (params, KV, loss).
5. **Multi-domain corpus** — a labeled mix (drama / code / Spanish) so experts have domains to split.
6. **Routing probe** — does the router specialize? domain→expert heatmap + the balancing tradeoff.
7. **Ablation** — isolate each piece on val loss: dense / +MoE / +MLA / both.

**Frontier feature, demonstrated from scratch**
8. **KV-cache for MLA** — incremental generation; cached output == parallel (O(T), not O(T²)).

**Measure the whole stack**
9. **Stack ablation** — train flipping ONE architecture/routing technique at a time; print a matrix of
    val CE + MI per setting (MoE, MLA, load-balancing, z-loss, QK-Norm, sandwich-norm, noisy top-k,
    top_k=1). `SCALE=nano` smoke test / `TOKENIZER=bpe SCALE=micro SEEDS=3` for the real measurement.
    **Every run is saved** under `results/<run-tag>/`: `metrics.csv` + `metrics.json`, `config.json`,
    bar charts with error bars (`val_ce.png`, `mi.png`), a routing heatmap per MoE setting, and the
    BASE model checkpoint — so a long run is never lost. (`.pt` is gitignored; the rest is kept.)

> The model exposes opt-in MoE/routing flags wired in for the ablation: router z-loss (`z_loss_gamma`)
> and noisy top-k (`noisy_topk`). Defaults are unchanged, so the verified char-level ablation reproduces.
>
> Three cross-cutting techniques (Muon optimizer, MTP, from-scratch BPE) were factored out into the
> companion repo **`frontier-llm-techniques-2026-Q1`**.

```bash
python steps/01_moe.py     # does it print OK? → on to step 2
python steps/02_mla.py
...
# or: bash run_all.sh   (runs every step + regenerates the result images)
```

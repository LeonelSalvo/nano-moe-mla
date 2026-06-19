# Building nanoMoLa step by step

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

**Frontier features, each demonstrated from scratch**
8. **KV-cache for MLA** — incremental generation; cached output == parallel (O(T), not O(T²)).
9. **BPE tokenizer** — learn merges, exact round-trip, shorter sequences than char-level.
10. **MTP** — multi-token prediction: a 2nd head predicts t+2; both losses drop.
11. **Muon** — the orthogonalized-momentum optimizer (Newton-Schulz), from scratch.

```bash
python steps/01_moe.py     # does it print OK? → on to step 2
python steps/02_mla.py
...
# or: bash run_all.sh   (runs every step + regenerates the result images)
```

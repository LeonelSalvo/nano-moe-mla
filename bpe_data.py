"""
bpe_data.py — a BPE-tokenized multi-domain dataset (the real-metrics upgrade to step 5).

Step 5 (CharMultiDomain) is char-level: tiny vocab, long sequences, fine for the mechanism demo.
For meaningful numbers we tokenize with BPE. The from-scratch BPE (the "how it works" receipt) lives
in the companion repo `frontier-llm-techniques-2026-Q1` (bpe.py); here we use Hugging Face `tokenizers`
(the same byte-level BPE, in Rust) because tokenizing tens of MB with a naive Python BPE would take hours.

Drop-in for CharMultiDomain: same interface (.names, .vocab_size, .get_batch, .decode), so the
routing probe / step-12 ablation work unchanged. It reads every data/domains/*.txt as a domain,
trains (or loads) a byte-level BPE over all of them, and caches the tokenized ids to data/bpe/.

  BPE_VOCAB=16000   (env) — vocabulary size.
"""

import os
import glob
import random
import numpy as np
import torch

HERE  = os.path.dirname(os.path.abspath(__file__))
DOM   = os.path.join(HERE, "data", "corpus")          # written by data_prep.py (balanced, big)
CACHE = os.path.join(HERE, "data", "bpe")
VOCAB = int(os.environ.get("BPE_VOCAB", "16000"))


class BpeMultiDomain:
    def __init__(self, block_size, device, split=0.9):
        from tokenizers import ByteLevelBPETokenizer            # imported lazily (only the BPE path needs it)
        files = sorted(glob.glob(os.path.join(DOM, "*.txt")))
        if not files:
            raise FileNotFoundError(f"no corpus files in {DOM} — run `python data_prep.py` first")
        self.names = [os.path.splitext(os.path.basename(f))[0] for f in files]
        os.makedirs(CACHE, exist_ok=True)

        # tokenizer: load the cached one, else train a byte-level BPE over all domains
        vj, mt = os.path.join(CACHE, "vocab.json"), os.path.join(CACHE, "merges.txt")
        if os.path.exists(vj) and os.path.exists(mt):
            self.tok = ByteLevelBPETokenizer(vj, mt)
        else:
            print(f"[bpe] training byte-level BPE (vocab={VOCAB}) over {len(files)} domains...")
            self.tok = ByteLevelBPETokenizer()
            self.tok.train(files=files, vocab_size=VOCAB, special_tokens=[])
            self.tok.save_model(CACHE)
        self.vocab_size = self.tok.get_vocab_size()

        # tokenize each domain (cache the ids), split train/val per domain
        self.train, self.val = {}, {}
        for path, name in zip(files, self.names):
            ids = self._encode_cached(path, name)
            n = int(len(ids) * split)
            self.train[name] = torch.tensor(ids[:n], dtype=torch.long)
            self.val[name]   = torch.tensor(ids[n:], dtype=torch.long)
        self.block_size, self.device = block_size, device

    def _encode_cached(self, path, name):
        npy = os.path.join(CACHE, f"{name}.npy")
        if os.path.exists(npy):
            return np.load(npy)
        text = open(path, encoding="utf-8", errors="ignore").read()
        ids, step = [], 1_000_000                                # chunk so a single encode isn't huge
        for i in range(0, len(text), step):
            ids.extend(self.tok.encode(text[i:i + step]).ids)
        arr = np.array(ids, dtype=np.int32)
        np.save(npy, arr)
        print(f"[bpe] {name}: {len(arr)/1e6:.2f}M tokens (cached)")
        return arr

    def get_batch(self, split, batch_size, domain=None):
        store = self.train if split == "train" else self.val
        picks = [domain or random.choice(self.names) for _ in range(batch_size)]
        xs, ys, doms = [], [], []
        for nm in picks:
            d = store[nm]
            i = torch.randint(len(d) - self.block_size - 1, (1,)).item()
            xs.append(d[i:i + self.block_size])
            ys.append(d[i + 1:i + 1 + self.block_size])
            doms.append(self.names.index(nm))
        x   = torch.stack(xs).to(self.device)
        y   = torch.stack(ys).to(self.device)
        dom = torch.tensor(doms, device=self.device)
        return x, y, dom

    def decode(self, t):
        return self.tok.decode([int(i) for i in t])


# ----------------------------- self-check -----------------------------
if __name__ == "__main__":
    data = BpeMultiDomain(block_size=128, device="cpu")
    print("=== bpe_data ===")
    print("domains:", data.names, " vocab:", data.vocab_size)
    for n in data.names:
        print(f"  {n:10s} train tok: {len(data.train[n]):>9d}   val tok: {len(data.val[n]):>8d}")
    x, y, dom = data.get_batch("train", 4)
    assert x.shape == y.shape and dom.shape[0] == x.shape[0]
    print("sample decode:", repr(data.decode(x[0][:24].tolist())[:60]))
    print("OK — BPE multi-domain dataset ready.")

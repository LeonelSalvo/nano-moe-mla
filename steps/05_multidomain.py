"""
STEP 5 — A multi-domain, LABELED dataset (so we can measure expert specialization)
==================================================================================

Why this step exists: on a single-domain corpus (just Shakespeare) the MoE experts have
nothing to split up — there are no sub-domains. To SEE whether the router learns to send
different kinds of text to different experts, we need a corpus made of clearly DIFFERENT
domains, AND we need to know each token's domain (the label) so we can later cross-tabulate
"domain → which expert".

Here we build a char-level corpus from 3 visibly different domains:
  • shakespeare — English drama (verse, character names)
  • code        — Python source (indentation, symbols, keywords)
  • spanish     — Spanish prose (different language → strong, easy-to-route signal)

We build ONE shared character vocabulary over all three (so the same token ids mean the
same thing), keep each domain's train/val separate, and serve batches that ALSO return a
domain label per sequence. Step 6 uses those labels to build the domain × expert heatmap.

Get the real data (run on your machine; small slices are enough):
  mkdir -p data/domains
  cp data/input.txt data/domains/shakespeare.txt 2>/dev/null || \
    curl -o data/domains/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
  curl -o data/domains/code.txt    https://raw.githubusercontent.com/python/cpython/main/Lib/argparse.py
  curl -o data/domains/spanish.txt https://www.gutenberg.org/files/2000/2000-0.txt

If a file is missing the script falls back to a tiny embedded sample so it still runs.

Run:  python steps/05_multidomain.py
"""

import os
import random
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

# tiny per-domain fallbacks (only used if the real files aren't downloaded yet)
_FALLBACK = {
    "shakespeare": ("KING:\nWhat news, my lord? Speak, for the hour is late,\n"
                    "And shadows fall upon the castle wall.\n") * 60,
    "code":        ("def route(tokens, experts, top_k=2):\n"
                    "    scores = router(tokens)\n"
                    "    idx = scores.topk(top_k).indices\n"
                    "    return sum(experts[i](tokens) for i in idx)\n") * 60,
    "spanish":     ("En un lugar de la Mancha, de cuyo nombre no quiero acordarme, "
                    "no ha mucho tiempo que vivia un hidalgo de los de lanza en astillero.\n") * 60,
}


def load_domains():
    """Read data/domains/<name>.txt for each domain; fall back to an embedded snippet."""
    base = os.path.join(HERE, "..", "data", "domains")
    files = {"shakespeare": "shakespeare.txt", "code": "code.txt", "spanish": "spanish.txt"}
    domains = []
    for name, fn in files.items():
        path = os.path.join(base, fn)
        if os.path.exists(path):
            text = open(path, encoding="utf-8", errors="ignore").read()
        else:
            text = _FALLBACK[name]
            print(f"[data] {path} missing → tiny embedded '{name}' sample (curl the real file for a real run)")
        domains.append((name, text))
    return domains


class CharMultiDomain:
    def __init__(self, domains, block_size, device, split=0.9):
        self.names = [name for name, _ in domains]               # e.g. ["shakespeare","code","spanish"]
        # ONE shared vocabulary over all domains, so a char id means the same everywhere
        all_text = "".join(text for _, text in domains)
        chars = sorted(set(all_text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)
        # tokenize each domain separately and split into train/val (kept per-domain on purpose,
        # so we can later draw VAL batches from one domain at a time)
        self.train, self.val = {}, {}
        for name, text in domains:
            d = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
            n = int(len(d) * split)
            self.train[name], self.val[name] = d[:n], d[n:]
        self.block_size, self.device = block_size, device

    def get_batch(self, split, batch_size, domain=None):
        """Return (x, y, dom): inputs, next-token targets, and a domain id per sequence.
        domain=None → a MIXED batch (random domain per sequence, for training).
        domain="code" → all sequences from that one domain (for the routing probe)."""
        store = self.train if split == "train" else self.val
        picks = [domain or random.choice(self.names) for _ in range(batch_size)]
        xs, ys, doms = [], [], []
        for nm in picks:
            d = store[nm]
            i = torch.randint(len(d) - self.block_size - 1, (1,)).item()
            xs.append(d[i:i + self.block_size])
            ys.append(d[i + 1:i + 1 + self.block_size])
            doms.append(self.names.index(nm))                    # domain → integer label
        x   = torch.stack(xs).to(self.device)
        y   = torch.stack(ys).to(self.device)
        dom = torch.tensor(doms, device=self.device)
        return x, y, dom

    def decode(self, t):
        return "".join(self.itos[int(i)] for i in t)


# ----------------------------- TEST (self-checking) -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    data = CharMultiDomain(load_domains(), block_size=64, device="cpu")

    print("=== Step 5: multi-domain dataset ===")
    print("domains:", data.names, " shared vocab size:", data.vocab_size)
    for name in data.names:
        print(f"  {name:12s} train chars: {len(data.train[name]):>8d}   val chars: {len(data.val[name]):>7d}")

    # (a) a MIXED batch carries a domain label per sequence
    x, y, dom = data.get_batch("train", batch_size=6)
    print("\nmixed batch  x:", tuple(x.shape), " domain labels:", dom.tolist(),
          "→", [data.names[i] for i in dom.tolist()])
    assert x.shape == y.shape and dom.shape[0] == x.shape[0]

    # (b) a per-domain batch: all sequences from one domain (used by the routing probe)
    for name in data.names:
        xb, _, db = data.get_batch("val", batch_size=4, domain=name)
        assert (db == data.names.index(name)).all()
        print(f"sample [{name}]: {data.decode(xb[0][:60]).strip()[:60]!r}")

    print("\nOK — labeled multi-domain corpus ready. On to step 6 (routing probe / heatmap).")

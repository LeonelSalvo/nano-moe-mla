"""
STEP 9 — BPE tokenizer from scratch  (feature #16)
==================================================

Char-level (what nano-moe-mla uses) has a tiny vocab but LONG sequences (one token per byte).
Byte-Pair Encoding (BPE) — what real LLMs use — starts from the 256 bytes and repeatedly
MERGES the most frequent adjacent pair into a new token. A bigger vocab packs common
chunks ("the", "    ", "def ") into single tokens → far fewer tokens per text → cheaper to
model and longer effective context. The round-trip (encode → decode) must be exact.

Run:  python steps/09_bpe.py
"""

import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))


class BPE:
    def __init__(self):
        self.merges = {}                                # (a, b) -> new_id, in learned order
        self.vocab  = {}                                # id -> bytes

    @staticmethod
    def _merge(ids, pair, new_id):
        """Replace every occurrence of the adjacent `pair` with `new_id`."""
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id); i += 2
            else:
                out.append(ids[i]); i += 1
        return out

    def train(self, text, vocab_size):
        ids = list(text.encode("utf-8"))               # start from raw bytes
        self.vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256
        while next_id < vocab_size:
            pairs = Counter(zip(ids, ids[1:]))          # count adjacent pairs
            if not pairs:
                break
            top = max(pairs, key=pairs.get)             # the most frequent pair → merge it
            ids = self._merge(ids, top, next_id)
            self.merges[top] = next_id
            self.vocab[next_id] = self.vocab[top[0]] + self.vocab[top[1]]
            next_id += 1

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        # apply the merges in the SAME order they were learned (each later merge assumes the
        # earlier ones are already applied)
        for pair, new_id in self.merges.items():
            ids = self._merge(ids, pair, new_id)
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


# ----------------------------- TEST (self-checking) -----------------------------
if __name__ == "__main__":
    # use real text if present, else a small embedded sample
    path = os.path.join(HERE, "..", "data", "input.txt")
    text = open(path, encoding="utf-8").read()[:50000] if os.path.exists(path) else (
        "the quick brown fox. the lazy dog. the the the def route(): return self.value\n" * 200)

    print("=== Step 9: BPE tokenizer ===")
    bpe = BPE()
    bpe.train(text, vocab_size=512)                    # 256 bytes + 256 learned merges
    print(f"learned {len(bpe.merges)} merges  →  vocab size {len(bpe.vocab)}")

    sample = text[:2000]
    enc = bpe.encode(sample)
    dec = bpe.decode(enc)

    # (a) round-trip must be EXACT
    print("round-trip exact?:", dec == sample)
    assert dec == sample, "BPE round-trip failed"

    # (b) compression: BPE tokens vs raw bytes (char-level)
    n_bytes = len(sample.encode("utf-8"))
    print(f"chars/bytes: {n_bytes}   BPE tokens: {len(enc)}   "
          f"compression: {n_bytes / len(enc):.2f}x shorter sequences")
    assert len(enc) < n_bytes, "BPE should shorten the sequence"

    # show a few learned chunks (multi-byte tokens)
    chunks = [bytes(v).decode("utf-8", "replace") for i, v in bpe.vocab.items() if i >= 256 and len(v) > 1][:8]
    print("sample learned tokens:", chunks)
    print("\nOK — BPE works: exact round-trip + shorter sequences than char-level.")

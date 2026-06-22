"""
data_prep.py — build a BIGGER, BALANCED multi-domain corpus for a real (micro) run.

The default step-5 files are tiny and imbalanced (code was a single argparse.py). For meaningful
val-loss / routing-specialization metrics we need each domain to be (a) big enough that a micro
model can't just memorize it and (b) roughly the SAME size, so the router sees each domain equally.

This downloads three domains, caps each to the SAME number of MB (balance > raw size for the probe),
and writes data/domains/{english,code,spanish}.txt:

  • english — a handful of large public-domain books (Project Gutenberg)
  • code    — real Python: the CPython source tree (.py files concatenated)
  • spanish — large public-domain Spanish books (Project Gutenberg)

  MB_PER_DOMAIN=8 python data_prep.py     # ~8 MB per domain (tune up for the long run)

Resilient: a failed download is skipped; domains are capped to the smallest one so they stay
balanced. After it runs, CHECK THE PRINTED SIZES — if one domain is short, add files by hand
(any .txt in data/domains/ for that domain) and re-run, or lower MB_PER_DOMAIN.
"""

import os
import io
import sys
import gzip
import tarfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DOM  = os.path.join(HERE, "data", "corpus")          # separate from the char-level data/domains/
MB   = float(os.environ.get("MB_PER_DOMAIN", "8"))
CAP  = int(MB * 1024 * 1024)

GUTENBERG = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
ENGLISH_IDS = [2701, 2600, 1342, 1400, 98, 345, 1661, 84, 1232, 2542]   # Moby Dick, War&Peace, ...
SPANISH_IDS = [2000, 5114, 15532, 57592, 49836, 25040]                  # Quijote, etc. (some may 404)
CPYTHON_TGZ = "https://github.com/python/cpython/archive/refs/tags/v3.12.0.tar.gz"


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "nano-moe-mla/data_prep"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def gutenberg(ids, cap):
    """Concatenate Gutenberg books until we hit `cap` bytes; skip any that fail."""
    out = []
    total = 0
    for bid in ids:
        if total >= cap:
            break
        try:
            txt = _get(GUTENBERG.format(id=bid)).decode("utf-8", "ignore")
            out.append(txt)
            total += len(txt.encode("utf-8"))
            print(f"    + gutenberg {bid}: {len(txt)/1e6:.1f}M chars (running {total/1e6:.1f}M)")
        except Exception as e:
            print(f"    ! gutenberg {bid} skipped ({e})")
    return "".join(out)


def cpython_py(cap):
    """Download the CPython source tarball and concatenate .py files until `cap` bytes."""
    try:
        raw = _get(CPYTHON_TGZ, timeout=180)
    except Exception as e:
        print(f"    ! cpython download failed ({e})")
        return ""
    out, total = [], 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        for m in tf:
            if total >= cap:
                break
            if m.isfile() and m.name.endswith(".py"):
                try:
                    txt = tf.extractfile(m).read().decode("utf-8", "ignore")
                except Exception:
                    continue
                out.append(txt)
                total += len(txt.encode("utf-8"))
    print(f"    + cpython: {total/1e6:.1f}M chars of .py")
    return "".join(out)


def main():
    os.makedirs(DOM, exist_ok=True)
    print(f"[data_prep] target ≈ {MB:.0f} MB/domain")
    print("  english (Gutenberg):");  english = gutenberg(ENGLISH_IDS, CAP)
    print("  code (CPython .py):");    code    = cpython_py(CAP)
    print("  spanish (Gutenberg):");   spanish = gutenberg(SPANISH_IDS, CAP)

    domains = {"english": english, "code": code, "spanish": spanish}
    have = {k: len(v.encode("utf-8")) for k, v in domains.items() if v}
    if not have:
        print("[data_prep] ERROR: every download failed. Check your connection and retry.")
        sys.exit(1)
    # BALANCE: cap every domain to the smallest non-empty one (so the router sees each equally)
    balance = min(min(have.values()), CAP)
    print(f"[data_prep] balancing all domains to {balance/1e6:.1f}M bytes each")
    for name, text in domains.items():
        if not text:
            print(f"  ! {name}: EMPTY — add a .txt by hand into data/domains/ and re-run")
            continue
        b = text.encode("utf-8")[:balance].decode("utf-8", "ignore")
        with open(os.path.join(DOM, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write(b)
        print(f"  ✓ {name}.txt  {len(b)/1e6:.1f}M chars")
    print(f"[data_prep] done → {DOM}. Now run with TOKENIZER=bpe (bpe_data.py reads this folder).")


if __name__ == "__main__":
    main()

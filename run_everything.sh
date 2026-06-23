#!/usr/bin/env bash
# Run EVERYTHING for nano-moe-mla, from a clean checkout to the full stack-ablation matrix:
#   environment  →  download real multi-domain data  →  steps 1..11  →  train + probe + ablation
#   →  step 12 (stack ablation matrix).
#
#   bash run_everything.sh              # nano scale: fast smoke test (a few minutes)
#   SCALE=micro bash run_everything.sh  # micro scale: the real run (sized for a 24 GB GPU)
#
# Re-runnable: skips the venv and any data file already present.
# Uses the venv's python directly (.venv/bin/python) so pip never touches the system env (PEP 668).
set -e
cd "$(dirname "$0")"
SCALE="${SCALE:-nano}"
echo "### nano-moe-mla — full run (scale=$SCALE) ###"

# 1) environment ------------------------------------------------------------
if [ ! -x .venv/bin/python ]; then
  echo "### creating .venv ###"
  python3 -m venv .venv || {
    echo "[error] could not create the venv. On Debian/Ubuntu/Pop!_OS run:"
    echo "        sudo apt install -y python3-venv python3-full"
    exit 1
  }
fi
PY="$(pwd)/.venv/bin/python"          # always use the venv's interpreter explicitly
echo "### installing requirements (torch, etc.) into .venv ###"
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q -r requirements.txt

# 2) data: TinyShakespeare + the 3 real domain files (small slices are enough) -----
mkdir -p data data/domains
dl () {  # dl <url> <dest> : download only if missing; on failure the scripts use a tiny fallback
  if [ ! -f "$2" ]; then
    curl -fsSL -o "$2" "$1" || echo "[warn] could not fetch $1 — an embedded fallback will be used"
  fi
}
dl https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt data/input.txt
[ -f data/domains/shakespeare.txt ] || cp data/input.txt data/domains/shakespeare.txt 2>/dev/null || true
dl https://raw.githubusercontent.com/python/cpython/main/Lib/argparse.py data/domains/code.txt
dl https://www.gutenberg.org/files/2000/2000-0.txt                       data/domains/spanish.txt

# 3) self-checking component tests (seconds each) ---------------------------
echo "### self-tests: building each piece from scratch ###"
for s in 01_moe 02_mla 03_block_model 08_kv_cache; do
  echo "--- steps/$s.py ---"
  "$PY" "steps/$s.py"
done

# 4) train + measurement (writes the result images) -------------------------
echo "### train + measure ###"
"$PY" steps/04_train.py && "$PY" plot_loss.py   # loss_curve.png
"$PY" steps/05_multidomain.py                   # corpus sanity
"$PY" steps/06_routing_probe.py                 # routing_heatmap_lb-*.png
"$PY" steps/07_ablation.py                       # ablation.png

# 5) the full stack-ablation matrix (each technique ON vs OFF) --------------
#    SEEDS>1 reports mean ± std (the error bar that tells signal from noise). Default 1 for speed.
#    TOKENIZER=bpe → build the bigger BALANCED corpus first, then measure on BPE tokens (real metrics).
TOKENIZER="${TOKENIZER:-char}"
if [ "$TOKENIZER" = "bpe" ]; then
  echo "### building balanced multi-domain corpus (data_prep.py) ###"
  MB_PER_DOMAIN="${MB_PER_DOMAIN:-8}" "$PY" data_prep.py
fi
echo "### stack ablation matrix (scale=$SCALE, tokenizer=$TOKENIZER, seeds=${SEEDS:-1}, iters=${ITERS:-default}) ###"
SCALE="$SCALE" SEEDS="${SEEDS:-1}" TOKENIZER="$TOKENIZER" ${ITERS:+ITERS="$ITERS"} "$PY" steps/09_stack_ablation.py

echo
echo "### DONE — everything ran (scale=$SCALE). Result images + the printed matrix above. ###"

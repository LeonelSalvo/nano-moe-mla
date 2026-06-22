#!/usr/bin/env bash
# Run EVERYTHING for nano-moe-mla, from a clean checkout to the full stack-ablation matrix:
#   environment  →  download real multi-domain data  →  steps 1..11  →  train + probe + ablation
#   →  step 12 (stack ablation matrix).
#
#   bash run_everything.sh              # nano scale: fast smoke test (a few minutes)
#   SCALE=micro bash run_everything.sh  # micro scale: the real run (sized for a 24 GB GPU)
#
# Re-runnable: skips the venv and any data file already present.
set -e
cd "$(dirname "$0")"
SCALE="${SCALE:-nano}"
echo "### nano-moe-mla — full run (scale=$SCALE) ###"

# 1) environment ------------------------------------------------------------
if [ ! -d .venv ]; then
  echo "### creating .venv + installing requirements (torch, etc.) ###"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

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
for s in 01_moe 02_mla 03_block_model 08_kv_cache 09_bpe 10_mtp 11_muon; do
  echo "--- steps/$s.py ---"
  python "steps/$s.py"
done

# 4) train + measurement (writes the result images) -------------------------
echo "### train + measure ###"
python steps/04_train.py && python plot_loss.py     # loss_curve.png
python steps/05_multidomain.py                      # corpus sanity
python steps/06_routing_probe.py                    # routing_heatmap_lb-*.png
python steps/07_ablation.py                         # ablation.png

# 5) the full stack-ablation matrix (each technique ON vs OFF) --------------
echo "### stack ablation matrix (scale=$SCALE) ###"
SCALE="$SCALE" python steps/12_stack_ablation.py

echo
echo "### DONE — everything ran (scale=$SCALE). Result images + the printed matrix above. ###"

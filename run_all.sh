#!/usr/bin/env bash
# Run everything in order: the fast self-checking steps, then the training/measurement
# steps that produce the result images. ~15 min total on a single GPU.
#
#   bash run_all.sh
set -e
cd "$(dirname "$0")"

echo "### fast self-tests (seconds each) ###"
for s in 01_moe 02_mla 03_block_model 08_kv_cache; do
  echo "--- steps/$s.py ---"
  python "steps/$s.py"
done

echo
echo "### training + measurement (minutes each, writes the result images) ###"
python steps/04_train.py        && python plot_loss.py        # loss_curve.png
python steps/06_routing_probe.py                              # routing_heatmap_lb-*.png
python steps/07_ablation.py                                   # ablation.png

echo
echo "### done — all steps ran; result images regenerated. ###"

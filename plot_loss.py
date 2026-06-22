"""
plot_loss.py — draw the train/val loss curve from data/train_log.csv (written by
steps/04_train.py). Saves loss_curve.png for the README.

    python plot_loss.py
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")                                  # windowless backend: just save a file
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(HERE, "data", "train_log.csv")
OUT  = os.path.join(HERE, "loss_curve.png")

# read the CSV the training loop wrote (one row per evaluation)
iters, train, val = [], [], []
with open(CSV) as f:
    for row in csv.DictReader(f):
        iters.append(int(row["iter"]))
        train.append(float(row["train"]))
        val.append(float(row["val"]))

# plot train and val together
plt.figure(figsize=(7, 4.5))
plt.plot(iters, train, label="train", linewidth=2)
plt.plot(iters, val,   label="val",   linewidth=2)
plt.xlabel("iteration")
plt.ylabel("loss (cross-entropy)")
plt.title("nano-moe-mla (sparse: MoE + MLA) — TinyShakespeare")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT, dpi=120)
print(f"curve saved to {OUT}  (final val: {val[-1]:.4f})")

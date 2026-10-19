"""Write the gene tables and lookup arrays that reference annotators work from.

    uv run --group eval python eval/reference_inputs.py DATASET OUTDIR

Writes OUTDIR/DATASET.{00-24,...}.txt (per metagene: usage share, effective number of genes, top 50
genes by weight and 25 more by weight x specificity) and OUTDIR/DATASET.npz for eval/query.py.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from datasets import load  # noqa: E402

from metagene_describer.summary import gene_specificity

name, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
genes, H, W = load(name)
H = H.T.astype(np.float64)
usage = W.sum(0).astype(np.float64)
k = H.shape[0]
w = H / H.sum(1, keepdims=True)
sp = gene_specificity(H, usage)
np.savez(f"{outdir}/{name}.npz", genes=np.array(genes), weights=w, specificity=sp)
for start in range(0, k, 25):
    stop = min(start + 25, k)
    with open(f"{outdir}/{name}.{start:02d}-{stop - 1:02d}.txt", "w") as f:
        for i in range(start, stop):
            nz = w[i][w[i] > 0]
            eff = float(np.exp(-(nz * np.log(nz)).sum()))
            f.write(f"### Metagene {i}\n")
            f.write(f"usage share {100 * usage[i] / usage.sum():.2f}%, effective number of genes {eff:.0f}\n")
            top = np.argsort(-w[i])[:50]
            f.write("top 50 by weight (gene weight% specificity):\n")
            f.write(", ".join(f"{genes[j]} {100 * w[i, j]:.2f} {sp[i, j]:.2f}" for j in top) + "\n")
            score = w[i] * sp[i]
            score[top] = -1
            spec = np.argsort(-score)[:25]
            f.write("next 25 by weight x specificity:\n")
            f.write(", ".join(f"{genes[j]} {100 * w[i, j]:.2f} {sp[i, j]:.2f}" for j in spec) + "\n\n")

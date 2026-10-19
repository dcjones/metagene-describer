"""Look up genes in one metagene, for reference annotators.

    python eval/query.py INPUTDIR DATASET METAGENE GENE [GENE ...]

Prints each gene's rank by weight, weight and specificity, or "not measured". With METAGENE=all,
lists the metagenes where each gene is most specific. INPUTDIR is where reference_inputs.py wrote
DATASET.npz.
"""

import sys

import numpy as np

d = np.load(f"{sys.argv[1]}/{sys.argv[2]}.npz")
genes = list(d["genes"])
w, sp = d["weights"], d["specificity"]
idx = {g: j for j, g in enumerate(genes)}
for g in sys.argv[4:]:
    if g not in idx:
        print(f"{g}: not measured")
        continue
    j = idx[g]
    if sys.argv[3] == "all":
        top = np.argsort(-sp[:, j])[:5]
        print(f"{g}: highest specificity in metagenes " + ", ".join(
            f"{i} (spec {sp[i, j]:.2f}, weight {100 * w[i, j]:.2f}%)" for i in top))
    else:
        i = int(sys.argv[3])
        rank = int((w[i] > w[i, j]).sum()) + 1
        print(f"{g}: rank {rank}/{len(genes)}, weight {100 * w[i, j]:.3f}%, specificity {sp[i, j]:.2f}")

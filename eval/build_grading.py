"""Build blind grading inputs: per metagene, the reference and the distinct candidate labels from
several result files, shuffled and unattributed.

    python eval/build_grading.py OUTDIR VARIANT [VARIANT ...]

Reads eval/references/DATASET.jsonl and eval/results/DATASET.VARIANT.tsv for each dataset that has
all variants. Writes OUTDIR/DATASET.jsonl (one line per metagene) and OUTDIR/key.json, mapping
DATASET/INDEX/CANDIDATE to its label and the variants that produced it. A grader writes
OUTDIR/DATASET.grades.jsonl; see eval/README.md.
"""

import json
import os
import random
import sys

import pandas as pd

base = os.path.dirname(os.path.abspath(__file__))
outdir, variants = sys.argv[1], sys.argv[2:]
os.makedirs(outdir, exist_ok=True)
rng = random.Random(0)
key = {}
for ref_file in sorted(os.listdir(f"{base}/references")):
    d = ref_file.removesuffix(".jsonl")
    paths = {v: f"{base}/results/{d}.{v}.tsv" for v in variants}
    if not all(os.path.exists(p) for p in paths.values()):
        print(f"skipping {d}: not every variant has results")
        continue
    ref = {r["index"]: r for r in map(json.loads, open(f"{base}/references/{ref_file}"))}
    tabs = {v: pd.read_csv(p, sep="\t", index_col=0) for v, p in paths.items()}
    with open(f"{outdir}/{d}.jsonl", "w") as f:
        for i in sorted(ref):
            cands = {}
            for v, t in tabs.items():
                cands.setdefault(str(t.label[i]), []).append(v)
            labels = list(cands)
            rng.shuffle(labels)
            ids = {f"c{j}": lab for j, lab in enumerate(labels)}
            for cid, lab in ids.items():
                key[f"{d}/{i}/{cid}"] = {"label": lab, "variants": cands[lab]}
            fields = ["label", "category", "confidence", "acceptable", "overreach", "evidence"]
            r = {k: ref[i][k] for k in fields}
            f.write(json.dumps({"index": i, "reference": r, "candidates": ids}) + "\n")
    print(d, len(ref), "metagenes")
json.dump(key, open(f"{outdir}/key.json", "w"), indent=1)

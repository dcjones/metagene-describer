"""Check that gene names used as examples in the prompts aren't dominant genes of eval metagenes.

    uv run --group eval python eval/check_prompt_leakage.py [--top N]

An example like 'PKM alone is not "Glycolysis"' is harmless in general, but if PKM dominates a
metagene in the eval set, it hands the model that metagene's answer and inflates the score. Prints
every prompt gene that ranks in the top N (default 3) of any eval metagene; exits 1 if any do.
Some hits are canonical markers named in instructions (e.g. CCR7 for naive T cells) rather than
answers; judge each one.
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from datasets import DATASETS, load  # noqa: E402

from metagene_describer import clusters, lookup, prompt, refine

p = argparse.ArgumentParser()
p.add_argument("--top", type=int, default=3)
a = p.parse_args()

text = "\n".join(
    [
        prompt.SYSTEM_PROMPT,
        lookup.PROMPT,
        refine.GROUP_SYSTEM_PROMPT,
        refine.RESOLVE_SYSTEM_PROMPT,
        clusters.SYSTEM_PROMPT,
        clusters.REFINE_SYSTEM_PROMPT,
    ]
)
tokens = set(re.findall(r"\b[A-Z][A-Z0-9-]{2,}\b", text))
found = False
for d in DATASETS:
    genes, H, _ = load(d)
    w = H.T / H.T.sum(axis=1, keepdims=True)
    index = {g: j for j, g in enumerate(genes)}
    for g in sorted(tokens & index.keys()):
        col = w[:, index[g]]
        k = int(col.argmax())
        rank = int((w[k] > col[k]).sum()) + 1
        if rank <= a.top:
            found = True
            print(f"{d}: {g} is #{rank} in metagene {k} ({100 * col[k]:.1f}% of its weight)")
sys.exit(1 if found else 0)

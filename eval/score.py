"""Count grades per variant from a graded directory made by build_grading.py.

    python eval/score.py GRADEDIR [VARIANT ...]
"""

import collections
import glob
import json
import os
import sys

GRADES = ["correct", "vague", "overreach", "wrong"]
gdir = sys.argv[1]
key = json.load(open(f"{gdir}/key.json"))
variants = sys.argv[2:] or sorted({v for k in key.values() for v in k["variants"]})
total = {v: collections.Counter() for v in variants}
for path in sorted(glob.glob(f"{gdir}/*.grades.jsonl")):
    d = os.path.basename(path).removesuffix(".grades.jsonl")
    per = {v: collections.Counter() for v in variants}
    for r in map(json.loads, open(path)):
        for cid, g in r["grades"].items():
            for v in key[f"{d}/{r['index']}/{cid}"]["variants"]:
                if v in per:
                    per[v][g] += 1
                    total[v][g] += 1
    print(d)
    for v in variants:
        print(f"  {v:10s} " + "  ".join(f"{g} {per[v][g]:3d}" for g in GRADES))
print("all")
for v in variants:
    print(f"  {v:10s} " + "  ".join(f"{g} {total[v][g]:3d}" for g in GRADES))

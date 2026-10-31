"""Score cluster labels from run_clusters.py against per-cell reference cell types.

    uv run --group eval python eval/score_clusters.py DATASET RESULTS.tsv [...]
        [--basis expr|metagene] [--res 2.0] [--detail] [--mapper-model M]

Each distinct label is mapped onto the cell type hierarchy in cluster_reference.py by one blind
request that sees only label text (cached in clusters/label-nodes.json, so a label is always
mapped the same way). A label that names alternatives or a mixture maps to several nodes.

Then, at each level of the hierarchy, each cell is graded against its reference cell type by its
cluster's label: correct, wrong, or unresolved (the label stops above this level). A label with n
nodes gives each 1/n of the credit. Cells whose reference type stops above a level are left out at
that level. The oracle row gives each cluster its majority reference type at each level: the best
any single label per cluster could do, given how pure the clusters are.
"""

import argparse
import collections
import json
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import cluster_reference as ref  # noqa: E402

from metagene_describer import AnthropicBackend  # noqa: E402

CACHE = os.path.join(os.path.dirname(__file__), "clusters", "label-nodes.json")
LEVELS = ["compartment", "lineage", "subset", "state", "fine state"]

MAPPER_PROMPT = """\
You map free-text cell type labels for clusters of human blood cells onto a fixed hierarchy of \
cell types. You see only the labels, not the data; judge each label by what it says.

The hierarchy (each node with its parent):
{tree}

For each label, give the most specific node or nodes that the label asserts:
- Map to the node that matches the label's identity, at the specificity the label states. A label \
that falls between nodes maps to the nearest node it fully implies: "Naive T cells" doesn't say CD4 \
or CD8, so it maps to "T cell"; "Memory B cells" maps to "memory B cell".
- Hedged qualifiers still count: "MAIT-like T cells" maps to "MAIT cell" and "Naive T cells \
(CD4-leaning)" to "naive CD4 T cell".
- If the label names alternatives or a mixture of identities ("CD8 T / NK cells", "naive/central \
memory CD4 T cells", "Macrophage/fibroblast mixture"), list each of them.
- Ignore states and programs that aren't in the hierarchy (cycling, activated, interferon-high, \
immediate-early high, cytotoxic) and notes on contamination or spillover such as "(+ dendritic \
cell signal)": map the identity they qualify.
- "Effector memory" and "central memory" map to those nodes; "memory" alone to the memory node. \
Regulatory T cells (Tregs) are CD4 unless the label says CD8.
- A label that names no cell type of this hierarchy maps to "blood cell".

Return every label exactly as given, with its nodes.
"""


def tree_text() -> str:
    lines = []

    def walk(node, depth):
        lines.append("  " * depth + node)
        for child in [c for c, p in ref.PARENT.items() if p == node]:
            walk(child, depth + 1)

    walk("blood cell", 0)
    return "\n".join(lines)


def map_labels(labels: list[str], model: str) -> dict[str, list[str]]:
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo = sorted(set(labels) - set(cache))
    if todo:
        random.Random(0).shuffle(todo)
        schema = {
            "type": "object",
            "properties": {
                "labels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "nodes": {"type": "array", "items": {"type": "string", "enum": ref.NODES}},
                        },
                        "required": ["label", "nodes"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["labels"],
            "additionalProperties": False,
        }
        backend = AnthropicBackend(model)
        data = backend.complete_json(
            MAPPER_PROMPT.format(tree=tree_text()), "Labels:\n" + "\n".join(todo), schema
        )
        for item in data["labels"]:
            nodes = [n for n in item["nodes"] if n in ref.NODES]
            if item["label"] in todo and nodes:
                cache[item["label"]] = nodes
        missing = set(todo) - set(cache)
        if missing:
            raise RuntimeError(f"the mapper returned no nodes for {sorted(missing)}")
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        json.dump(dict(sorted(cache.items())), open(CACHE, "w"), indent=1)
        print(f"mapped {len(todo)} new labels with {model}", file=sys.stderr)
    return {label: cache[label] for label in labels}


def grade(label_nodes: list[str], ref_path: list[str], level: int) -> tuple[float, float, float]:
    """(correct, wrong, unresolved) credit at a level (1-based) for one cell."""
    out = np.zeros(3)
    for node in label_nodes:
        path = ref.lineage(node) if node != "blood cell" else []
        depth = min(level, len(path))
        if path[:depth] != ref_path[:depth]:
            out[1] += 1
        elif len(path) >= level:
            out[0] += 1
        else:
            out[2] += 1
    return tuple(out / len(label_nodes))


def score(cluster_nodes: dict[str, list[str]], counts: dict[str, collections.Counter]) -> np.ndarray:
    """[levels, 3] cell counts: correct, wrong, unresolved."""
    out = np.zeros((len(LEVELS), 3))
    for c, nodes in cluster_nodes.items():
        for r, n in counts[c].items():
            path = ref.lineage(r)
            for lv in range(1, len(path) + 1):
                out[lv - 1] += n * np.array(grade(nodes, path, lv))
    return out


def oracle(counts: dict[str, collections.Counter]) -> np.ndarray:
    out = np.zeros((len(LEVELS), 3))
    for cnt in counts.values():
        for lv in range(1, len(LEVELS) + 1):
            at = collections.Counter()
            for r, n in cnt.items():
                path = ref.lineage(r)
                if len(path) >= lv:
                    at[path[lv - 1]] += n
            if at:
                best = max(at.values())
                out[lv - 1] += [best, sum(at.values()) - best, 0]
    return out


def report(name: str, s: np.ndarray, width: int):
    cells = []
    for lv in range(len(LEVELS)):
        total = s[lv].sum()
        c, w, u = 100 * s[lv] / total if total else (0, 0, 0)
        cells.append(f"{c:5.1f} {w:5.1f} {u:5.1f}")
    print(f"{name:{width}s}  " + "  ".join(cells))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=list(ref.REFERENCE))
    p.add_argument("results", nargs="+")
    p.add_argument("--basis", default="expr", choices=list(ref.CLUSTERS))
    p.add_argument("--res", default="2.0")
    p.add_argument("--detail", action="store_true", help="print each cluster's label and grade")
    p.add_argument("--mapper-model", default="claude-opus-5-5")
    a = p.parse_args()

    clusters = ref.clusters(a.dataset, a.basis, a.res).astype(str)
    nodes = ref.reference_nodes(a.dataset)
    if len(clusters) != len(nodes):
        raise ValueError(f"{len(clusters)} clustered cells but {len(nodes)} reference cell types")
    counts = collections.defaultdict(collections.Counter)
    for c, r in zip(clusters, nodes):
        counts[c][r] += 1

    tables = {}
    for path in a.results:
        df = pd.read_csv(path, sep="\t", keep_default_na=False, dtype={"cluster": str})
        sizes = {c: sum(n.values()) for c, n in counts.items()}
        if dict(zip(df.cluster, df.n_cells)) != sizes:
            raise ValueError(f"{path} doesn't match the {a.basis} r{a.res} clusters")
        tables[os.path.basename(path).removesuffix(".tsv")] = df
    mapping = map_labels(sorted({l for df in tables.values() for l in df.label}), a.mapper_model)

    width = max(len(n) for n in [*tables, "oracle"])
    print(f"{'':{width}s}  " + "  ".join(f"{l:^17s}" for l in LEVELS))
    print(f"{'':{width}s}  " + "  ".join(" corr wrong unres" for _ in LEVELS))
    report("oracle", oracle(counts), width)
    for name, df in tables.items():
        report(name, score({c: mapping[l] for c, l in zip(df.cluster, df.label)}, counts), width)

    if a.detail:
        for name, df in tables.items():
            print(f"\n{name}")
            for c, label in zip(df.cluster, df.label):
                total = sum(counts[c].values())
                top = ", ".join(f"{r} {100 * n / total:.0f}%" for r, n in counts[c].most_common(3))
                print(f"{c:>4} {total:6d}  {label}  -> {' | '.join(mapping[label])}\n"
                      f"{'':12s}ref: {top}")


if __name__ == "__main__":
    main()

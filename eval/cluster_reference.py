"""Reference cell types for cluster evaluation: a hierarchy of PBMC cell types, the mapping from the
reference annotation's cell type names onto it, and loaders for clusters and reference types.

Cluster and cell type tables come from rules in $METAGENE_EXAMPLES/../Snakefile; their rows are in
the same order as the cells of the metagene run.
"""

import os

import numpy as np
import polars as pl

from datasets import ROOT

BASE = os.path.dirname(ROOT)

# node -> parent. Levels (depth below the root): 1 compartment, 2 lineage, 3 subset, 4 state,
# 5 fine state. A cell or label at a shallow node is unresolved at deeper levels.
PARENT = {
    "lymphoid cell": "blood cell",
    "myeloid cell": "blood cell",
    "progenitor cell": "blood cell",
    "platelet": "blood cell",
    "erythrocyte": "blood cell",
    # lymphoid
    "T cell": "lymphoid cell",
    "NK cell": "lymphoid cell",
    "innate lymphoid cell": "lymphoid cell",
    "B cell": "lymphoid cell",
    "CD4 T cell": "T cell",
    "CD8 T cell": "T cell",
    "unconventional T cell": "T cell",
    "naive CD4 T cell": "CD4 T cell",
    "memory CD4 T cell": "CD4 T cell",
    "regulatory T cell": "CD4 T cell",
    "central memory CD4 T cell": "memory CD4 T cell",
    "effector memory CD4 T cell": "memory CD4 T cell",
    "naive regulatory T cell": "regulatory T cell",
    "memory regulatory T cell": "regulatory T cell",
    "naive CD8 T cell": "CD8 T cell",
    "memory CD8 T cell": "CD8 T cell",
    "CD8 regulatory T cell": "CD8 T cell",
    "central memory CD8 T cell": "memory CD8 T cell",
    "effector memory CD8 T cell": "memory CD8 T cell",
    "MAIT cell": "unconventional T cell",
    "gamma-delta T cell": "unconventional T cell",
    "CD56dim NK cell": "NK cell",
    "CD56bright NK cell": "NK cell",
    "naive B cell": "B cell",
    "transitional B cell": "B cell",
    "memory B cell": "B cell",
    "plasma cell": "B cell",
    # myeloid
    "monocyte": "myeloid cell",
    "dendritic cell": "myeloid cell",
    "classical monocyte": "monocyte",
    "non-classical monocyte": "monocyte",
    "conventional dendritic cell": "dendritic cell",
    "plasmacytoid dendritic cell": "dendritic cell",
    "cDC1": "conventional dendritic cell",
    "cDC2": "conventional dendritic cell",
    "AXL+ dendritic cell": "dendritic cell",
    # progenitors
    "common myeloid progenitor": "progenitor cell",
    "common lymphoid progenitor": "progenitor cell",
    "basophil/mast progenitor": "progenitor cell",
}
NODES = ["blood cell", *PARENT]

# Reference annotation names (Cell Ontology labels) -> nodes.
REFERENCE = {
    "scrnaseq-aging-pbmc-250k": {
        "naive thymus-derived CD4-positive, alpha-beta T cell": "naive CD4 T cell",
        "central memory CD4-positive, alpha-beta T cell": "central memory CD4 T cell",
        "effector memory CD4-positive, alpha-beta T cell": "effector memory CD4 T cell",
        "CD4-positive, alpha-beta memory T cell": "memory CD4 T cell",
        "naive regulatory T cell": "naive regulatory T cell",
        "memory regulatory T cell": "memory regulatory T cell",
        "memory CCR4-positive regulatory T cell": "memory regulatory T cell",
        "naive thymus-derived CD8-positive, alpha-beta T cell": "naive CD8 T cell",
        "central memory CD8-positive, alpha-beta T cell": "central memory CD8 T cell",
        "effector memory CD8-positive, alpha-beta T cell": "effector memory CD8 T cell",
        "CD8-positive, alpha-beta memory T cell": "memory CD8 T cell",
        "CD8-positive, alpha-beta regulatory T cell": "CD8 regulatory T cell",
        "mucosal-associated invariant T cell": "MAIT cell",
        "mature gamma-delta T cell": "gamma-delta T cell",
        "T cell": "T cell",
        # Thymocyte names in blood are T cells whose subset the reference didn't resolve.
        "double negative thymocyte": "T cell",
        "CD8aa(I) thymocyte": "T cell",
        "CD16-positive, CD56-dim natural killer cell, human": "CD56dim NK cell",
        "CD16-negative, CD56-bright natural killer cell, human": "CD56bright NK cell",
        "natural killer cell": "NK cell",
        "innate lymphoid cell": "innate lymphoid cell",
        "naive B cell": "naive B cell",
        "transitional stage B cell": "transitional B cell",
        "memory B cell": "memory B cell",
        # "Be cell" (effector B cell) has no subset of its own here.
        "Be cell": "B cell",
        "plasma cell": "plasma cell",
        "CD14-positive, CD16-negative classical monocyte": "classical monocyte",
        "CD14-low, CD16-positive monocyte": "non-classical monocyte",
        "CD1c-positive myeloid dendritic cell": "cDC2",
        "CD141-positive myeloid dendritic cell": "cDC1",
        "Axl+ dendritic cell, human": "AXL+ dendritic cell",
        "plasmacytoid dendritic cell": "plasmacytoid dendritic cell",
        "common myeloid progenitor": "common myeloid progenitor",
        "common lymphoid progenitor": "common lymphoid progenitor",
        "basophil mast progenitor cell": "basophil/mast progenitor",
        "platelet": "platelet",
        "erythrocyte": "erythrocyte",
    }
}

CELLTYPES = {"scrnaseq-aging-pbmc-250k": "celltypes/scrnaseq-aging-pbmc-250k-celltypes.parquet"}
CLUSTERS = {
    "expr": "results/{dataset}/leiden-expr-r{res}.parquet",
    "metagene": "results/{dataset}/metagene-k100-leiden-r{res}.parquet",
}


def lineage(node: str) -> list[str]:
    """The path from the first level below the root down to node."""
    path = []
    while node != "blood cell":
        path.append(node)
        node = PARENT[node]
    return path[::-1]


def reference_nodes(dataset: str) -> np.ndarray:
    """The reference node of each cell, in cell order."""
    t = pl.read_parquet(f"{BASE}/{CELLTYPES[dataset]}")
    names = t["cell_type"].cast(pl.String)
    missing = set(names.unique()) - set(REFERENCE[dataset])
    if missing:
        raise ValueError(f"reference cell types with no node: {sorted(missing)}")
    return names.replace_strict(REFERENCE[dataset]).to_numpy()


def clusters(dataset: str, basis: str = "expr", res: str = "2.0") -> np.ndarray:
    """The Leiden cluster of each cell, in cell order."""
    t = pl.read_parquet(f"{BASE}/{CLUSTERS[basis].format(dataset=dataset, res=res)}")
    return t["cluster"].to_numpy()

"""Reference cell types for cluster evaluation: a hierarchy of cell types per tissue, the mapping from
each dataset's reference cell type names onto it, and loaders for clusters and reference types.

Cluster and cell type tables come from rules in $METAGENE_EXAMPLES/../Snakefile; their rows are in
the same order as the cells of the metagene run.
"""

import os
from dataclasses import dataclass

import numpy as np
import polars as pl

from datasets import ROOT

BASE = os.path.dirname(ROOT)


@dataclass
class Hierarchy:
    name: str
    tissue: str  # what the cells are, for the label mapper, e.g. "human blood cells"
    root: str
    parent: dict[str, str]  # node -> parent
    levels: list[str]  # names of the levels below the root

    @property
    def nodes(self) -> list[str]:
        return [self.root, *self.parent]

    def lineage(self, node: str) -> list[str]:
        """The path from the first level below the root down to node."""
        path = []
        while node != self.root:
            path.append(node)
            node = self.parent[node]
        return path[::-1]

    def text(self) -> str:
        lines = []

        def walk(node, depth):
            lines.append("  " * depth + node)
            for child in [c for c, p in self.parent.items() if p == node]:
                walk(child, depth + 1)

        walk(self.root, 0)
        return "\n".join(lines)


PBMC = Hierarchy(
    name="pbmc",
    tissue="human blood cells",
    root="blood cell",
    levels=["compartment", "lineage", "subset", "state", "fine state"],
    parent={
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
    },
)

CERVIX = Hierarchy(
    name="cervix",
    tissue="cells of human cervical cancer tissue",
    root="cell",
    levels=["compartment", "lineage", "subset"],
    parent={
        "immune cell": "cell",
        "stromal cell": "cell",
        "epithelial cell": "cell",
        # immune
        "T cell": "immune cell",
        "B lineage cell": "immune cell",
        "myeloid cell": "immune cell",
        "cytotoxic T cell": "T cell",
        "naive/memory T cell": "T cell",
        "regulatory T cell": "T cell",
        "exhausted T cell": "T cell",
        "B cell": "B lineage cell",
        "plasma cell": "B lineage cell",
        "macrophage": "myeloid cell",
        "dendritic cell": "myeloid cell",
        "neutrophil": "myeloid cell",
        "mast cell": "myeloid cell",
        # stromal
        "fibroblast": "stromal cell",
        "smooth muscle cell": "stromal cell",
        "endothelial cell": "stromal cell",
        "interstitial fibroblast": "fibroblast",
        "cancer-associated fibroblast": "fibroblast",
        # epithelial; the vendor's tumor states are squamous
        "squamous epithelial or tumor cell": "epithelial cell",
        "endocervical glandular cell": "epithelial cell",
        "metabolic invasive basal tumor cell": "squamous epithelial or tumor cell",
        "migratory invasive basal tumor cell": "squamous epithelial or tumor cell",
        "hypoxic tumor cell": "squamous epithelial or tumor cell",
        "differentiating tumor cell": "squamous epithelial or tumor cell",
        "parabasal tumor cell": "squamous epithelial or tumor cell",
        "dyskeratotic tumor cell": "squamous epithelial or tumor cell",
        "proliferative parabasal cell": "squamous epithelial or tumor cell",
        "endocervical columnar cell": "endocervical glandular cell",
        "endocervical ciliated cell": "endocervical glandular cell",
    },
)


@dataclass
class Reference:
    hierarchy: Hierarchy
    celltypes: str  # path under BASE
    names: dict[str, str]  # reference cell type name -> node; missing types map to the root


REFERENCES = {
    "scrnaseq-aging-pbmc-250k": Reference(
        PBMC,
        "celltypes/scrnaseq-aging-pbmc-250k-celltypes.parquet",
        {
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
        },
    ),
    "atera-cervical-cancer-50k": Reference(
        CERVIX,
        "celltypes/atera-cervical-cancer-50k-celltypes.parquet",
        {
            "Cytotoxic T Cells": "cytotoxic T cell",
            "Naive & Memory T Cells": "naive/memory T cell",
            "Regulatory T Cells": "regulatory T cell",
            "Exhausted T Cells": "exhausted T cell",
            "B Cells": "B cell",
            "Plasma Cells": "plasma cell",
            "Macrophages": "macrophage",
            "Dendritic Cells": "dendritic cell",
            "Neutrophils": "neutrophil",
            "Mast Cells": "mast cell",
            "Interstitial Fibroblasts": "interstitial fibroblast",
            "Cancer Associated Fibroblasts": "cancer-associated fibroblast",
            "Smooth Muscle": "smooth muscle cell",
            "Stroma & Smooth Muscle": "stromal cell",
            "Endothelial Cells": "endothelial cell",
            "Metabolic Invasive Basal Cells": "metabolic invasive basal tumor cell",
            "Migratory Invasive Basal Cells": "migratory invasive basal tumor cell",
            "Hypoxic Tumor Cells": "hypoxic tumor cell",
            "Differentiating Tumor Cells": "differentiating tumor cell",
            "Parabasal Tumor Cells": "parabasal tumor cell",
            "Dyskeratotic Tumor Cells": "dyskeratotic tumor cell",
            "Proliferative Parabasal Cells": "proliferative parabasal cell",
            "Endocervical Columnar Cells": "endocervical columnar cell",
            "Endocervical Ciliated Cells": "endocervical ciliated cell",
            # Groups that don't name a cell type, left unscored.
            "OR4F17+ Cells": "cell",
            "Detachment": "cell",
        },
    ),
}

CLUSTERS = {
    "expr": "results/{dataset}/leiden-expr-r{res}.parquet",
    "metagene": "results/{dataset}/metagene-k100-leiden-r{res}.parquet",
}


def reference_nodes(dataset: str) -> np.ndarray:
    """The reference node of each cell, in cell order. Cells with no reference type get the root."""
    r = REFERENCES[dataset]
    t = pl.read_parquet(f"{BASE}/{r.celltypes}")
    names = t["cell_type"].cast(pl.String)
    missing = set(names.drop_nulls().unique()) - set(r.names)
    if missing:
        raise ValueError(f"reference cell types with no node: {sorted(missing)}")
    return names.replace_strict(r.names, default=r.hierarchy.root).fill_null(r.hierarchy.root).to_numpy()


def clusters(dataset: str, basis: str = "expr", res: str = "2.0") -> np.ndarray:
    """The Leiden cluster of each cell, in cell order."""
    t = pl.read_parquet(f"{BASE}/{CLUSTERS[basis].format(dataset=dataset, res=res)}")
    return t["cluster"].to_numpy()

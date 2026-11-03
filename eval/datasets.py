"""The example datasets: metagene-k100 runs stored as AnnData zarr (W in obsm["W_metagene"],
H in varm["H_metagene"]). Set METAGENE_EXAMPLES to the directory holding them."""

import os

import numpy as np
import zarr

ROOT = os.environ.get("METAGENE_EXAMPLES", "/mnt/extra2/metagene-examples/results")

DATASETS = {
    "scrnaseq-aging-pbmc-250k": "Human PBMCs from donors of varying ages, 10x scRNA-seq",
    "atera-cervical-cancer": "Human cervical cancer, Atera spatial transcriptomics, whole transcriptome",
    "xenium-5k-ovarian-cancer": "Human high-grade serous ovarian cancer, 10x Xenium Prime 5k panel, FFPE",
    # A 50k-cell region of the Atera cervical cancer demo, segmented with proseg (segreg benchmark).
    "atera-cervical-cancer-50k": "Human cervical cancer, Atera spatial transcriptomics, whole transcriptome, FFPE",
}


def load(name: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Gene names, H [genes, k] and W [cells, k] for a dataset."""
    g = zarr.open_group(f"{ROOT}/{name}/metagene-k100.zarr", mode="r")
    index = g["var/_index"]
    # Gene names may be stored as a nullable-string-array group.
    values = index["values"][:] if isinstance(index, zarr.Group) else index[:]
    genes = [str(x) for x in values]
    return genes, np.asarray(g["varm/H_metagene"][:]), np.asarray(g["obsm/W_metagene"][:])

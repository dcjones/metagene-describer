"""Convenience wrappers for AnnData objects factorized with metagene.nmf.

metagene.nmf, given an AnnData object, stores W in adata.obsm["W_nmf"] ([cells, k]) and Hᵀ in
adata.varm["Ht_nmf"] ([genes, k]). These wrappers read those, call describe_metagenes or
describe_clusters, and store the results back in the object. anndata is never imported here; the
object is used through its attributes. pandas is, but anndata depends on it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .clusters import ClusterDescription, describe_clusters
from .describe import MetageneDescription, describe_metagenes, from_dataframe, to_dataframe

__all__ = ["annotate_metagenes", "annotate_clusters", "read_descriptions"]

# List fields are joined with this in tables stored in adata.uns, so that they can be split back.
# Commas show up in cell type names ("CD4+ T cells, naive"); semicolons almost never do.
SEP = "; "


def _uns_table(descriptions):
    """to_dataframe, made writable to h5ad and zarr: no None in string columns and a string index."""
    df = to_dataframe(descriptions, sep=SEP)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda v: "" if v is None else v)
    if df.index.name == "cluster":
        df.index = df.index.map(str)
    return df


def read_descriptions(adata, key: str = "metagene_descriptions"):
    """Metagene or cluster descriptions stored in adata.uns[key] by annotate_metagenes or
    annotate_clusters, as MetageneDescription or ClusterDescription objects."""
    if key not in adata.uns:
        raise KeyError(f'adata.uns has no "{key}"; run annotate_metagenes first')
    return from_dataframe(adata.uns[key], sep=SEP)


def _factors(adata, varm_key: str, obsm_key: str | None):
    if varm_key not in adata.varm:
        raise KeyError(f'adata.varm has no "{varm_key}"; run metagene.nmf on the AnnData object first')
    # Hᵀ is [genes, k]; transpose explicitly rather than rely on shape guessing, which is ambiguous
    # when k is close to the number of genes.
    H = np.asarray(adata.varm[varm_key]).T
    W = None
    if obsm_key is not None:
        if obsm_key not in adata.obsm:
            raise KeyError(f'adata.obsm has no "{obsm_key}"')
        W = np.asarray(adata.obsm[obsm_key])
    return H, W


def annotate_metagenes(
    adata,
    *,
    varm_key: str = "Ht_nmf",
    obsm_key: str | None = "W_nmf",
    key_added: str = "metagene_descriptions",
    **kwargs,
) -> list[MetageneDescription]:
    """describe_metagenes on an AnnData object factorized with metagene.nmf, storing the results
    in adata.uns[key_added].

    Parameters
    ----------
    adata : AnnData
        Genes are adata.var_names.
    varm_key : str
        Gene weights per metagene, [genes, k] (Hᵀ), in adata.varm.
    obsm_key : str or None
        Per-cell metagene usage, [cells, k] (W), in adata.obsm. None to describe without usage.
    key_added : str
        The descriptions are stored in adata.uns[key_added] as a DataFrame indexed by metagene, with
        list fields joined by "; ". read_descriptions turns it back into MetageneDescription objects.
    **kwargs
        Passed to describe_metagenes (context, backend, model, lookup, refine, ...).

    Returns
    -------
    The list of MetageneDescription, as from describe_metagenes.
    """
    H, W = _factors(adata, varm_key, obsm_key)
    descriptions = describe_metagenes(list(adata.var_names), H, usage=W, **kwargs)
    adata.uns[key_added] = _uns_table(descriptions)
    return descriptions


def _categorical(values: Sequence, mapping: dict[str, str]):
    import pandas as pd

    categories = list(dict.fromkeys(mapping.values()))
    return pd.Categorical([mapping.get(v) for v in values], categories=categories)


def annotate_clusters(
    adata,
    cluster_key: str = "leiden",
    *,
    descriptions: Sequence[MetageneDescription] | None = None,
    metagene_key: str = "metagene_descriptions",
    varm_key: str = "Ht_nmf",
    obsm_key: str = "W_nmf",
    key_added: str | None = None,
    **kwargs,
) -> list[ClusterDescription]:
    """describe_clusters on an AnnData object factorized with metagene.nmf, storing labels in obs
    and the full results in uns.

    Parameters
    ----------
    adata : AnnData
        Genes are adata.var_names.
    cluster_key : str
        Column of adata.obs with a cluster label per cell, e.g. from scanpy.tl.leiden.
    descriptions : sequence of MetageneDescription, optional
        Metagene descriptions. By default, read from adata.uns[metagene_key], as stored by
        annotate_metagenes.
    metagene_key : str
        Where annotate_metagenes stored the metagene descriptions.
    varm_key, obsm_key : str
        Hᵀ [genes, k] in adata.varm and W [cells, k] in adata.obsm, as stored by metagene.nmf.
    key_added : str, optional
        Prefix for the stored results; defaults to cluster_key. Each cell's cluster label goes in
        adata.obs[f"{key_added}_label"] and its broad label in adata.obs[f"{key_added}_broad_label"]
        (both categorical), and the full descriptions in adata.uns[f"{key_added}_descriptions"] as a
        DataFrame indexed by cluster (as strings), with list fields joined by "; ". With broad=False,
        no broad label column is written, and any earlier one is removed.
    **kwargs
        Passed to describe_clusters (context, notes, backend, model, refine, broad, ...).

    Returns
    -------
    The list of ClusterDescription, as from describe_clusters.
    """
    if cluster_key not in adata.obs:
        raise KeyError(f'adata.obs has no column "{cluster_key}"')
    if descriptions is None:
        descriptions = read_descriptions(adata, metagene_key)
    H, W = _factors(adata, varm_key, obsm_key)
    clusters = adata.obs[cluster_key]
    results = describe_clusters(clusters, descriptions, list(adata.var_names), H, W, **kwargs)

    key = cluster_key if key_added is None else key_added
    # Match cells to results by string, so that e.g. int cluster ids from np.unique still match a
    # categorical column of the same ids.
    cell_ids = clusters.astype(str).tolist()
    adata.obs[f"{key}_label"] = _categorical(cell_ids, {str(r.cluster): r.label for r in results})
    broad_key = f"{key}_broad_label"
    if all(r.broad_label is not None for r in results):
        adata.obs[broad_key] = _categorical(
            cell_ids, {str(r.cluster): str(r.broad_label) for r in results}
        )
    elif broad_key in adata.obs:
        del adata.obs[broad_key]
    adata.uns[f"{key}_descriptions"] = _uns_table(results)
    return results

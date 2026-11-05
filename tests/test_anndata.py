import numpy as np
import pandas as pd
import pytest

from metagene_describer import (
    MetageneDescription,
    annotate_clusters,
    annotate_metagenes,
    from_dataframe,
    read_descriptions,
    to_dataframe,
)

from test_clusters import DESCS, GENES, FakeBackend, make_H, make_W

ad = pytest.importorskip("anndata")


class MetageneBackend:
    def complete_json(self, system, user, schema, tools=()):
        b = "MS4A1" in user
        return {
            "description": "B cells." if b else "T cells.",
            "key_genes": ["MS4A1"] if b else ["CD3E", "CD3D"],
            "category": "cell_type",
            "cell_types": ["B cells, naive", "B cells"] if b else ["T cells"],
            "confidence": "high",
            "label": "B cell markers" if b else "T cell markers",
        }


def make_adata(clusters=None):
    labels, W = make_W()
    adata = ad.AnnData(np.zeros((len(labels), len(GENES)), dtype=np.float32))
    adata.var_names = GENES
    adata.obs["leiden"] = pd.Categorical(labels if clusters is None else clusters)
    adata.obsm["W_nmf"] = W
    adata.varm["Ht_nmf"] = make_H().T
    return adata


def test_annotate_metagenes(tmp_path):
    adata = make_adata()
    out = annotate_metagenes(adata, backend=MetageneBackend(), lookup=False)
    df = adata.uns["metagene_descriptions"]
    assert list(df.index) == [0, 1, 2] and df.loc[1, "cell_types"] == "B cells, naive; B cells"
    adata.write_h5ad(tmp_path / "a.h5ad")
    back = read_descriptions(ad.read_h5ad(tmp_path / "a.h5ad"))
    assert back == out


def test_round_trip():
    descs = DESCS + [MetageneDescription(3, "x", "", "ambiguous", "low", error="boom")]
    assert from_dataframe(to_dataframe(descs)) == descs


def test_annotate_clusters(tmp_path):
    adata = make_adata()
    adata.uns["metagene_descriptions"] = to_dataframe(DESCS, sep="; ")
    backend = FakeBackend(refine=[], broad=[{"broad_label": "Lymphocytes", "labels": ["T cells"]}])
    out = annotate_clusters(adata, backend=backend)
    labels = adata.obs["leiden_label"]
    assert isinstance(labels.dtype, pd.CategoricalDtype)
    assert labels.tolist() == ["T cells"] * 10 + ["B cells"] * 10
    assert adata.obs["leiden_broad_label"].tolist() == ["Lymphocytes"] * 10 + ["B cells"] * 10
    df = adata.uns["leiden_descriptions"]
    assert list(df.index) == ["B", "T"] and df.loc["B", "contaminants"] == "T cells"
    adata.write_zarr(tmp_path / "a.zarr")
    back = read_descriptions(ad.read_zarr(tmp_path / "a.zarr"), "leiden_descriptions")
    assert [(b.cluster, b.label, b.key_metagenes, b.initial_label) for b in back] == [
        (o.cluster, o.label, o.key_metagenes, o.initial_label) for o in out
    ]

    annotate_clusters(adata, backend=backend, broad=False, key_added="leiden")
    assert "leiden_broad_label" not in adata.obs


def test_annotate_clusters_int_ids():
    labels, _ = make_W()
    adata = make_adata(np.where(labels == "T", 10, 2))
    annotate_clusters(adata, descriptions=DESCS, backend=FakeBackend(), key_added="c")
    assert adata.obs["c_label"].notna().all()
    assert list(adata.uns["c_descriptions"].index) == ["2", "10"]

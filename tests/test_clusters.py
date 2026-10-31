import numpy as np
import pytest

from metagene_describer import (
    BackendError,
    MetageneDescription,
    build_cluster_prompts,
    describe_clusters,
    to_dataframe,
)
from metagene_describer.clusters import summarize_clusters

GENES = ["CD3E", "CD3D", "CD2", "MS4A1", "CD79A", "ACTB", "MALAT1", "MKI67"]


def make_H():
    # metagene 0: T cells, 1: B cells, 2: cycling; 0 and 1 load heavily on ACTB/MALAT1
    H = np.zeros((3, len(GENES)))
    H[0, [0, 1, 2]] = [5, 4, 3]
    H[1, [3, 4]] = [6, 5]
    H[:2, [5, 6]] = 10
    H[2, 7] = 2
    return H


def make_W():
    # 10 T cells, then 10 B cells, two of which also carry T cell signal (doublets).
    W = np.zeros((20, 3))
    W[:10, 0] = 1.0
    W[10:, 1] = 1.0
    W[10:12, 0] = 1.0
    W[:5, 2] = 0.5  # half the T cells are cycling
    clusters = np.array(["T"] * 10 + ["B"] * 10)
    return clusters, W


DESCS = [
    MetageneDescription(0, "T cell markers", "CD3 genes.", "cell_type", "high", key_genes=["CD3E"]),
    MetageneDescription(1, "B cell markers", "MS4A1, CD79A.", "cell_type", "high"),
    MetageneDescription(2, "Cycling cells", "MKI67.", "program", "high"),
]


class FakeBackend:
    def __init__(self, refine=None, fail=()):
        self.calls, self.refine_calls = [], []
        self.refine, self.fail = refine, set(fail)

    def complete_json(self, system, user, schema, tools=()):
        if "labels" in schema["properties"]:
            self.refine_calls.append(user)
            if self.refine is None:
                raise BackendError("bad json")
            return {"labels": self.refine}
        self.calls.append(user)
        if len(self.calls) in self.fail:
            raise BackendError("bad json")
        b = "Cluster B" in user
        return {
            "description": "B cells with T cell doublets." if b else "T cells.",
            "key_metagenes": [1, 99] if b else [0],
            "contaminants": ["T cells"] if b else [],
            "category": "cell_type",
            "cell_types": ["B cells"] if b else ["T cells"],
            "confidence": "high",
            "label": "B cells." if b else "T cells",
        }


def test_summarize_clusters():
    clusters, W = make_W()
    H = make_H()
    out = summarize_clusters(clusters, GENES, H, W)
    assert [s.cluster for s in out] == ["B", "T"]
    b, t = out
    assert b.n_cells == 10 and b.cell_share == pytest.approx(0.5)
    shares = {m.index: m for m in b.metagenes}
    # Modeled counts: 10 * 31 from metagene 1, 2 * 32 from metagene 0.
    assert shares[1].share == pytest.approx(310 / 374)
    assert shares[0].share == pytest.approx(64 / 374)
    assert shares[0].prevalence == pytest.approx(0.2)
    assert shares[0].peak == ("T", pytest.approx(320 / 325))
    assert shares[1].peak is None
    assert b.metagenes[0].index == 1
    assert {m.index for m in t.metagenes} == {0, 2}
    assert t.genes[0][0] in {"CD3E", "MKI67"}
    assert "MS4A1" not in [g for g, _, _ in t.genes]


def test_rescaling_invariance():
    clusters, W = make_W()
    H = make_H()
    H2, W2 = H.copy(), W.copy()
    H2[0] *= 100
    W2[:, 0] /= 100
    a = summarize_clusters(clusters, GENES, H, W)
    b = summarize_clusters(clusters, GENES, H2, W2)
    for x, y in zip(a, b):
        assert [m.share for m in x.metagenes] == pytest.approx([m.share for m in y.metagenes])
        assert [m.prevalence for m in x.metagenes] == [m.prevalence for m in y.metagenes]


def test_prompt():
    clusters, W = make_W()
    _, users = build_cluster_prompts(
        clusters, DESCS, GENES, make_H(), W, context="Human PBMC", notes={"B": "from patient P3"}
    )
    b = users[0]
    assert "Human PBMC" in b and "from patient P3" in b and "Cells: 10" in b
    assert '"B cell markers" [cell_type, high]' in b
    assert "largest share in cluster T" in b
    assert "from patient P3" not in users[1]


def test_describe_clusters_and_refine():
    clusters, W = make_W()
    backend = FakeBackend(
        refine=[
            {"cluster": "T", "reason": "consistent wording", "label": "T cells, cycling"},
            {"cluster": "nope", "reason": "", "label": "x"},
        ]
    )
    out = describe_clusters(clusters, DESCS, GENES, make_H(), W, backend=backend)
    b, t = out
    assert b.label == "B cells" and b.key_metagenes == [1] and b.contaminants == ["T cells"]
    assert b.metagene_label == "B cell markers + T cell markers"
    assert t.label == "T cells, cycling" and t.initial_label == "T cells"
    assert b.initial_label is None
    assert "Cluster B (10 cells) [cell_type] B cells" in backend.refine_calls[0]
    df = to_dataframe(out)
    assert list(df.index) == ["B", "T"] and df.loc["B", "contaminants"] == "T cells"


def test_failures_fall_back(capsys):
    clusters, W = make_W()
    backend = FakeBackend(fail={1, 2})
    out = describe_clusters(clusters, DESCS, GENES, make_H(), W, backend=backend, max_attempts=2)
    failed = [o for o in out if o.error]
    assert len(failed) == 1 and failed[0].label == failed[0].metagene_label
    assert "refinement failed" in capsys.readouterr().err


def test_bad_usage_shape():
    clusters, W = make_W()
    with pytest.raises(ValueError):
        describe_clusters(clusters, DESCS, GENES, make_H(), W.T, backend=FakeBackend())


def test_numeric_string_clusters_sort_as_numbers():
    _, W = make_W()
    clusters = np.array(["10"] * 10 + ["2"] * 10)
    out = summarize_clusters(clusters, GENES, make_H(), W)
    assert [s.cluster for s in out] == ["2", "10"]
    assert out[1].metagenes[0].index == 0  # cluster "10" holds the T cells

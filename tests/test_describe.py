import numpy as np
import pytest

from metagene_describer import BackendError, build_prompts, describe_metagenes
from metagene_describer.backends import parse_json
from metagene_describer.summary import summarize

GENES = ["CD3E", "CD3D", "CD2", "MS4A1", "CD79A", "ACTB", "MALAT1", "MKI67"]


def make_H():
    # metagene 0: T cells, 1: B cells, both loaded heavily on ACTB/MALAT1; 2: unused
    H = np.zeros((3, len(GENES)))
    H[0, [0, 1, 2]] = [5, 4, 3]
    H[1, [3, 4]] = [6, 5]
    H[:2, [5, 6]] = 10
    return H


class FakeBackend:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses or {}

    def complete_json(self, system, user, schema):
        self.calls.append(user)
        r = self.responses.get(len(self.calls))
        if isinstance(r, Exception):
            raise r
        return r or {
            "description": "T cell markers CD3E and CD3D.",
            "key_genes": ["CD3E", "CD3D", "NOTSHOWN"],
            "category": "cell_type",
            "cell_types": ["T cells"],
            "confidence": "high",
            "label": "T cell markers.",
        }


def test_summary_specificity_and_order():
    s = summarize(GENES, make_H())
    assert [g.gene for g in s[0].top[:2]] == ["ACTB", "MALAT1"]
    by_gene = {g.gene: g for g in s[0].top}
    assert by_gene["CD3E"].specificity == pytest.approx(1.0)
    assert by_gene["ACTB"].specificity == pytest.approx(0.5, abs=0.1)
    assert s[2].top == []


def test_usage_rescales_specificity():
    H = make_H()
    # Rescaling a factor's H row (with W compensating) must not change specificity once usage is given.
    H2 = H.copy()
    H2[0] *= 100
    usage = H.sum(axis=1)
    a = summarize(GENES, H, usage)
    b = summarize(GENES, H2, usage / 100)
    assert [g.specificity for g in a[0].top] == pytest.approx([g.specificity for g in b[0].top])


def test_transposed_weights():
    _, a = build_prompts(GENES, make_H())
    _, b = build_prompts(GENES, make_H().T)
    assert a == b


def test_describe():
    backend = FakeBackend()
    out = describe_metagenes(GENES, make_H(), backend=backend, context="Human PBMC")
    assert len(out) == 3 and len(backend.calls) == 2  # all-zero metagene isn't sent
    assert out[0].label == "T cell markers"
    assert out[0].key_genes == ["CD3E", "CD3D"]
    assert out[2].category == "inactive"
    assert "Human PBMC" in backend.calls[0]
    assert out[1].gene_label.startswith("MS4A1")


def test_failures_fall_back_to_gene_label():
    backend = FakeBackend({1: BackendError("bad json"), 2: BackendError("bad json")})
    out = describe_metagenes(GENES, make_H(), backend=backend, metagenes=[0], max_attempts=2)
    assert out[0].error == "bad json" and out[0].label == out[0].gene_label


def test_retry_then_success():
    backend = FakeBackend({1: BackendError("bad json")})
    out = describe_metagenes(GENES, make_H(), backend=backend, metagenes=[0])
    assert out[0].error is None and len(backend.calls) == 2


def test_notes_and_low_usage():
    usage = np.array([1000.0, 1000.0, 0.1])
    backend = FakeBackend()
    out = describe_metagenes(
        GENES, make_H(), usage=usage, backend=backend, notes={1: "mostly from sample P3"}
    )
    assert out[2].category == "inactive"
    assert "mostly from sample P3" in backend.calls[1]


def test_parse_json_fenced():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}

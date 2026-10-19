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

    def complete_json(self, system, user, schema, tools=()):
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


class RefineBackend(FakeBackend):
    def __init__(self, groups, labels):
        super().__init__()
        self.groups, self.labels = groups, labels
        self.group_calls, self.resolve_calls = [], []

    def complete_json(self, system, user, schema, tools=()):
        if "groups" in schema["properties"]:
            self.group_calls.append(user)
            return {"groups": self.groups}
        if "labels" in schema["properties"]:
            self.resolve_calls.append(user)
            return {"labels": self.labels}
        return super().complete_json(system, user, schema)


def _desc(i, label, category="cell_type"):
    from metagene_describer import MetageneDescription

    return MetageneDescription(i, label, "", category, "high")


def test_refine():
    from metagene_describer import refine_labels

    H = np.vstack([make_H(), make_H()[:1]])  # metagene 3 duplicates the T cell metagene
    descs = [
        _desc(0, "T cell markers"),
        _desc(1, "B cell markers"),
        _desc(2, "X/Y-high", "ambiguous"),
        _desc(3, "T-cell markers"),
    ]
    backend = RefineBackend(
        groups=[{"concept": "T cells", "indices": [0, 3, 2, 99]}],
        labels=[
            {"index": 0, "reason": "same", "label": "T cell markers."},
            {"index": 3, "reason": "same", "label": "T cell markers"},
            {"index": 1, "reason": "not in group", "label": "nope"},
        ],
    )
    out = refine_labels(descs, GENES, H, backend=backend)
    assert [o.label for o in out] == [
        "T cell markers",
        "B cell markers",
        "X/Y-high",
        "T cell markers",
    ]
    assert out[3].initial_label == "T-cell markers" and out[0].initial_label is None
    assert "X/Y-high" not in backend.group_calls[0]  # ambiguous metagenes aren't sent
    assert len(backend.resolve_calls) == 1
    prompt = backend.resolve_calls[0]
    assert "Metagene 0" in prompt and "Metagene 3" in prompt and "Metagene 2" not in prompt
    assert "distinguishing genes: none" in prompt  # identical weights
    assert "1.00\t1.00" in prompt  # cosine similarity


def test_exact_duplicates_always_grouped():
    from metagene_describer.refine import _parse_groups

    descs = [_desc(0, "A"), _desc(1, "a"), _desc(2, "B"), _desc(3, "C")]
    groups = _parse_groups({"groups": [{"concept": "BC", "indices": [2, 3]}]}, descs)
    assert sorted(sorted(g.indices) for g in groups) == [[0, 1], [2, 3]]
    groups = _parse_groups({"groups": [{"concept": "AB", "indices": [1, 2]}]}, descs)
    assert [sorted(g.indices) for g in groups] == [[0, 1, 2]]


def test_distinguishing_genes():
    from metagene_describer.refine import _Group, resolve_prompt
    from metagene_describer.summary import gene_specificity, summarize

    H = make_H()
    H[1, 0] = 0.1  # B cell metagene has a little CD3E
    summaries = {s.index: s for s in summarize(GENES, H)}
    descs = {0: _desc(0, "Lymphocytes"), 1: _desc(1, "Lymphocytes")}
    W = H[:2] / H[:2].sum(axis=1, keepdims=True)
    spec = gene_specificity(H[:2])
    prompt = resolve_prompt(_Group("lymphocytes", [0, 1]), descs, summaries, GENES, W, spec)
    t_line = prompt.split("Metagene 0")[1].split("\n")[2]
    b_line = prompt.split("Metagene 1")[1].split("\n")[2]
    assert t_line.startswith("  distinguishing genes: CD3E") and "MS4A1" not in t_line
    assert "MS4A1" in b_line and "ACTB" not in b_line


def test_describe_refine_failure_keeps_results(capsys):
    class Broken(FakeBackend):
        def complete_json(self, system, user, schema, tools=()):
            if "groups" in schema["properties"]:
                raise BackendError("bad json")
            return super().complete_json(system, user, schema)

    out = describe_metagenes(GENES, make_H(), backend=Broken(), refine=True)
    assert out[0].label == "T cell markers"
    assert "refinement failed" in capsys.readouterr().err

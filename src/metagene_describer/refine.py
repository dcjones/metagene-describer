"""Joint label refinement: reconcile near-duplicate labels and differentiate duplicates.

Two stages. First, one request sees only the labels and groups those that name the same concept
(exact duplicates are always grouped). Then, for each group, a request sees each member's top genes,
the genes that distinguish it from the other members (computed here, not by the model), and how
similar the members' weights are, and settles on either one shared wording or labels that state
real differences.
"""

from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

from .backends import Backend, BackendError
from .summary import MetageneSummary, gene_specificity

if TYPE_CHECKING:
    from .describe import MetageneDescription

# Gene-based labels are already distinct, and inactive or failed metagenes have nothing to refine.
SKIP_CATEGORIES = {"ambiguous", "sample_specific", "inactive"}

GROUP_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string"},
                    "indices": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["concept", "indices"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["groups"],
    "additionalProperties": False,
}

RESOLVE_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "reason": {"type": "string"},
                    "label": {"type": "string"},
                },
                "required": ["index", "reason", "label"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}

GROUP_SYSTEM_PROMPT = """\
You are reviewing labels for the metagenes (factors) of one non-negative matrix factorization of \
transcriptomics data. Each label was written for one metagene without seeing the others.

Find groups of labels that name the same concept, so that a reader could not tell from the labels \
alone how the metagenes differ. This includes labels that are worded differently but mean the same \
thing, e.g. "Immediate early stress genes", "Immediate early stress response" and "Immediate early \
response"; or "Cycling cells" and "Cell cycle proliferation program". Identical labels always \
belong to the same group.

Do not group labels that name related but distinct concepts: "CD4 T cell markers", "Naive T cell \
markers" and "Regulatory T cell markers" are different labels for different things, and "S-phase \
DNA replication" and "G2/M cell cycle" are different phases.

Each metagene may be in at most one group, and each group needs at least 2 metagenes. Give each \
group a short name for its concept. Return an empty list if there are no such groups.
"""

RESOLVE_SYSTEM_PROMPT = """\
You are an expert in single-cell and spatial transcriptomics. The metagenes (factors) below, from \
one non-negative matrix factorization, have labels that name the same concept. Decide whether they \
differ in a way worth putting in their labels, and label them accordingly.

For each metagene you are given its current label and:
- top genes: its most informative genes, with specificity (the share of the gene's modeled \
expression this metagene accounts for, across all metagenes).
- distinguishing genes: genes weighted much more in this metagene than in the other members of \
the group, with the share of the group's weight on that gene that falls in this metagene (1.00 \
means no other member uses it).
You are also given the cosine similarity between the members' gene weights.

Rules:
- Keep each metagene's identity: the cell type, lineage or program its current label names. You \
may add a qualifier (a subtype, state, isotype or added program), reword the label to match the \
group's shared wording, or add distinguishing genes. Replace or narrow the identity itself (e.g. \
"Epithelial markers" to "Mesothelial cells", or "B cell markers" to "Plasma cells") only when the \
canonical markers of the new identity are among that metagene's top or distinguishing genes, and \
name those markers in the reason. A canonical marker that isn't listed is evidence against the \
narrower identity, not for it.
- If the distinguishing genes show a meaningful biological difference, put it in the label. \
Prefer a biological description over gene names when one fits: a lineage ("Monocyte immediate \
early response" vs "T cell immediate early response"), a subtype ("Naive CD8 T cells"), an \
isotype (IgA vs IgG), or an added program ("Naive T cells, interferon-high").
- If the difference is real but has no good biological name, add the 1-2 most distinguishing \
genes in parentheses, e.g. "Naive T cell markers (SOX4)". Use only genes from that metagene's \
distinguishing genes for this, never just its top gene.
- If members are highly similar or their distinguishing genes are weak or uninformative, give \
them the same label. Broadly expressed genes (housekeeping, cytoskeletal, ribosomal, \
mitochondrial, ferritin, TMSB4X, ACTB and the like) are not a meaningful difference, even when \
they are listed as distinguishing. Similar metagenes with somewhat different weightings are \
expected in NMF, and an invented distinction is worse than a repeated label.
- Whether you differentiate or not, use one consistent wording for the shared concept across all \
members.
- Keep labels to at most 6 words, plus an optional parenthetical. Don't use the word "metagene" \
and don't end with a period.

Return a label for every metagene in the group, each with its index and a brief reason.
"""


@dataclasses.dataclass
class _Group:
    concept: str
    indices: list[int]


def group_prompt(descriptions: list[MetageneDescription]) -> str:
    lines = ["Metagenes (index [category] label):"]
    lines += [f"{d.index} [{d.category}] {d.label}" for d in descriptions]
    return "\n".join(lines)


def _parse_groups(data: dict, descriptions: list[MetageneDescription]) -> list[_Group]:
    allowed = {d.index for d in descriptions}
    groups, used = [], set()
    for g in data.get("groups") or []:
        if not isinstance(g, dict):
            continue
        idx = [i for i in g.get("indices") or [] if i in allowed and i not in used]
        idx = list(dict.fromkeys(idx))
        if len(idx) >= 2:
            groups.append(_Group(str(g.get("concept") or ""), idx))
            used.update(idx)

    # Exact duplicates are always grouped, whatever the model returned: merge them into the group
    # that already holds one of them, or start a new group.
    by_label: dict[str, list[int]] = {}
    for d in descriptions:
        by_label.setdefault(d.label.strip().lower(), []).append(d.index)
    for label, idx in by_label.items():
        if len(idx) < 2:
            continue
        home = next((g for g in groups if set(g.indices) & set(idx)), None)
        if home is None:
            groups.append(_Group(label, [i for i in idx if i not in used]))
        else:
            home.indices += [i for i in idx if i not in used]
        used.update(idx)
    return [g for g in groups if len(g.indices) >= 2]


def resolve_prompt(
    group: _Group,
    descriptions: dict[int, MetageneDescription],
    summaries: dict[int, MetageneSummary],
    genes: list[str],
    weights: np.ndarray,
    specificity: np.ndarray,
    context: str | None = None,
    n_top: int = 12,
    n_distinct: int = 8,
) -> str:
    """weights: row-normalized [k, n_genes], so rows are comparable within the group.
    specificity: [k, n_genes], each gene's share of modeled expression per metagene."""
    idx = group.indices
    Wg = weights[idx]  # [members, n_genes]

    lines = []
    if context:
        lines += ["Dataset context:", context.strip(), ""]
    lines.append(f"Shared concept: {group.concept}")
    lines.append("")
    for row, i in enumerate(idx):
        d, s = descriptions[i], summaries[i]
        top = sorted(s.top + s.specific, key=lambda g: -g.weight * g.specificity)[:n_top]
        w = Wg[row]
        others = np.delete(Wg, row, axis=0).max(axis=0)
        share = np.divide(w, w + others, out=np.zeros_like(w), where=(w + others) > 0)
        # Genes this member uses a lot and the other members don't. Weighting by specificity
        # across all metagenes keeps ubiquitous genes, whose weights differ only by noise, out.
        score = w * share * specificity[i]
        order = np.argsort(-score)[:n_distinct]
        distinct = [(genes[j], share[j]) for j in order if share[j] >= 0.6 and w[j] > 0]
        lines.append(f"Metagene {i} [{d.category}, {d.confidence}] {d.label}")
        lines.append("  top genes: " + ", ".join(f"{g.gene} {g.specificity:.2f}" for g in top))
        lines.append(
            "  distinguishing genes: "
            + (", ".join(f"{g} {x:.2f}" for g, x in distinct) if distinct else "none")
        )
        lines.append("")

    norms = np.linalg.norm(Wg, axis=1, keepdims=True)
    unit = np.divide(Wg, norms, out=np.zeros_like(Wg), where=norms > 0)
    sim = unit @ unit.T
    lines.append("Cosine similarity of gene weights:")
    lines.append("\t" + "\t".join(str(i) for i in idx))
    for row, i in enumerate(idx):
        lines.append(f"{i}\t" + "\t".join(f"{x:.2f}" for x in sim[row]))
    return "\n".join(lines)


def refine(
    descriptions: list[MetageneDescription],
    summaries: dict[int, MetageneSummary],
    genes: list[str],
    weights: np.ndarray,
    usage: np.ndarray | None,
    backend: Backend,
    context: str | None = None,
    max_workers: int = 8,
) -> list[MetageneDescription]:
    """Apply both stages, returning new descriptions.

    weights is H as [k, n_genes], and usage the per-metagene usage [k] or None. Raises BackendError if the grouping request fails; a failed
    group request leaves that group's labels unchanged.
    """
    eligible = [d for d in descriptions if d.category not in SKIP_CATEGORIES and d.error is None]
    if len(eligible) < 2:
        return list(descriptions)

    data = backend.complete_json(GROUP_SYSTEM_PROMPT, group_prompt(eligible), GROUP_SCHEMA)
    if not isinstance(data.get("groups"), list):
        raise BackendError("grouping response has no list of groups")
    groups = _parse_groups(data, eligible)

    row_sums = weights.sum(axis=1, keepdims=True)
    normalized = np.divide(weights, row_sums, out=np.zeros_like(weights), where=row_sums > 0)
    specificity = gene_specificity(weights, usage)
    by_index = {d.index: d for d in descriptions}

    def resolve(group: _Group) -> dict[int, tuple[str, str]]:
        prompt = resolve_prompt(group, by_index, summaries, genes, normalized, specificity, context)
        try:
            data = backend.complete_json(RESOLVE_SYSTEM_PROMPT, prompt, RESOLVE_SCHEMA)
        except Exception:
            return {}
        out = {}
        for item in data.get("labels") or []:
            if not isinstance(item, dict):
                continue
            i = item.get("index")
            label = str(item.get("label") or "").strip().rstrip(".")
            if i in group.indices and label:
                out[i] = (label, str(item.get("reason") or "").strip())
        return out

    new_labels: dict[int, tuple[str, str]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for result in pool.map(resolve, groups):
            new_labels.update(result)

    out = []
    for d in descriptions:
        if d.index in new_labels and new_labels[d.index][0] != d.label:
            label, reason = new_labels[d.index]
            d = dataclasses.replace(d, label=label, initial_label=d.label, refine_reason=reason)
        out.append(d)
    return out

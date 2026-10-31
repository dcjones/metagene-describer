"""Cluster annotation from metagene composition.

Each cell's modeled expression is a sum over metagenes, W[i, k] * H[k]. Rescaled by H's row sums,
W[i, k] * H[k].sum() is the modeled count that metagene k contributes to cell i, which is comparable
across metagenes. A cluster is summarized by each metagene's share of its modeled counts, how
enriched that share is relative to all other cells, and the fraction of its cells in which the
metagene is a sizable part of the cell. The model labels the cluster from these, the metagenes'
descriptions, and the cluster's top modeled genes.
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .backends import Backend, BackendError, make_backend
from .describe import MetageneDescription
from .prompt import CONFIDENCES
from .summary import as_metagene_by_gene

__all__ = ["ClusterDescription", "describe_clusters", "build_cluster_prompts", "summarize_clusters"]

# A metagene is "present" in a cell when it accounts for at least this share of the cell's counts.
PRESENT = 0.1
# Metagenes below this share of a cluster's counts are not listed among its largest.
MIN_SHARE = 0.005
# Further metagenes and top genes must be at least this enriched over the other cells.
MIN_FOLD = 2.0

CATEGORIES = ["cell_type", "cell_state", "mixed", "technical", "ambiguous"]

# Field order matters for structured output: the label is generated last, conditioned on the rest.
SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "key_metagenes": {"type": "array", "items": {"type": "integer"}},
        "contaminants": {"type": "array", "items": {"type": "string"}},
        "category": {"type": "string", "enum": CATEGORIES},
        "cell_types": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": CONFIDENCES},
        "label": {"type": "string"},
    },
    "required": [
        "description",
        "key_metagenes",
        "contaminants",
        "category",
        "cell_types",
        "confidence",
        "label",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You are an expert in single-cell and spatial transcriptomics. You annotate clusters of cells (e.g. \
from Leiden clustering) using a non-negative matrix factorization (NMF) of the cells x genes count \
matrix. Each cell's expression is modeled as a sum of metagenes (factors), and each metagene has \
already been annotated from its genes. Your annotations become the cell type labels of the \
dataset, so they must be accurate and calibrated, not merely plausible.

For one cluster you are given its size and its metagene composition. For each listed metagene:
- share: the share of the cluster's modeled counts that comes from the metagene. Shares sum to \
100% over all metagenes; only the largest and the most enriched are listed.
- cells: the fraction of the cluster's cells in which the metagene accounts for at least 10% of the \
cell's modeled counts. A metagene with a modest share in most cells is part of these cells' \
expression, or diffuse contamination of all of them. One with a large share in a minority of cells \
marks a subpopulation, such as doublets or a distinct subset.
- fold: the metagene's share in this cluster divided by its share in all other cells.
- the cluster where the metagene has its largest share, when that is a different cluster.
- its annotation: label, category (cell_type, cell_state, program, technical, sample_specific, \
ambiguous), confidence, key genes and description.
You are also given the cluster's top modeled genes, ranked by expression and enrichment over the \
other cells, with each gene's share of the cluster's modeled counts and its fold over the rest. Use \
them to check the metagene annotations, which can be wrong or too broad.

How to read the composition:
- Identity comes from cell_type metagenes, and from cell_state metagenes that name a lineage. \
cell_state and program metagenes (cycling, interferon response, stress, hypoxia) can add a \
qualifier to an identity, e.g. "Cycling CD8 T cells", but don't define one. A program alone does \
not make a cell type.
- technical metagenes (mitochondrial, ribosomal, single-gene, diffuse background) say nothing \
about identity. Look past them. If they dominate the cluster and no lineage is clear, the cluster \
is probably low quality: use category technical.
- sample_specific metagenes usually mark malignant cells or another clone-like population confined \
to one sample. In a tumor dataset, a cluster dominated by them is usually malignant cells.
- NMF often splits a lineage into a core metagene and add-on metagenes (subsets, states). Several \
metagenes of the same lineage in one cluster are expected and are not a mixture.
- Related lineages that share markers (T and NK cells, monocytes and macrophages, the epithelial \
subtypes of one tissue) are not contamination of each other. Decide which one the cluster is, or \
name both if they can't be told apart.

Mixtures and contamination:
Clusters often carry signal from a second, unrelated lineage: ambient RNA, doublets, or, in spatial \
data, transcripts from neighboring cells assigned to the wrong cell by segmentation. Decide which \
lineage is the cluster's own (usually the largest share, enriched, and present in most cells), \
then how much foreign signal there is:
- A foreign lineage with a small share (roughly under 5%) is background. Ignore it, or mention it \
in the description.
- A larger foreign share (roughly 5-20%), especially from a metagene whose largest share is in \
another cluster, is contamination worth noting. List the lineage in contaminants and mention it in \
the description, but keep the label on the cluster's own identity.
- When the foreign signal is large (roughly over 20%), also note it in the label, after the \
identity, e.g. "Fibroblasts (+ macrophage signal)".
- When two lineages contribute comparably and neither is clearly the cluster's own, use category \
mixed and name both, e.g. "Macrophage/fibroblast mixture". Doublets usually show the second lineage \
strongly in a fraction of the cells; segmentation spillover usually shows it weakly in most cells.
These thresholds are guides, not rules. Weigh how lineage-specific the foreign metagene is, how \
many cells carry it, and the dataset context.

Rules:
- Be calibrated. Name a subtype or state only when a metagene or the top genes support it. \
Otherwise give the broader lineage, e.g. "Myeloid cells" rather than "M2 macrophages". Set \
confidence to high only for clear signatures.
- A cluster label is a call about the majority of the cluster's cells, not a guarantee about every \
cell. When the evidence leans clearly toward a subset, put the subset in the label, and use \
confidence to say how sure you are. The label should be as specific as your description: if the \
description says the cluster leans toward a subset, so should the label.
- When two related identities are possible and the evidence favors one (its lineage markers are \
among the top genes or in an enriched metagene, and the other's are not), name the favored one. \
Name both, e.g. "A/B cells", only when they are about equally supported.
- When metagenes for two subsets or states of the same lineage are both present, decide between \
them by their relative share, enrichment and how many cells carry them, not by whether each is \
present at all. Markers shared between related states don't decide it.
- Many clusters in a dataset will share an identity; that is expected. Use standard cell type \
names, so the same identity gets the same wording in every cluster.
- If no lineage is supported, use category ambiguous and label the cluster by the labels of its \
1-2 largest informative metagenes, joined by " + ". Don't invent a cell type.
- Take the dataset context into account. Expect cell types that belong in the tissue.

Output fields:
- description: 1-3 sentences on what the cluster is, citing the metagenes (as M<index>) and genes \
that support it, and any contamination.
- key_metagenes: the indices of the 1-5 metagenes the call rests on.
- contaminants: lineages present as contamination, doublets or spillover. Empty if none.
- category: cell_type (a cell type or lineage), cell_state (a cell type with a state or subtype \
qualifier), mixed, technical, or ambiguous.
- cell_types: candidate cell types, most likely first.
- confidence: high, medium, or low.
- label: at most 6 words, plus an optional parenthetical. Don't use the word "cluster" and don't end \
with a period.
"""

REFINE_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cluster": {"type": "string"},
                    "reason": {"type": "string"},
                    "label": {"type": "string"},
                },
                "required": ["cluster", "reason", "label"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}

REFINE_SYSTEM_PROMPT = """\
You are reviewing cell type labels for the clusters of one dataset. Each label was written for one \
cluster without seeing the others, so the same identity may be worded differently in different \
clusters.

For each cluster you are given its size, its current category and label, and its largest \
metagenes (NMF factors), with their share of the cluster's modeled counts and their labels.

Revise the labels so that:
- Clusters with the same identity use exactly the same wording, e.g. not "CD8 T cells" in one \
cluster and "CD8+ T cells" or "Cytotoxic T cells" in another, unless the clusters really differ. \
Repeated labels are expected and fine: clustering often splits one cell type into several clusters.
- Clusters that share an identity but differ in a way their composition clearly shows (a state or \
program metagene with a much larger share in one, or a different subtype metagene) get a \
qualifier stating it, e.g. "Fibroblasts" and "Fibroblasts, interferon-high". Small differences in \
shares are not a difference.
- Each cluster keeps its identity, its contamination note if it has one, and, for ambiguous \
clusters, its metagene-based label. Change the wording and the qualifiers, not the annotation.
- Never make a label broader or more hedged to match another cluster. If one cluster names a \
single identity and another with a similar composition names two alternatives, leave both as they \
are.

Return only the clusters whose label you change, each with its cluster id as given, a brief \
reason, and the new label. Return an empty list if no label needs to change.
"""


@dataclass
class MetageneShare:
    index: int
    share: float  # share of the cluster's modeled counts from this metagene
    fold: float | None  # share here / share in all other cells; None with a single cluster
    prevalence: float  # fraction of the cluster's cells where the metagene is >= PRESENT of the cell
    peak: tuple[object, float] | None  # (cluster, share) where the metagene's share is largest,
    # if that is another cluster


@dataclass
class ClusterSummary:
    position: int  # index among the sorted cluster ids
    cluster: object
    n_cells: int
    cell_share: float  # fraction of all cells
    metagenes: list[MetageneShare]  # largest by share, then the most enriched of the rest
    genes: list[tuple[str, float, float | None]]  # (gene, share of modeled counts, fold over rest)


@dataclass
class ClusterDescription:
    cluster: object
    n_cells: int
    label: str
    description: str
    category: str  # one of CATEGORIES
    confidence: str
    cell_types: list[str] = field(default_factory=list)
    key_metagenes: list[int] = field(default_factory=list)
    contaminants: list[str] = field(default_factory=list)
    # The labels of the largest informative metagenes, e.g. "T cell markers + Cycling cells";
    # computed locally, never by the LLM.
    metagene_label: str = ""
    top_metagenes: list[int] = field(default_factory=list)  # the metagenes the model was shown
    error: str | None = None  # set when the LLM call failed; label then falls back to metagene_label
    initial_label: str | None = None  # the label before refinement, if refinement changed it
    refine_reason: str | None = None  # why refinement changed the label


def _rows(X: np.ndarray) -> np.ndarray:
    s = X.sum(axis=1, keepdims=True)
    return np.divide(X, s, out=np.zeros_like(X), where=s > 0)


def _fold(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / b, np.where(a > 0, np.inf, 1.0))


def _enrichment(share: np.ndarray, fold: np.ndarray | None) -> np.ndarray:
    if fold is None:
        return share
    return share * np.log2(np.clip(fold, 1.0, 1e3))


def summarize_clusters(
    clusters,
    genes: Sequence[str],
    H: np.ndarray,
    W: np.ndarray,
    *,
    n_metagenes: int = 10,
    n_enriched: int = 5,
    n_genes: int = 20,
) -> list[ClusterSummary]:
    """Per-cluster metagene composition and top modeled genes, from per-cell cluster labels,
    H [k, n_genes] and W [cells, k]. Clusters are in sorted order of their labels."""
    clusters = np.asarray(clusters)
    W = np.asarray(W, dtype=np.float64)
    k = H.shape[0]
    if W.shape != (len(clusters), k):
        raise ValueError(f"usage has shape {W.shape}, expected [{len(clusters)} cells, {k}]")
    if np.any(W < 0) or not np.all(np.isfinite(W)):
        raise ValueError("usage must be finite and non-negative")
    ids, codes = np.unique(clusters, return_inverse=True)
    ids = ids.tolist()
    m = len(ids)
    # Leiden labels are numeric strings; order them as numbers, not as text ("2" before "10").
    if all(isinstance(i, str) and i.isdigit() for i in ids):
        perm = sorted(range(m), key=lambda i: int(ids[i]))
        rank = np.empty(m, dtype=np.intp)
        rank[perm] = np.arange(m)
        ids, codes = [ids[i] for i in perm], rank[codes]

    scale = H.sum(axis=1)
    counts = np.zeros((m, k))  # modeled counts per cluster and metagene
    prevalence = np.zeros((m, k))
    sizes = np.bincount(codes, minlength=m)
    for c in range(m):
        Ec = W[codes == c] * scale
        counts[c] = Ec.sum(axis=0)
        cell = Ec.sum(axis=1, keepdims=True)
        frac = np.divide(Ec, cell, out=np.zeros_like(Ec), where=cell > 0)
        prevalence[c] = (frac >= PRESENT).mean(axis=0)

    share = _rows(counts)
    fold = _fold(share, _rows(counts.sum(axis=0) - counts)) if m > 1 else None
    peak = share.argmax(axis=0)

    gcounts = counts @ _rows(H)  # modeled counts per cluster and gene
    gshare = _rows(gcounts)
    gfold = _fold(gshare, _rows(gcounts.sum(axis=0) - gcounts)) if m > 1 else None

    mg_score = _enrichment(share, fold)
    gene_score = _enrichment(gshare, gfold)
    if gfold is not None:
        gene_score[gfold < MIN_FOLD] = 0.0  # broadly expressed genes aren't markers
    out = []
    for c in range(m):
        order = np.argsort(-share[c], kind="stable")[:n_metagenes]
        top = [int(j) for j in order if share[c, j] >= MIN_SHARE]
        rest = mg_score[c].copy()
        rest[top] = -1.0
        if fold is not None:
            rest[fold[c] < MIN_FOLD] = -1.0
        rest[share[c] < MIN_SHARE / 5] = -1.0
        enriched = [int(j) for j in np.argsort(-rest, kind="stable")[:n_enriched] if rest[j] > 0]

        def mshare(j: int) -> MetageneShare:
            p = int(peak[j])
            return MetageneShare(
                index=j,
                share=float(share[c, j]),
                fold=float(fold[c, j]) if fold is not None else None,
                prevalence=float(prevalence[c, j]),
                peak=(ids[p], float(share[p, j])) if p != c else None,
            )

        g_order = [j for j in np.argsort(-gene_score[c], kind="stable")[:n_genes]
                   if gene_score[c, j] > 0]
        out.append(
            ClusterSummary(
                position=c,
                cluster=ids[c],
                n_cells=int(sizes[c]),
                cell_share=float(sizes[c] / len(clusters)),
                metagenes=[mshare(j) for j in top + enriched],
                genes=[
                    (str(genes[j]), float(gshare[c, j]),
                     float(gfold[c, j]) if gfold is not None else None)
                    for j in g_order
                ],
            )
        )
    return out


def _pct(x: float, digits: int = 1) -> str:
    floor = 10.0**-digits
    return f"{100 * x:.{digits}f}%" if 100 * x >= floor else f"<{floor:g}%"


def _fmt_fold(x: float | None) -> str:
    if x is None:
        return "n/a"
    return ">100x" if x > 100 else f"{x:.1f}x"


def cluster_prompt(
    summary: ClusterSummary,
    descriptions: Mapping[int, MetageneDescription],
    *,
    n_clusters: int,
    n_metagenes: int,
    context: str | None = None,
    note: str | None = None,
) -> str:
    lines = []
    if context:
        lines += ["Dataset context:", context.strip(), ""]
    lines.append(f"Clusters: {n_clusters}. Metagenes in the model: {n_metagenes}.")
    lines.append("")
    lines.append(f"Cluster {summary.cluster}")
    lines.append(f"Cells: {summary.n_cells} ({_pct(summary.cell_share)} of all cells)")
    if note:
        lines.append(f"Analyst note: {note.strip()}")
    lines.append("")
    lines.append("Metagenes (largest share first, then the most enriched of the rest):")
    for s in summary.metagenes:
        d = descriptions.get(s.index)
        head = (
            f"M{s.index}  share {_pct(s.share)}, in {_pct(s.prevalence)} of cells, "
            f"fold {_fmt_fold(s.fold)}"
        )
        if d is None:
            lines.append(f"{head}  (not annotated)")
        else:
            lines.append(f'{head}  "{d.label}" [{d.category}, {d.confidence}]')
            genes = d.key_genes or d.top_genes[:8]
            if genes:
                lines.append(f"    key genes: {', '.join(genes)}")
            if d.description:
                lines.append(f"    {d.description}")
        if s.peak is not None:
            lines.append(f"    largest share in cluster {s.peak[0]} ({_pct(s.peak[1])})")
    if summary.genes:
        lines.append("")
        lines.append("Top modeled genes (gene, share of the cluster's counts, fold over rest):")
        for g, x, f in summary.genes:
            lines.append(f"{g}\t{_pct(x, 2)}\t{_fmt_fold(f)}")
    return "\n".join(lines)


def _metagene_label(summary: ClusterSummary, descriptions: Mapping[int, MetageneDescription]) -> str:
    informative = [
        descriptions[s.index].label
        for s in summary.metagenes
        if s.index in descriptions
        and descriptions[s.index].category not in ("technical", "inactive")
    ]
    if not informative:
        informative = [descriptions[s.index].label if s.index in descriptions else f"M{s.index}"
                       for s in summary.metagenes]
    return " + ".join(dict.fromkeys(informative[:2])) or "Unannotated"


def _prepare(clusters, descriptions, genes, weights, usage, n_metagenes, n_enriched, n_genes, notes):
    genes = [str(g) for g in genes]
    H = as_metagene_by_gene(weights, len(genes))
    summaries = summarize_clusters(
        clusters, genes, H, usage, n_metagenes=n_metagenes, n_enriched=n_enriched, n_genes=n_genes
    )
    by_index = {d.index: d for d in descriptions}
    note_map = dict(notes) if notes is not None else {}
    return H, summaries, by_index, note_map


def build_cluster_prompts(
    clusters,
    descriptions: Sequence[MetageneDescription],
    genes: Sequence[str],
    weights,
    usage,
    *,
    context: str | None = None,
    notes: Mapping[object, str] | None = None,
    n_metagenes: int = 10,
    n_enriched: int = 5,
    n_genes: int = 20,
) -> tuple[str, list[str]]:
    """The system prompt and per-cluster user prompts describe_clusters would send."""
    H, summaries, by_index, note_map = _prepare(
        clusters, descriptions, genes, weights, usage, n_metagenes, n_enriched, n_genes, notes
    )
    users = [
        cluster_prompt(
            s,
            by_index,
            n_clusters=len(summaries),
            n_metagenes=H.shape[0],
            context=context,
            note=note_map.get(s.cluster),
        )
        for s in summaries
    ]
    return SYSTEM_PROMPT, users


def _parse_result(
    data: dict, s: ClusterSummary, metagene_label: str, k: int
) -> ClusterDescription:
    category = data.get("category")
    confidence = data.get("confidence")
    label = str(data.get("label") or "").strip().rstrip(".") or metagene_label
    key = [int(i) for i in data.get("key_metagenes") or [] if isinstance(i, int) and 0 <= i < k]
    return ClusterDescription(
        cluster=s.cluster,
        n_cells=s.n_cells,
        label=label,
        description=str(data.get("description") or "").strip(),
        category=category if category in CATEGORIES else "ambiguous",
        confidence=confidence if confidence in CONFIDENCES else "low",
        cell_types=[str(c) for c in data.get("cell_types") or []],
        key_metagenes=list(dict.fromkeys(key)),
        contaminants=[str(c) for c in data.get("contaminants") or []],
        metagene_label=metagene_label,
        top_metagenes=[m.index for m in s.metagenes],
    )


def refine_prompt(
    results: Sequence[ClusterDescription],
    summaries: Sequence[ClusterSummary],
    descriptions: Mapping[int, MetageneDescription],
    context: str | None = None,
    n_show: int = 5,
) -> str:
    lines = []
    if context:
        lines += ["Dataset context:", context.strip(), ""]
    for r, s in zip(results, summaries):
        lines.append(f"Cluster {r.cluster} ({r.n_cells} cells) [{r.category}] {r.label}")
        for m in s.metagenes[:n_show]:
            d = descriptions.get(m.index)
            name = f'"{d.label}" [{d.category}]' if d is not None else "(not annotated)"
            lines.append(f"    M{m.index} {_pct(m.share)} {name}")
    return "\n".join(lines)


def _refine(
    results: list[ClusterDescription],
    summaries: list[ClusterSummary],
    descriptions: Mapping[int, MetageneDescription],
    backend: Backend,
    context: str | None,
) -> list[ClusterDescription]:
    """One request sees every cluster's label and composition and rewords labels for consistency.
    Raises BackendError if the response is unusable."""
    if len(results) < 2:
        return results
    prompt = refine_prompt(results, summaries, descriptions, context)
    data = backend.complete_json(REFINE_SYSTEM_PROMPT, prompt, REFINE_SCHEMA)
    if not isinstance(data.get("labels"), list):
        raise BackendError("refinement response has no list of labels")
    new_labels = {}
    for item in data["labels"]:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip().rstrip(".")
        if label:
            new_labels[str(item.get("cluster")).strip()] = (label, str(item.get("reason") or ""))
    out = []
    for r in results:
        new = new_labels.get(str(r.cluster))
        if new is not None and new[0] != r.label and r.error is None:
            r = dataclasses.replace(r, label=new[0], initial_label=r.label, refine_reason=new[1])
        out.append(r)
    return out


def describe_clusters(
    clusters,
    descriptions: Sequence[MetageneDescription],
    genes: Sequence[str],
    weights,
    usage,
    *,
    context: str | None = None,
    notes: Mapping[object, str] | None = None,
    backend: Backend | str = "anthropic",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    n_metagenes: int = 10,
    n_enriched: int = 5,
    n_genes: int = 20,
    max_workers: int = 8,
    max_attempts: int = 2,
    refine: bool = True,
    verbose: bool = False,
) -> list[ClusterDescription]:
    """Label cell clusters with an LLM, from their metagene composition and the metagenes'
    descriptions.

    Parameters
    ----------
    clusters : array-like, length n_cells
        A cluster label per cell, e.g. Leiden clusters. Any sortable labels (ints or strings).
    descriptions : sequence of MetageneDescription
        From describe_metagenes on the same factorization. Metagenes without a description are
        shown to the model as unannotated.
    genes : sequence of str, length n_genes
        Gene names, in the order of the weights' gene axis.
    weights : array-like, shape [k, n_genes] (or [n_genes, k])
        Gene weights per metagene, e.g. NMFResult.H.
    usage : array-like, shape [n_cells, k]
        Per-cell metagene usage, e.g. NMFResult.W. It is rescaled by the weights' row sums, so each
        entry is the modeled count a metagene contributes to a cell.
    context : str, optional
        Free-text description of the dataset, as for describe_metagenes.
    notes : mapping from cluster label to str, optional
        Per-cluster notes for the model, e.g. "85% of cells from patient P3".
    backend, model, api_key, base_url : see describe_metagenes.
    n_metagenes : int
        Number of metagenes with the largest share of the cluster's counts to show the model.
    n_enriched : int
        Number of further metagenes, ranked by share * log2(fold over other cells), to show.
    n_genes : int
        Number of top modeled genes to show, ranked by share * log2(fold over other cells).
    max_workers, max_attempts : see describe_metagenes.
    refine : bool
        After labeling each cluster, make one request that sees all clusters' labels and
        compositions, to word the same identity the same way and add qualifiers where clusters with
        the same identity clearly differ. If it fails, a warning is printed and the unrefined
        labels are returned.

    Returns
    -------
    list of ClusterDescription, in sorted order of cluster labels. A cluster whose request failed
    has error set, label equal to its metagene_label, and category "ambiguous".
    """
    H, summaries, by_index, note_map = _prepare(
        clusters, descriptions, genes, weights, usage, n_metagenes, n_enriched, n_genes, notes
    )
    if isinstance(backend, str):
        backend = make_backend(backend, model=model, api_key=api_key, base_url=base_url)
    k = H.shape[0]

    def run(s: ClusterSummary) -> ClusterDescription:
        metagene_label = _metagene_label(s, by_index)
        prompt = cluster_prompt(
            s,
            by_index,
            n_clusters=len(summaries),
            n_metagenes=k,
            context=context,
            note=note_map.get(s.cluster),
        )
        result, error = None, None
        for _ in range(max_attempts):
            try:
                data = backend.complete_json(SYSTEM_PROMPT, prompt, SCHEMA)
                result = _parse_result(data, s, metagene_label, k)
                break
            except BackendError as e:
                error = str(e)
            except Exception as e:  # API errors after the SDK's own retries; don't retry again
                error = f"{type(e).__name__}: {e}"
                break
        if result is None:
            result = ClusterDescription(
                cluster=s.cluster,
                n_cells=s.n_cells,
                label=metagene_label,
                description="",
                category="ambiguous",
                confidence="low",
                metagene_label=metagene_label,
                top_metagenes=[m.index for m in s.metagenes],
                error=error,
            )
        if verbose:
            status = f"error: {result.error}" if result.error else result.label
            print(f"cluster {s.cluster}: {status}", file=sys.stderr)
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(run, summaries))
    if refine:
        try:
            results = _refine(results, summaries, by_index, backend, context)
        except Exception as e:
            print(f"warning: cluster label refinement failed: {type(e).__name__}: {e}", file=sys.stderr)
    return results

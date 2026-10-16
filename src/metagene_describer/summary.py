from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GeneStat:
    gene: str
    weight: float  # share of the metagene's total loading, in [0, 1]
    specificity: float  # share of the gene's modeled expression due to this metagene, in [0, 1]


@dataclass
class MetageneSummary:
    index: int
    top: list[GeneStat]  # highest weight
    specific: list[GeneStat]  # highest weight * specificity, excluding those in `top`
    effective_genes: float  # perplexity of the weight distribution
    usage_share: float | None  # share of total usage across metagenes, if usage was given


def as_metagene_by_gene(H, n_genes: int) -> np.ndarray:
    """Return H as a float64 [k, n_genes] array, transposing [n_genes, k] input."""
    H = np.asarray(H, dtype=np.float64)
    if H.ndim != 2:
        raise ValueError(f"weights must be 2-dimensional, got shape {H.shape}")
    if H.shape[1] != n_genes:
        if H.shape[0] == n_genes:
            H = H.T
        else:
            raise ValueError(f"weights have shape {H.shape}, but there are {n_genes} genes")
    if np.any(H < 0) or not np.all(np.isfinite(H)):
        raise ValueError("weights must be finite and non-negative")
    return H


def as_usage(usage, k: int) -> np.ndarray:
    """Per-metagene total usage from W [cells, k] (summed over cells) or a length-k vector."""
    usage = np.asarray(usage, dtype=np.float64)
    if usage.ndim == 2:
        if usage.shape[1] != k:
            raise ValueError(f"usage matrix has shape {usage.shape}, expected [cells, {k}]")
        usage = usage.sum(axis=0)
    if usage.shape != (k,):
        raise ValueError(f"usage has shape {usage.shape}, expected ({k},) or [cells, {k}]")
    return usage


def summarize(
    genes: list[str],
    H: np.ndarray,
    usage: np.ndarray | None = None,
    n_top: int = 30,
    n_specific: int = 10,
) -> list[MetageneSummary]:
    """Per-metagene gene statistics from H [k, n_genes] and optional usage [k].

    W H is invariant to rescaling a factor, so H's rows aren't comparable on their own. With usage
    (the column sums of W), usage[k] * H[k, g] is metagene k's total modeled expression of gene g,
    which makes specificity exact. Without it, rows are normalized to sum to 1, which treats every
    metagene as equally used.
    """
    k = H.shape[0]
    row_sums = H.sum(axis=1, keepdims=True)
    weights = np.divide(H, row_sums, out=np.zeros_like(H), where=row_sums > 0)

    scale = usage[:, None] if usage is not None else row_sums
    mass = weights * scale  # [k, n_genes] modeled expression attributed to each metagene
    gene_mass = mass.sum(axis=0, keepdims=True)
    specificity = np.divide(mass, gene_mass, out=np.zeros_like(mass), where=gene_mass > 0)

    usage_share = None
    if usage is not None and usage.sum() > 0:
        usage_share = usage / usage.sum()

    out = []
    for i in range(k):
        w, s = weights[i], specificity[i]
        top_idx = np.argsort(-w, kind="stable")[:n_top]
        top_idx = top_idx[w[top_idx] > 0]
        score = w * s
        score[top_idx] = -1.0
        spec_idx = np.argsort(-score, kind="stable")[:n_specific]
        spec_idx = spec_idx[score[spec_idx] > 0]

        nz = w[w > 0]
        effective = float(np.exp(-(nz * np.log(nz)).sum())) if nz.size else 0.0

        def stats(idx):
            return [GeneStat(genes[j], float(w[j]), float(s[j])) for j in idx]

        out.append(
            MetageneSummary(
                index=i,
                top=stats(top_idx),
                specific=stats(spec_idx),
                effective_genes=effective,
                usage_share=float(usage_share[i]) if usage_share is not None else None,
            )
        )
    return out

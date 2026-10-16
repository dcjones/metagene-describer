from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from .backends import Backend, BackendError, make_backend
from .prompt import CATEGORIES, CONFIDENCES, SCHEMA, SYSTEM_PROMPT, user_prompt
from .summary import MetageneSummary, as_metagene_by_gene, as_usage, summarize

__all__ = ["MetageneDescription", "describe_metagenes", "build_prompts", "to_dataframe"]


@dataclass
class MetageneDescription:
    index: int
    label: str
    description: str
    category: str  # one of prompt.CATEGORIES, or "inactive" for metagenes skipped by usage
    confidence: str
    cell_types: list[str] = field(default_factory=list)
    key_genes: list[str] = field(default_factory=list)
    gene_label: str = ""  # top genes by weight * specificity, e.g. "CD3E/CD2/CD3D"; never from the LLM
    top_genes: list[str] = field(default_factory=list)
    error: str | None = None  # set when the LLM call failed; label then falls back to gene_label


def _gene_label(s: MetageneSummary, n: int = 3) -> str:
    stats = sorted(s.top + s.specific, key=lambda g: -g.weight * g.specificity)
    return "/".join(g.gene for g in stats[:n])


def _prepare(
    genes, weights, usage, n_top, n_specific, notes
) -> tuple[list[str], list[MetageneSummary], dict[int, str | None]]:
    genes = [str(g) for g in genes]
    H = as_metagene_by_gene(weights, len(genes))
    k = H.shape[0]
    usage = as_usage(usage, k) if usage is not None else None
    summaries = summarize(genes, H, usage, n_top=n_top, n_specific=n_specific)
    if notes is None:
        note_map = {}
    elif isinstance(notes, Mapping):
        note_map = dict(notes)
    else:
        if len(notes) != k:
            raise ValueError(f"notes has {len(notes)} entries, but there are {k} metagenes")
        note_map = dict(enumerate(notes))
    return genes, summaries, note_map


def build_prompts(
    genes: Sequence[str],
    weights,
    *,
    usage=None,
    context: str | None = None,
    notes: Sequence[str | None] | Mapping[int, str] | None = None,
    n_top: int = 30,
    n_specific: int = 10,
) -> tuple[str, list[str]]:
    """The system prompt and per-metagene user prompts describe_metagenes would send."""
    genes, summaries, note_map = _prepare(genes, weights, usage, n_top, n_specific, notes)
    users = [
        user_prompt(
            s, n_genes=len(genes), n_metagenes=len(summaries), context=context, note=note_map.get(s.index)
        )
        for s in summaries
    ]
    return SYSTEM_PROMPT, users


def _parse_result(data: dict, s: MetageneSummary, gene_label: str) -> MetageneDescription:
    shown = {g.gene for g in s.top + s.specific}
    category = data.get("category")
    confidence = data.get("confidence")
    label = str(data.get("label") or "").strip().rstrip(".") or gene_label
    return MetageneDescription(
        index=s.index,
        label=label,
        description=str(data.get("description") or "").strip(),
        category=category if category in CATEGORIES else "ambiguous",
        confidence=confidence if confidence in CONFIDENCES else "low",
        cell_types=[str(c) for c in data.get("cell_types") or []],
        # Drop any gene the model named that it wasn't shown.
        key_genes=[str(g) for g in data.get("key_genes") or [] if str(g) in shown],
        gene_label=gene_label,
        top_genes=[g.gene for g in s.top],
    )


def describe_metagenes(
    genes: Sequence[str],
    weights,
    *,
    usage=None,
    context: str | None = None,
    notes: Sequence[str | None] | Mapping[int, str] | None = None,
    metagenes: Sequence[int] | None = None,
    backend: Backend | str = "anthropic",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    n_top: int = 30,
    n_specific: int = 10,
    min_usage_share: float = 1e-3,
    max_workers: int = 8,
    max_attempts: int = 2,
    verbose: bool = False,
) -> list[MetageneDescription]:
    """Label and describe each metagene with an LLM, from its top-weighted genes.

    Parameters
    ----------
    genes : sequence of str, length n_genes
        Gene names, in the order of the weights' gene axis.
    weights : array-like, shape [k, n_genes] (or [n_genes, k])
        Gene weights per metagene, e.g. NMFResult.H from metagene.nmf.
    usage : array-like, shape [cells, k] or [k], optional
        Metagene usage: W itself (e.g. NMFResult.W) or its column sums. Since W H is invariant to
        rescaling a factor, this is needed to compare a gene's loading across metagenes exactly
        (the specificity shown to the model). Without it, each metagene is treated as equally used.
        It also lets metagenes with negligible usage be skipped (see min_usage_share).
    context : str, optional
        Free-text description of the dataset, given to the model for every metagene, e.g.
        "Human cervical cancer, Xenium 5k panel, tumor and adjacent normal tissue".
    notes : sequence of str or None, or mapping from metagene index to str, optional
        Per-metagene notes for the model, e.g. "92% of usage comes from patient P3".
    metagenes : sequence of int, optional
        Describe only these metagenes (by index). The default is all of them.
    backend : Backend or {"anthropic", "openai"}
        An object with complete_json(system, user, schema) -> dict, or the name of a built-in
        backend, constructed with model, api_key and base_url. "openai" works with any
        OpenAI-compatible endpoint given base_url. For more control (effort, json_mode,
        temperature, ...) construct AnthropicBackend or OpenAIBackend directly.
    model, api_key, base_url : str, optional
        Used to construct a built-in backend when backend is a name. api_key and base_url fall back
        to the provider's environment variables (ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL or
        OPENAI_API_KEY / OPENAI_BASE_URL).
    n_top : int
        Number of top genes by weight to show the model.
    n_specific : int
        Number of further genes, ranked by weight * specificity, to show the model.
    min_usage_share : float
        With usage given, metagenes with less than this share of total usage are labeled
        "Inactive" without calling the model. Metagenes with all-zero weights always are.
    max_workers : int
        Number of concurrent requests.
    max_attempts : int
        Attempts per metagene when the response is refused, truncated or unparseable. HTTP errors
        such as rate limits are retried by the provider's SDK.

    Returns
    -------
    list of MetageneDescription, in metagene order. A metagene whose request failed has error set,
    label equal to its gene_label, and category "ambiguous".
    """
    genes, summaries, note_map = _prepare(genes, weights, usage, n_top, n_specific, notes)
    if isinstance(backend, str):
        backend = make_backend(backend, model=model, api_key=api_key, base_url=base_url)
    k = len(summaries)
    if metagenes is not None:
        summaries = [summaries[i] for i in metagenes]

    def run(s: MetageneSummary) -> MetageneDescription:
        gene_label = _gene_label(s)
        if not s.top or (s.usage_share is not None and s.usage_share < min_usage_share):
            return MetageneDescription(
                index=s.index,
                label="Inactive",
                description="Negligible usage; not sent to the model.",
                category="inactive",
                confidence="high",
                gene_label=gene_label,
                top_genes=[g.gene for g in s.top],
            )
        prompt = user_prompt(
            s, n_genes=len(genes), n_metagenes=k, context=context, note=note_map.get(s.index)
        )
        result, error = None, None
        for _ in range(max_attempts):
            try:
                data = backend.complete_json(SYSTEM_PROMPT, prompt, SCHEMA)
                result = _parse_result(data, s, gene_label)
                break
            except BackendError as e:
                error = str(e)
            except Exception as e:  # API errors after the SDK's own retries; don't retry again
                error = f"{type(e).__name__}: {e}"
                break
        if result is None:
            result = MetageneDescription(
                index=s.index,
                label=gene_label,
                description="",
                category="ambiguous",
                confidence="low",
                gene_label=gene_label,
                top_genes=[g.gene for g in s.top],
                error=error,
            )
        if verbose:
            status = f"error: {result.error}" if result.error else result.label
            print(f"metagene {s.index}: {status}", file=sys.stderr)
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(run, summaries))


def to_dataframe(descriptions: Sequence[MetageneDescription]):
    """Descriptions as a pandas DataFrame indexed by metagene, with list fields joined by ", "."""
    import pandas as pd

    rows = []
    for d in descriptions:
        row = asdict(d)
        for key in ("cell_types", "key_genes", "top_genes"):
            row[key] = ", ".join(row[key])
        rows.append(row)
    return pd.DataFrame(rows).set_index("index")

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np

from . import lookup as _lookup
from .backends import Backend, BackendError, make_backend
from .prompt import CATEGORIES, CONFIDENCES, SCHEMA, SYSTEM_PROMPT, user_prompt
from .refine import refine as _refine
from .summary import MetageneSummary, as_metagene_by_gene, as_usage, summarize

__all__ = [
    "MetageneDescription",
    "describe_metagenes",
    "refine_labels",
    "build_prompts",
    "to_dataframe",
    "from_dataframe",
]


@dataclass
class MetageneDescription:
    index: int
    label: str
    description: str
    category: str  # one of prompt.CATEGORIES, or "inactive" for metagenes skipped by usage
    confidence: str
    cell_types: list[str] = field(default_factory=list)
    key_genes: list[str] = field(default_factory=list)
    # Top genes by weight * specificity, e.g. "CD3E/CD2/CD3D"; computed locally, never by the LLM.
    gene_label: str = ""
    top_genes: list[str] = field(default_factory=list)
    error: str | None = None  # set when the LLM call failed; label then falls back to gene_label
    initial_label: str | None = None  # the label before refinement, if refinement changed it
    refine_reason: str | None = None  # why refinement changed the label
    looked_up: list[str] = field(default_factory=list)  # genes the model checked, with lookup=True


def _gene_label(s: MetageneSummary, n: int = 3) -> str:
    stats = sorted(s.top + s.specific, key=lambda g: -g.weight * g.specificity)
    return "/".join(g.gene for g in stats[:n])


def _prepare(
    genes, weights, usage, n_top, n_specific, notes
) -> tuple[list[str], np.ndarray, np.ndarray | None, list[MetageneSummary], dict[int, str | None]]:
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
    return genes, H, usage, summaries, note_map


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
    genes, _, _, summaries, note_map = _prepare(genes, weights, usage, n_top, n_specific, notes)
    users = [
        user_prompt(
            s,
            n_genes=len(genes),
            n_metagenes=len(summaries),
            context=context,
            note=note_map.get(s.index),
        )
        for s in summaries
    ]
    return SYSTEM_PROMPT, users


def _parse_result(
    data: dict, s: MetageneSummary, gene_label: str, looked_up: Sequence[str] = ()
) -> MetageneDescription:
    shown = {g.gene for g in s.top + s.specific} | set(looked_up)
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
        # Drop any gene the model named that it wasn't shown or didn't look up.
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
    lookup: bool = True,
    refine: bool = False,
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
    lookup : bool
        Give the model a lookup_genes tool, and require it to check the canonical markers of any
        cell type, state or program it names (their weight in this metagene and which metagene
        they peak in) before answering. On the example datasets this cut overreaching labels by
        about three quarters, at about twice the cost (extra round trips for roughly half the
        metagenes). The backend must support tools (both built-in backends do); set False for one
        that doesn't, or to save cost.
    refine : bool
        After describing each metagene, make one more request that sees all the labels together,
        to differentiate duplicates and reconcile near-duplicates (see refine_labels). If it fails,
        a warning is printed and the unrefined descriptions are returned.

    Returns
    -------
    list of MetageneDescription, in metagene order. A metagene whose request failed has error set,
    label equal to its gene_label, and category "ambiguous".
    """
    genes, H, usage_v, summaries, note_map = _prepare(
        genes, weights, usage, n_top, n_specific, notes
    )
    if isinstance(backend, str):
        backend = make_backend(backend, model=model, api_key=api_key, base_url=base_url)
    k = len(summaries)
    gene_lookup = _lookup.GeneLookup(genes, H, usage_v) if lookup else None
    system = SYSTEM_PROMPT + _lookup.PROMPT if lookup else SYSTEM_PROMPT
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
        looked_up: list[str] = []
        for _ in range(max_attempts):
            try:
                if gene_lookup is not None:
                    tools = [gene_lookup.tool(s.index, looked_up)]
                    data = backend.complete_json(system, prompt, SCHEMA, tools=tools)
                else:
                    data = backend.complete_json(system, prompt, SCHEMA)
                result = _parse_result(data, s, gene_label, looked_up)
                result.looked_up = list(dict.fromkeys(looked_up))
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
        results = list(pool.map(run, summaries))
    if refine:
        try:
            by_index = {s.index: s for s in summaries}
            results = _refine(results, by_index, genes, H, usage_v, backend, context, max_workers)
        except Exception as e:
            print(f"warning: label refinement failed: {type(e).__name__}: {e}", file=sys.stderr)
    return results


def refine_labels(
    descriptions: Sequence[MetageneDescription],
    genes: Sequence[str],
    weights,
    *,
    usage=None,
    context: str | None = None,
    backend: Backend | str = "anthropic",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_workers: int = 8,
) -> list[MetageneDescription]:
    """Revise labels jointly, so that labels for the same concept are worded the same, and
    metagenes that differ get labels that say how.

    Only confidently labeled metagenes take part (not ambiguous, sample_specific, inactive or
    failed ones). First, one request groups labels that name the same concept, including
    near-synonyms; identical labels are always grouped. Then, for each group, a request sees each
    member's top genes, the genes that distinguish it from the other members (computed from the
    weights), and the members' pairwise similarity. The model either gives the group one shared
    wording, or labels that state a real difference; it is told to leave labels identical when the
    differences are weak. Changed descriptions keep their old label in initial_label and the
    model's explanation in refine_reason. Descriptions are not rewritten.

    genes, weights, usage and context should be the same as were given to describe_metagenes.
    Raises BackendError if the grouping response is unusable.
    """
    genes, H, usage_v, summaries, _ = _prepare(genes, weights, usage, 30, 10, None)
    if isinstance(backend, str):
        backend = make_backend(backend, model=model, api_key=api_key, base_url=base_url)
    by_index = {s.index: s for s in summaries}
    return _refine(list(descriptions), by_index, genes, H, usage_v, backend, context, max_workers)


def to_dataframe(descriptions, sep: str = ", "):
    """Metagene or cluster descriptions as a pandas DataFrame, indexed by metagene or cluster, with
    list fields joined by sep."""
    import pandas as pd

    rows = []
    for d in descriptions:
        row = asdict(d)
        for key, value in row.items():
            if isinstance(value, list):
                row[key] = sep.join(str(v) for v in value)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.set_index("index" if "index" in df.columns else "cluster")


def from_dataframe(df, sep: str = ", "):
    """Metagene or cluster descriptions from a DataFrame made by to_dataframe with the same sep.

    The kind is taken from the index name ("index" for metagenes, "cluster" for clusters). Empty or
    missing values become None in optional fields and [] in list fields. Cluster ids come back as
    they are stored, so ids written to a file as strings stay strings.
    """
    import dataclasses

    from .clusters import ClusterDescription

    cls = ClusterDescription if df.index.name == "cluster" else MetageneDescription
    out = []
    for key, row in zip(df.index, df.to_dict("records")):
        kwargs: dict[str, object] = (
            {"cluster": key} if cls is ClusterDescription else {"index": int(key)}
        )
        for f in dataclasses.fields(cls):
            if f.name not in row:
                continue
            value = row[f.name]
            missing = value is None or (isinstance(value, float) and np.isnan(value)) or value == ""
            if f.default_factory is list:
                items = [] if missing else str(value).split(sep)
                kwargs[f.name] = [int(v) for v in items] if f.type == "list[int]" else items
            elif missing and f.default is None:
                kwargs[f.name] = None
            elif f.type == "int":
                kwargs[f.name] = int(value)
            elif f.type == "str":
                kwargs[f.name] = "" if missing else str(value)
            else:
                kwargs[f.name] = value
        out.append(cls(**kwargs))
    return out

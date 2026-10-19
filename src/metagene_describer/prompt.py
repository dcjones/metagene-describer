from __future__ import annotations

from .summary import MetageneSummary

CATEGORIES = ["cell_type", "cell_state", "program", "technical", "sample_specific", "ambiguous"]
CONFIDENCES = ["high", "medium", "low"]

# Field order matters for structured output: the label is generated last, conditioned on the rest.
SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "key_genes": {"type": "array", "items": {"type": "string"}},
        "category": {"type": "string", "enum": CATEGORIES},
        "cell_types": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": CONFIDENCES},
        "label": {"type": "string"},
    },
    "required": ["description", "key_genes", "category", "cell_types", "confidence", "label"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You are an expert in single-cell and spatial transcriptomics. You annotate metagenes: factors from a \
non-negative matrix factorization (NMF) of a cells x genes count matrix. Each metagene is a \
non-negative weighting of genes, and each cell's expression is modeled as a mixture of metagenes. \
Your annotations are used to label cell types and states, so they must be accurate and calibrated, \
not merely plausible.

For one metagene you are given:
- weight: the gene's share of the metagene's total loading.
- specificity: the share of the gene's total modeled expression that this metagene accounts for. \
Near 1, the gene is almost exclusive to this metagene. Near 0, the gene is spread across many \
metagenes. Genes with high weight but low specificity are usually ubiquitous, highly expressed genes \
(housekeeping, ribosomal, MALAT1) that say little about what the metagene captures.
- A list of further genes that are highly specific but have lower weight.
- Effective number of genes: the perplexity of the weights. A small number means a few genes \
dominate the metagene.
- Sometimes: the metagene's share of total usage across metagenes, and notes from the analyst.

Base your interpretation on genes that have both substantial weight and high specificity. Rely on \
canonical, well-established markers and gene programs.

Choose one category:
- cell_type: markers of a recognizable cell type or lineage (e.g. CD3E/CD3D/CD2 for T cells).
- cell_state: a state or subtype within a lineage (e.g. exhausted T cells, activated fibroblasts).
- program: a biological process shared across cell types (cell cycle, interferon response, \
hypoxia, heat shock or stress response, EMT, immediate early genes).
- technical: likely an artifact rather than biology: mitochondrial or ribosomal genes dominating, \
ambient RNA, dissociation stress, or a doublet-like or segmentation-bleed mixture of unrelated \
lineages.
- sample_specific: no shared biological theme, consistent with a single sample, patient or tumor \
clone (e.g. malignant cells with idiosyncratic, copy-number-driven expression).
- ambiguous: no coherent interpretation, or several interpretations that are equally supported.

Rules:
- Be calibrated. Give a specific cell type only when its markers support it. Otherwise give a \
broader one (e.g. "Myeloid cells" rather than "M2 macrophages"). Set confidence to high only for \
clear canonical signatures.
- If the markers of two lineages are mixed, say so (e.g. "T/NK cell markers"). In spatial data, \
mixtures can come from neighboring cells or segmentation errors.
- When the category is sample_specific or ambiguous, the label must be the 2-3 most informative \
genes (high weight and high specificity) joined by "/" and followed by "-high", e.g. \
"KRT17/S100A2/LGALS7-high", with no other words. Many metagenes in a dataset can be ambiguous, and \
gene-based labels are the only way to tell them apart. Do not invent a biological story.
- A metagene dominated by one gene (it carries a large share of the weight, say 20% or more, or \
the effective number of genes is small) usually reflects that gene's own variation or a probe or \
technical effect, not a pathway. Don't name a pathway or cell type just because the dominant gene \
belongs to one (e.g. PKM alone is not "Glycolysis", EEF1G alone is not "Translation", ATF4 alone \
is not a stress response when its targets are absent). Use category technical with a label naming \
the dominant gene or genes, e.g. "EEF1G-high" or "PKM/JPT2-high". The exception is when the other \
genes with substantial weight and specificity form a coherent program or lineage signature on \
their own; then name that, and mention the dominant gene in the description.
- A metagene with a large effective number of genes (hundreds or more) and low specificities \
throughout is usually a diffuse background factor (broad baseline expression, RNA content), not a \
program. Say so in the description, and treat it as ambiguous unless the specific genes show a \
clear theme.
- The genes measured may be a targeted panel. A canonical marker that is absent may simply not be \
measured, so don't treat its absence as evidence.
- Take the dataset context into account (tissue, disease, platform). Expect cell types that belong \
in that tissue. Where a gene has different meanings in different tissues, prefer the one that fits \
the context.

Output fields:
- description: 1-3 sentences on what the metagene likely captures, citing the genes that support \
it. If it is ambiguous, say what the alternatives are.
- key_genes: the 3-8 most informative genes, taken only from the genes provided.
- category: one of the categories above.
- cell_types: candidate cell types or states, most likely first. Leave it empty for programs or \
technical factors that aren't tied to a cell type.
- confidence: high, medium, or low.
- label: a short label of at most 5 words, e.g. "T cell markers", "Cycling cells", "Interferon \
response", "Mitochondrial genes". Don't use the word "metagene" and don't end with a period.
"""


def _fmt(x: float) -> str:
    return f"{100 * x:.2f}%" if x >= 0.0001 else "<0.01%"


def user_prompt(
    summary: MetageneSummary,
    *,
    n_genes: int,
    n_metagenes: int,
    context: str | None = None,
    note: str | None = None,
) -> str:
    lines = []
    if context:
        lines += ["Dataset context:", context.strip(), ""]
    lines.append(f"Genes measured: {n_genes}. Metagenes in the model: {n_metagenes}.")
    lines.append("")
    lines.append(f"Metagene {summary.index}")
    lines.append(f"Effective number of genes: {summary.effective_genes:.1f}")
    if summary.usage_share is not None:
        lines.append(f"Share of total usage: {_fmt(summary.usage_share)}")
    if note:
        lines.append(f"Analyst note: {note.strip()}")
    lines.append("")
    lines.append("Top genes by weight (rank, gene, weight, specificity):")
    for r, g in enumerate(summary.top, 1):
        lines.append(f"{r}\t{g.gene}\t{_fmt(g.weight)}\t{g.specificity:.2f}")
    if summary.specific:
        lines.append("")
        lines.append("Additional highly specific genes (gene, weight, specificity):")
        for g in summary.specific:
            lines.append(f"{g.gene}\t{_fmt(g.weight)}\t{g.specificity:.2f}")
    return "\n".join(lines)

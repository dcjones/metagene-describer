"""The lookup_genes tool: lets the model check genes beyond those in its prompt."""

from __future__ import annotations

import numpy as np

from .backends import Tool
from .summary import gene_specificity

MAX_GENES_PER_CALL = 30

PROMPT = """
You have a lookup_genes tool for checking genes that aren't listed above. Unless the category \
will be ambiguous, sample_specific or technical, you must call it before answering: look up the \
canonical markers of the cell type, subtype, state or program you intend to name, and those of \
the main alternatives you are ruling out (e.g. for an endothelial call, PECAM1, CDH5 and VWF; for \
naive T cells, CCR7, SELL, LEF1 and TCF7; for a program such as an interferon response or \
hypoxia, its core genes). Do this even when the call looks obvious. Read the results this way:
- A marker with substantial weight and specificity in this metagene supports the call.
- A measured marker with negligible weight here that peaks in another metagene argues against \
the call, unless this metagene looks like an add-on module (a state or program) of a lineage \
whose core markers sit in that other metagene. NMF often splits a cell type into a core factor \
and add-on factors; if that is what you see, name the lineage in the label with a qualifier and \
say so in the description.
- For a program (interferon, hypoxia, cell cycle, ...), if its core genes peak in another \
metagene, this one carries at most a weak share of it; say "weak" or don't name the program.
- "not measured" means the gene isn't in the data (e.g. not on a targeted panel), so it is not \
evidence either way.
Make a few well-chosen lookups, batching genes into one call, and don't look up genes already \
listed above.
"""

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "genes": {
            "type": "array",
            "items": {"type": "string"},
            "description": f"Gene symbols to look up, at most {MAX_GENES_PER_CALL}.",
        }
    },
    "required": ["genes"],
    "additionalProperties": False,
}

DESCRIPTION = (
    "Look up genes in this dataset. For each gene: its rank by weight, weight and specificity in "
    "the metagene being labeled, and the metagene where the gene is most specific, with that "
    "metagene's top genes. Reports genes that aren't measured."
)


class GeneLookup:
    """Gene statistics across all metagenes, for answering lookup_genes calls."""

    def __init__(self, genes: list[str], H: np.ndarray, usage: np.ndarray | None, n_peak: int = 4):
        row_sums = H.sum(axis=1, keepdims=True)
        self.genes = genes
        self.weights = np.divide(H, row_sums, out=np.zeros_like(H), where=row_sums > 0)
        self.specificity = gene_specificity(H, usage)
        self.index = {g: j for j, g in enumerate(genes)}
        lower: dict[str, list[int]] = {}
        for j, g in enumerate(genes):
            lower.setdefault(g.lower(), []).append(j)
        # Case-insensitive fallback (e.g. "Pecam1" for "PECAM1"), when it's unambiguous.
        self.lower = {g: js[0] for g, js in lower.items() if len(js) == 1}
        score = self.weights * self.specificity
        top = np.argsort(-score, axis=1)[:, :n_peak]
        self.peak_genes = [[genes[j] for j in row] for row in top]

    def _find(self, gene: str) -> int | None:
        j = self.index.get(gene)
        return j if j is not None else self.lower.get(gene.lower())

    def describe(self, metagene: int, query: list[str]) -> str:
        w, n = self.weights[metagene], len(self.genes)
        lines = []
        for gene in query[:MAX_GENES_PER_CALL]:
            j = self._find(str(gene))
            if j is None:
                lines.append(f"{gene}: not measured in this dataset")
                continue
            name = self.genes[j]
            rank = int((w > w[j]).sum()) + 1
            line = (
                f"{name}: rank {rank} of {n} here, weight {100 * w[j]:.3f}%, "
                f"specificity {self.specificity[metagene, j]:.2f}"
            )
            peak = int(np.argmax(self.specificity[:, j]))
            if self.specificity[peak, j] == 0:
                line += "; no weight in any metagene"
            elif peak == metagene:
                line += "; most specific to this metagene"
            else:
                line += (
                    f"; most specific to metagene {peak} (specificity "
                    f"{self.specificity[peak, j]:.2f}, weight {100 * self.weights[peak, j]:.3f}%; "
                    f"its top genes: {', '.join(self.peak_genes[peak])})"
                )
            lines.append(line)
        if len(query) > MAX_GENES_PER_CALL:
            lines.append(f"(only the first {MAX_GENES_PER_CALL} genes were looked up)")
        return "\n".join(lines)

    def tool(self, metagene: int, log: list[str]) -> Tool:
        """A lookup_genes tool for one metagene, recording queried genes in log."""

        def run(arguments: dict) -> str:
            query = [str(g) for g in arguments.get("genes") or []]
            log.extend(query[:MAX_GENES_PER_CALL])
            return self.describe(metagene, query)

        return Tool("lookup_genes", DESCRIPTION, INPUT_SCHEMA, run)

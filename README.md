# metagene-describer

Label and describe [metagene](https://github.com/dcjones/metagene) NMF factors with an LLM.

For each metagene, the model is shown its top genes, each with its weight and specificity. Before
naming a cell type or program, it checks that call's canonical markers with a gene lookup tool. It
returns a short label ("T cell markers"), a 1–3 sentence description, a category, a confidence,
candidate cell types, and the genes it based its answer on. Metagenes with no coherent theme (common in tumor
samples) are labeled by their dominant genes, e.g. `KRT17/S100A2/LGALS7-high`.

## Install

```sh
uv add "metagene-describer[anthropic]"   # Claude, through the Anthropic SDK
uv add "metagene-describer[openai]"      # any OpenAI-compatible endpoint
```

## Usage

```python
import metagene
from metagene_describer import describe_metagenes, to_dataframe

result = metagene.nmf(X, k=100)  # X: cells x genes

descriptions = describe_metagenes(
    gene_names,
    result.H,              # [k, n_genes]
    usage=result.W,        # optional, but recommended (see below)
    context="Human cervical cancer, Xenium 5k panel, tumor and adjacent normal tissue",
    verbose=True,
)
df = to_dataframe(descriptions)  # requires pandas
```

The default backend is Anthropic with `claude-haiku-5-5`. The API key and endpoint are read from
`ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL`, or you can pass them as arguments:

```python
describe_metagenes(genes, H, model="claude-sonnet-5-5", api_key="...")
```

For an OpenAI-compatible server (OpenAI, OpenRouter, vLLM, Ollama, an institutional gateway):

```python
describe_metagenes(genes, H, backend="openai", model="deepseek/deepseek-v4.1-flash",
                   base_url="https://openrouter.ai/api/v1", api_key="...")
```

The server must support tool calls for the default `lookup=True`. Prefer models that support
structured outputs. Accuracy varies a lot between models: on the example datasets, Haiku 5.5 was
the most accurate, DeepSeek v4.1 Flash and gpt-6-luna came close, and several others often gave
biological stories to technical factors. See `eval/README.md` for the comparison.

For more control, build the backend yourself:

```python
from metagene_describer import AnthropicBackend, OpenAIBackend

backend = AnthropicBackend("claude-haiku-5-5", effort="low")
backend = OpenAIBackend("qwen3-32b", base_url="http://localhost:8000/v1", json_mode="object", temperature=0)
describe_metagenes(genes, H, backend=backend)
```

`json_mode` sets how the OpenAI-compatible backend asks for JSON. `"schema"` (the default) uses a
strict JSON schema, `"object"` uses JSON mode, and `"prompt"` relies on the prompt alone. Use
`"object"` or `"prompt"` for servers that don't support schemas. Any other object with a
`complete_json(system, user, schema, tools=()) -> dict` method also works as a backend; it only
needs to handle `tools` if you use `lookup=True` (the default).

### Options

- `context`: free text about the dataset (tissue, disease, platform, panel). It is sent with every
  metagene.
- `notes`: per-metagene notes, as a list or a `{index: str}` dict. Use them for facts the gene list
  can't show, such as `"92% of usage comes from patient P3"`. This helps the model recognize
  sample-specific factors.
- `metagenes`: describe only a subset of metagenes, by index.
- `n_top`, `n_specific`: how many genes to show the model.
- `min_usage_share`: with `usage` given, metagenes below this share of total usage are labeled
  `Inactive` without calling the model.
- `lookup` (default `True`): let the model look up genes, and require it to check markers before
  naming a cell type, state or program (see below).
- `refine` (default `False`): reconcile labels across metagenes (see below).

### Gene lookup

The model sees only each metagene's top genes, so it can't tell on its own whether a call's other
canonical markers are present. With `lookup=True` it gets a `lookup_genes` tool. For each gene the
tool returns:
- its rank, weight and specificity in the metagene being labeled;
- the metagene where the gene is most specific, with that metagene's top genes;
- "not measured" if the gene isn't in the dataset, e.g. not on a targeted panel.

The model must use the tool before naming a cell type, state or program. It then sees, for
example, that PECAM1 and CDH5 peak in another metagene before calling a SOX17-high factor
endothelial. It also sees when a factor is an add-on module whose lineage markers sit in a sibling
metagene. No marker lists are built in, so this works for any species or panel; gene symbols are
matched case-insensitively when there's no exact match.

On three example datasets (300 metagenes, graded against reference annotations), required lookup
cut labels that claim more than the evidence supports from 19 to 4. The trade-off was a few more
vague labels: 12 instead of 6–9. It roughly doubles the cost, because about half the metagenes take
extra round trips. Each result's `looked_up` field lists the genes the model checked.

`build_prompts(genes, H, ...)` returns the exact prompts without calling the model, so you can
inspect them.

### Refining labels jointly

Each metagene is labeled on its own, so similar metagenes often get the same label, or labels like
"Immediate early stress genes" and "Immediate early response" that differ in wording only.
`refine=True` (or `refine_labels(descriptions, genes, H, usage=W, context=...)` on existing
results) adds a joint step:

1. One request sees all the confident labels and groups those that name the same concept.
2. For each group, a request sees each member's top genes, the genes that distinguish it from the
   other members, and how similar the members' weights are. The distinguishing genes are computed
   locally and weighted by specificity, so ubiquitous genes don't show up as differences. The
   model then either gives the group one shared wording, or labels that state a real difference,
   e.g. "IgA class-switched B cells" vs "IgG3 class-switched B cells". When the differences are
   weak, labels stay identical.

Changed labels keep the original in `initial_label`, with the model's reason in `refine_reason`.
Ambiguous and sample-specific metagenes keep their gene-based labels.

### Why pass `usage`

NMF doesn't fix the scale of a factor: scaling `W[:, k]` up and `H[k]` down leaves `W H` unchanged.
So raw `H` values can't be compared across metagenes. Each gene's *specificity* is the share of its
modeled expression that a metagene accounts for, computed as `usage[k] * H[k, g]` normalized over
`k`. Without usage, every metagene is treated as equally used. Specificity is what lets the model
discount genes such as ACTB or MALAT1, which carry heavy weight in many metagenes.

## Output

Each `MetageneDescription` has:

| field | |
|---|---|
| `label` | short label; falls back to `gene_label` if the request failed |
| `description` | 1–3 sentences, citing the supporting genes |
| `category` | `cell_type`, `cell_state`, `program`, `technical`, `sample_specific`, `ambiguous` (or `inactive`) |
| `confidence` | `high`, `medium`, `low` |
| `cell_types` | candidate cell types or states, most likely first |
| `key_genes` | the most informative genes (only genes the model was shown or looked up) |
| `gene_label` | top 3 genes by weight × specificity, computed locally |
| `top_genes` | the top genes by weight that the model was shown |
| `error` | set if the request failed |
| `initial_label`, `refine_reason` | set when refinement changed the label |
| `looked_up` | genes the model checked with the lookup tool |

## Labeling clusters

Once the metagenes are described, `describe_clusters` labels cell clusters (e.g. Leiden clusters)
from their metagene composition:

```python
from metagene_describer import describe_clusters

clusters = describe_clusters(
    adata.obs["leiden"],   # one cluster label per cell
    descriptions,          # from describe_metagenes
    gene_names,
    result.H,
    result.W,
    context="Human cervical cancer, Xenium 5k panel, tumor and adjacent normal tissue",
)
df = to_dataframe(clusters)
```

`W[i, k] * H[k].sum()` is the modeled count that metagene k contributes to cell i, which puts
metagenes on a common scale. For each cluster, the model sees:
- its largest metagenes, plus the most enriched of the rest. Each is shown with its share of the
  cluster's modeled counts, its fold over all other cells, the fraction of the cluster's cells in
  which it makes up at least 10% of the cell, and the cluster where its share is largest;
- each of those metagenes' label, category, key genes and description;
- the cluster's top modeled genes, as a check on the metagene labels.

The prompt treats `cell_type` metagenes as identity, `cell_state` and `program` metagenes as
qualifiers, and `technical` ones as noise. It also asks the model to judge foreign-lineage signal:
ignore small amounts, note moderate amounts in `contaminants`, and put large amounts in the label,
e.g. "Fibroblasts (+ macrophage signal)". Clusters where no lineage dominates are categorized as
`mixed`. Many clusters are expected to share a label. With `refine=True` (the default), one final
request sees every cluster's label and composition, makes the wording consistent, and adds
qualifiers where clusters with the same identity clearly differ.

The labels are fine-grained, so one cell type split across several clusters often has several
labels, e.g. "Plasma cells (interferon-high)" and "IgG plasma cells". With `broad=True` (the
default), one more request sees every distinct label and groups them under broad cell type labels.
These name the major cell type alone, dropping subsets, states and contamination notes, and are
stored in `broad_label`. Mixtures of unrelated lineages become "Mixed", and low-quality or
unassigned clusters become "Unassigned".

`notes` takes a `{cluster: str}` dict, e.g. `{"7": "85% of cells from patient P3"}`.
`build_cluster_prompts` returns the prompts without calling the model.

Each `ClusterDescription` has `cluster`, `n_cells`, `label`, `description`, `category`
(`cell_type`, `cell_state`, `mixed`, `technical`, `ambiguous`), `confidence`, `cell_types`,
`key_metagenes`, `contaminants`, `metagene_label` (the fallback label, built from the labels of
the largest informative metagenes), `top_metagenes`, `error`, `initial_label` /
`refine_reason`, and `broad_label`.

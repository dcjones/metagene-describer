# metagene-describer

Label and describe [metagene](https://github.com/dcjones/metagene) NMF factors with an LLM.

For each metagene, the model is shown its top genes, each with its weight and specificity. It returns
a short label ("T cell markers"), a 1–3 sentence description, a category, a confidence, candidate
cell types, and the genes it based its answer on. Metagenes with no coherent theme (common in tumor
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
describe_metagenes(genes, H, backend="openai", model="gpt-...", base_url="https://.../v1", api_key="...")
```

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
`complete_json(system, user, schema) -> dict` method also works as a backend.

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
| `key_genes` | the most informative genes (only genes the model was shown) |
| `gene_label` | top 3 genes by weight × specificity, computed locally |
| `top_genes` | the top genes by weight that the model was shown |
| `error` | set if the request failed |
| `initial_label`, `refine_reason` | set when refinement changed the label |

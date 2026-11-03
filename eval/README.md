# Evaluation

Reference annotations and graded results for three example metagene runs (k = 100), used to compare
configurations of `describe_metagenes`.

| dataset | context |
|---|---|
| `scrnaseq-aging-pbmc-250k` | Human PBMCs from donors of varying ages, 10x scRNA-seq (250k cells) |
| `atera-cervical-cancer` | Human cervical cancer, Atera spatial transcriptomics, whole transcriptome |
| `xenium-5k-ovarian-cancer` | Human high-grade serous ovarian cancer, 10x Xenium Prime 5k panel, FFPE |

The datasets themselves are not in the repository. `datasets.py` reads them from
`$METAGENE_EXAMPLES` (default `/mnt/extra2/metagene-examples/results`). The spatial runs are the
ones regenerated after a proseg bug that zeroed counts for hundreds of genes was fixed. Results
from before that fix are not included.

## Contents

- `references/DATASET.jsonl`: one reference annotation per metagene, written blind by Claude Opus
  5.5 subagents who could look up any gene's weight and specificity (`prompts/annotate.md`). Each
  has a label, category, confidence, a list of acceptable alternative labels, a list of labels that
  would overreach, and the evidence.
- `results/DATASET.VARIANT.tsv`: `describe_metagenes` output (`to_dataframe`), all with
  `claude-haiku-5-5`:
  - `nolookup-a`, `nolookup-b`: two runs without gene lookup, to measure run-to-run variation
  - `lookup-optional`: lookup tool available but not required (an earlier prompt)
  - `lookup-required`: the current default
  - `refine-haiku`, `refine-sonnet`: `nolookup-a` refined with Haiku 5.5 or Sonnet 5.5
  - `or-MODEL`: other models through OpenRouter (the OpenAI-compatible backend, with the
    defaults at the time)
  - `lookup-required-b`: a second Haiku run of `lookup-required`
  - `*-sg`, `*-sg-b`: the same configurations with a first version of the rule for metagenes
    dominated by one gene. Its examples (PKM, EEF1G, ATF4) were taken from Xenium metagenes, which
    leaked answers into the eval, so don't rely on these.
  - `*-sg2`, `*-sg2-b`: the single-gene rule with clean examples
  - `*-am2`, `*-am2-b`: the single-gene rule plus the rule for add-on factors of a lineage, with
    clean examples (the current default prompt)
- `grades/BATCH/`: blind grades (`prompts/grade.md`). Each batch has the grading inputs
  (`DATASET.jsonl`), the judge's grades (`DATASET.grades.jsonl`) and `key.json`, which maps
  candidates back to variants. Batches are `lookup` (the four first-pass variants), `refine`
  (`nolookup-a` and the two refinements) `models` (`lookup-required` and the OpenRouter
  models), `single-gene` and `single-gene-haiku` (the leaked first version of the single-gene rule;
  superseded) and `rules-clean` (no rule, single-gene rule, and both rules, with clean examples).

## Results

```
$ python eval/score.py eval/grades/lookup
all
  lookup-optional correct 277  vague   8  overreach  15  wrong   0
  lookup-required correct 283  vague  12  overreach   4  wrong   1
  nolookup-a      correct 273  vague   6  overreach  19  wrong   2
  nolookup-b      correct 271  vague   9  overreach  19  wrong   1

$ python eval/score.py eval/grades/refine
all
  nolookup-a      correct 273  vague   8  overreach  17  wrong   2
  refine-haiku    correct 272  vague   7  overreach  19  wrong   2
  refine-sonnet   correct 270  vague   8  overreach  20  wrong   2
```

```
$ python eval/score.py eval/grades/models
all
  lookup-required                   correct 285  vague  11  overreach   4  wrong   0
  or-deepseek-v4.1-flash            correct 269  vague  19  overreach  12  wrong   0
  or-gemini-3.1-flash-lite          correct 247  vague  15  overreach  38  wrong   0
  or-gpt-6-luna                     correct 264  vague  24  overreach  12  wrong   0
  or-mistral-small-3.2-24b-instruct correct 177  vague  21  overreach  95  wrong   7
  or-qwen3.6-35b-a3b                correct 215  vague  17  overreach  62  wrong   6
```

Approximate cost per 300 metagenes, from token counts and list prices as of October 2026:
DeepSeek v4.1 Flash $0.13, gpt-6-luna $0.23, Haiku 5.5 $0.27, Mistral Small 3.2 $0.40, Gemini
3.1 Flash Lite $0.55, Qwen 3.6 35B $1.55.

```
$ python eval/score.py eval/grades/rules-clean
all
  lookup-required                   correct 281  vague  12  overreach   7  wrong   0
  lookup-required-b                 correct 281  vague  11  overreach   8  wrong   0
  lookup-required-sg2               correct 274  vague  18  overreach   8  wrong   0
  lookup-required-sg2-b             correct 277  vague  15  overreach   8  wrong   0
  lookup-required-am2               correct 276  vague  13  overreach  11  wrong   0
  lookup-required-am2-b             correct 285  vague   8  overreach   7  wrong   0
  or-deepseek-v4.1-flash            correct 263  vague  24  overreach  13  wrong   0
  or-deepseek-v4.1-flash-sg2        correct 266  vague  26  overreach   8  wrong   0
  or-deepseek-v4.1-flash-am2        correct 270  vague  23  overreach   7  wrong   0
  or-gpt-6-luna                     correct 259  vague  29  overreach  12  wrong   0
  or-gpt-6-luna-sg2                 correct 269  vague  27  overreach   4  wrong   0
  or-gpt-6-luna-am2                 correct 275  vague  15  overreach  10  wrong   0
```

With both rules (`am2`), DeepSeek gains 7 correct labels and gpt-6-luna 16, and Haiku is
unchanged within its run-to-run variation: two runs of the same Haiku prompt differ by up to 9
labels. Totals for the same variant also shift by a few labels between batches, because a
different judge grades each batch, so compare variants only within a batch.

### Keeping the eval honest

Examples in the prompts must not come from the eval datasets. A prompt example like 'PKM alone is
not "Glycolysis"' hands the model the answer for any eval metagene dominated by PKM. Before
evaluating a prompt change, run

```sh
uv run --group eval python eval/check_prompt_leakage.py
```

It lists prompt genes that rank in the top 3 of any eval metagene. The hits that remain now are
canonical markers named in instructions (CCR7, SELL, TCF7, VWF) and examples of ubiquitous genes
(ACTB, TMSB4X). These don't say what any particular metagene is.

The grades only score each label against its own reference, so they don't measure what refinement
is for (distinguishing similar metagenes). They show only that refinement doesn't hurt accuracy.

## Evaluating a new configuration

```sh
# 1. Label each dataset (any backend; see run.py --help)
uv run --group eval python eval/run.py xenium-5k-ovarian-cancer eval/results/xenium-5k-ovarian-cancer.NEW.tsv

# 2. Build a blind batch with the new variant and at least one existing one for comparison
uv run --group eval python eval/build_grading.py eval/grades/NEW-batch lookup-required NEW

# 3. Have a judge grade each DATASET.jsonl per prompts/grade.md into DATASET.grades.jsonl

# 4. Count
python eval/score.py eval/grades/NEW-batch
```

To build a reference for a new dataset, add it to `datasets.py`, write annotator inputs with
`reference_inputs.py DATASET DIR`, and annotate per `prompts/annotate.md`.

## Cluster labels

`describe_clusters` is scored on two datasets with reference cell types: the PBMC dataset
(published annotation) and `atera-cervical-cancer-50k`, a 50k-cell region of the Atera cervical
cancer demo segmented with proseg, with the vendor's cell groups matched by `original_cell_id`.
These tables come from rules in `$METAGENE_EXAMPLES/../Snakefile`. Their rows are in the same cell
order as the metagene run:
- `celltypes/scrnaseq-aging-pbmc-250k-celltypes.parquet` (`pbmc_250k_celltypes`): reference cell
  types for the 250k downsampled cells, matched by barcode from the full table.
- `results/DATASET/leiden-expr-r2.0.parquet` (`leiden_expression`): Leiden clusters (resolution 2,
  seed 0) from the standard scanpy pipeline on log-normalized expression. 29 clusters.
- `results/DATASET/metagene-k100-leiden-r2.0.parquet` (`leiden_metagene`): Leiden clusters on
  each cell's metagene composition. 24 clusters.
- `data/atera-cervical-cancer-50k.zarr` (`atera_cervical_50k_data`), its metagene run, and
  `celltypes/atera-cervical-cancer-50k-celltypes.parquet` (`atera_cervical_50k_celltypes`; 8 cells
  have no vendor group). Its clusters are `leiden-expr-r1.0` (52 clusters).

`cluster_reference.py` places each dataset's reference cell types in a hierarchy. For PBMC, the
levels are compartment, lineage (T, NK, B, monocyte, DC…), subset (CD4, CD8, MAIT/γδ,
CD56dim/bright…), state (naive, memory, Treg) and fine state (central vs effector memory). For
cervix, they are compartment (immune, stromal, epithelial), lineage (T, B/plasma, myeloid,
fibroblast, smooth muscle, endothelial, squamous/tumor, glandular) and subset (the vendor's T cell
groups, B vs plasma, macrophage/DC/neutrophil/mast, the fibroblast types, and the vendor's seven
tumor states). The vendor groups "OR4F17+ Cells" and "Detachment" aren't cell types and are left
unscored. `score_clusters.py` maps
each distinct label onto that hierarchy with one blind request that sees only label text
(`claude-opus-5-5`, cached per hierarchy in `clusters/label-nodes.json`). A label naming alternatives (e.g. "CD8
T/NK cells") maps to several nodes, and each gets an equal share of the credit. At each level,
each cell is then graded correct, wrong, or unresolved (the label stops above that level), from its
cluster's label and its reference type. The oracle row gives each cluster its majority reference
type: the best one label per cluster can do, given how pure the clusters are.

```sh
uv run --group eval python eval/run_clusters.py scrnaseq-aging-pbmc-250k \
    eval/results/scrnaseq-aging-pbmc-250k.lookup-required-am2.tsv \
    eval/clusters/results/scrnaseq-aging-pbmc-250k.expr.NEW.tsv
uv run --group eval python eval/score_clusters.py scrnaseq-aging-pbmc-250k \
    eval/clusters/results/scrnaseq-aging-pbmc-250k.expr.*.tsv [--detail]
```

Results (`expr` clusters, Haiku 5.5, metagene labels from `lookup-required-am2`). Each level shows
% of cells correct / wrong / unresolved:

| | lineage | subset | state | fine state |
|---|---|---|---|---|
| oracle | 99.1 / 0.9 / 0 | 96.7 / 3.3 / 0 | 89.0 / 11.0 / 0 | 79.4 / 20.6 / 0 |
| v1 | 94.9 / 5.0 / 0.1 | 68.9 / 8.1 / 23.1 | 45.4 / 24.0 / 30.6 | 23.7 / 48.1 / 28.2 |
| v1-b | 98.3 / 1.6 / 0.1 | 67.8 / 4.2 / 28.1 | 34.1 / 15.7 / 50.2 | 10.2 / 21.8 / 68.0 |
| v2 | 98.3 / 1.6 / 0.1 | 88.4 / 4.9 / 6.7 | 61.8 / 21.6 / 16.5 | 23.7 / 43.1 / 33.1 |
| v2-b | 98.3 / 1.6 / 0.1 | 84.5 / 4.7 / 10.9 | 56.6 / 21.3 / 22.2 | 23.7 / 42.6 / 33.7 |

v1 is the first prompt. v2 adds three rules:
- the label should be as specific as the description, since it is a call about most of the cells;
- name the favored identity rather than "A/B" unless both are about equally supported;
- decide between two states of one lineage by their relative strength.

It also tells the refinement step never to broaden or hedge a label to match another cluster. In
v1, refinement had turned a correct "CD8 cytotoxic effector T cells" into "CD8 T/NK cells".

One cluster changes the score by up to 12% of cells, so compare runs against the run-to-run
spread (v1 vs v1-b). The largest remaining error is one 27k-cell cluster of effector/central
memory CD4 T cells, labeled "naive/central memory-like" in every run. Its naive CD4 metagenes
together outweigh its one CD4 memory metagene, and only its top genes point to memory.

v3 and later were tuned on Atera as well (`--res 1.0`, metagene labels from
`atera-cervical-cancer-50k.lookup-required-am2`), with PBMC checked for regressions:
- v3: a cluster dominated by a program still has a lineage, so find it ("Cycling epithelial
  cells"); and contamination must rest on a listed foreign-lineage metagene, not on single genes.
  v2 had called S100A8/A9 in squamous epithelium "myeloid signal".
- v4: the listed metagenes are grouped as lineage (cell_type, cell_state), program, and other, so
  identity evidence comes first.
- v5: a metagene labeled "A/B" leaves the choice open; decide it from the other metagenes and the
  top genes.

| Atera | compartment | lineage | subset |
|---|---|---|---|
| oracle | 86.6 / 13.4 / 0 | 76.6 / 23.4 / 0 | 49.0 / 51.0 / 0 |
| v2, v2-b | 74.3–74.8 / 15–16 / 9–11 | 64.8 / 23–24 / 11–12 | 16–26 / 44–47 / 28–41 |
| v3, v3-b | 75.1–82.3 / 17–18 / 0–8 | 65.7–73.6 / 25 / 1–10 | 20–26 / 42–47 / 27–38 |
| v4, v4-b | 82.3 / 17.7 / 0 | 73.5–73.6 / 25.5 / 1 | 21–22 / 52 / 26 |
| v5, v5-b | 81.8–82.3 / 17–18 / 0–1 | 73.0–73.6 / 25 / 1–2 | 25.6 / 48 / 26–27 |

| PBMC | lineage | subset | state |
|---|---|---|---|
| v4, v4-b | 96.2–97.0 / 2.9–3.7 / 0.1 | 80.0–89.9 / 7.0–7.1 / 3–13 | 61.6–66.7 / 25–26 / 8–13 |
| v5, v5-b | 97.0–98.6 / 1.3–2.9 / 0.1 | 84.8–94.1 / 5.4–8.6 / 0.5–7 | 62.9–77.6 / 22–27 / 0–10 |

The Atera oracle is low: even with each cluster given its majority vendor group, only 77% of cells
are right at lineage level. The vendor annotated the full dataset with its own segmentation, and
about 30 of the 52 clusters have under 150 cells and a near-uniform mix of vendor groups. Read the
subset level with care. Most of it is the vendor's tumor-state vocabulary (dyskeratotic,
differentiating, metabolic-invasive basal…), which the labels aren't expected to reproduce. Known
gaps:
- No metagene was annotated as neutrophils or mast cells, so those cells end up in low-quality or
  mixed clusters.
- The small, mixed clusters are mostly labeled "Fibroblasts".
- The 7.7k-cell T cell cluster that is 58% cytotoxic T is labeled as Tregs.

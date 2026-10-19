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
  - `*-sg`, `*-sg-b`: the same configurations with the rule for metagenes dominated by one gene
    (now the default prompt)
- `grades/BATCH/`: blind grades (`prompts/grade.md`). Each batch has the grading inputs
  (`DATASET.jsonl`), the judge's grades (`DATASET.grades.jsonl`) and `key.json`, which maps
  candidates back to variants. Batches are `lookup` (the four first-pass variants), `refine`
  (`nolookup-a` and the two refinements) `models` (`lookup-required` and the OpenRouter
  models), `single-gene` (Haiku, DeepSeek and gpt-6-luna with and without the single-gene
  rule) and `single-gene-haiku` (two Haiku runs with and two without it).

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
$ python eval/score.py eval/grades/single-gene
all
  lookup-required                   correct 278  vague  14  overreach   8  wrong   0
  lookup-required-sg                correct 272  vague  18  overreach  10  wrong   0
  or-deepseek-v4.1-flash            correct 263  vague  24  overreach  13  wrong   0
  or-deepseek-v4.1-flash-sg         correct 268  vague  25  overreach   7  wrong   0
  or-gpt-6-luna                     correct 256  vague  33  overreach  10  wrong   1
  or-gpt-6-luna-sg                  correct 263  vague  32  overreach   5  wrong   0

$ python eval/score.py eval/grades/single-gene-haiku
all
  lookup-required                   correct 281  vague  14  overreach   4  wrong   1
  lookup-required-b                 correct 279  vague  15  overreach   6  wrong   0
  lookup-required-sg                correct 273  vague  18  overreach   9  wrong   0
  lookup-required-sg-b              correct 281  vague  16  overreach   3  wrong   0
```

The single-gene rule helps DeepSeek and gpt-6-luna, mostly on Xenium. For Haiku its effect is
within run-to-run variation: two runs of the same prompt differ by up to 8 labels. Totals for the
same variant also shift by a few labels between batches, because a different judge grades each
batch, so compare variants only within a batch.

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

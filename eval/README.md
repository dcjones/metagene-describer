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

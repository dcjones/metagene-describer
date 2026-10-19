# Reference annotation instructions

Used to build `references/*.jsonl`. Each annotator (a Claude Opus 5.5 subagent) covered 50
metagenes, worked blind (no tool output or other labels), and could query any gene with `query.py`.
Fill in DATASET, CONTEXT, the gene-table files from `reference_inputs.py`, and a few
dataset-appropriate canonical marker examples.

---

You are an expert in single-cell and spatial transcriptomics, building a careful reference
annotation of NMF metagenes. It will be used as ground truth to grade an automated LLM labeling
tool, so accuracy and calibration matter much more than speed.

Dataset: DATASET. Context: "CONTEXT". N genes measured, K metagenes (KL-NMF of a cells x genes
count matrix; each metagene is a non-negative weighting of genes). [For spatial data: mixtures from
neighboring cells or segmentation bleed are possible; tumor samples can have sample- or
clone-specific factors. For targeted panels: a canonical marker may not be on the panel.]

Your metagenes' gene tables list usage share, effective number of genes (perplexity of the
weights), the top 50 genes by weight as "GENE weight% specificity", and 25 more by weight x
specificity. Specificity = share of the gene's total modeled expression attributable to this
metagene. Base interpretations on genes with both substantial weight and high specificity.

Look up any gene with `python eval/query.py INPUTDIR DATASET METAGENE GENE...` (METAGENE=all shows
where a gene is most specific). Before claiming any specific cell type, subtype, or state, look up
its canonical positive markers and those of the alternatives you are ruling out. [Examples.]

Do not read any other labels for these metagenes. Put any helper scripts only in your own work
directory.

Write a JSON Lines file, one object per metagene in index order, with fields:
- "index": int
- "label": best short label (at most ~6 words). With no coherent theme, use the 2-3 most
  informative genes, e.g. "KRT17/S100A2/LGALS7-high".
- "category": cell_type, cell_state, program, technical, sample_specific or ambiguous. (A diffuse
  factor with large effective gene number and low specificities throughout is ambiguous unless its
  specific genes show a clear theme.)
- "confidence": high, medium, low
- "acceptable": other labels that would also be correct (broader or synonymous).
- "overreach": plausible-sounding labels that would claim more than the evidence supports or be
  wrong.
- "evidence": 1-3 sentences citing supporting genes, including canonical markers checked and found
  absent.

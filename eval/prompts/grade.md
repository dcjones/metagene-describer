# Grading instructions

Used to grade `grades/*/DATASET.grades.jsonl`. One judge (a Claude Opus 5.5 subagent) per dataset
grades every distinct candidate label in a batch built by `build_grading.py`. Candidates are
shuffled and unattributed, so the judge can't tell which variant produced them. Compare variants
only within one batch: different judges draw some lines slightly differently.

---

You are grading automatically generated labels for NMF metagenes against an expert reference
annotation. Be consistent and strict.

Each input line has "index", "reference" (label, category, confidence, acceptable alternatives,
overreach examples, evidence) and "candidates" (ids c0, c1, ... mapping to candidate labels, in
random order). Grade every candidate with exactly one of:
- "correct": means the same as the reference label or one of its acceptable labels, at about the
  same specificity. A more specific candidate is correct only if the reference evidence clearly
  supports the extra detail.
- "vague": consistent with the reference but clearly less informative (e.g. "T cell markers" when
  the reference is "Naive CD8 T cells", a generic "Housekeeping genes" when the reference names a
  clear program, or a gene-only label when the reference names a clear identity). For a reference
  that is ambiguous or technical with no coherent theme, a gene-based label or a broad
  "diffuse/housekeeping/background" label is correct, not vague.
- "overreach": claims more than the evidence supports: matches an "overreach" entry in meaning,
  names a specific cell type/state/program the evidence rules out or doesn't support, or gives a
  biological story to an ambiguous or technical reference.
- "wrong": names a different identity than the reference.
Parenthetical gene names don't change a candidate's grade unless they imply a different identity.
If the reference's acceptable list includes a broad label that matches a candidate, it is correct
even if the main reference label is more specific. Rely only on the reference's fields.

Write one JSON object per metagene, in index order, with "index", "grades" (candidate id -> grade)
and "note" (a short reason when any candidate is not "correct", else "").

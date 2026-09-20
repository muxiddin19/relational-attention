# Evaluation results

This file replaces an earlier version of this README that reported ICDE-2027-era
numbers, several of which have since been withdrawn or corrected -- see the
paper's Conclusion for the full list of corrections. The historical version is
preserved in git history rather than deleted.

## A note on how one of the bugs in this project was first suspected

The previous version of this file reported, side by side:

| File | Variant | Seed | answer_accuracy |
|---|---|---|---|
| eval_no_gating_s42.json | RelTransformer -Gating | 42 | 0.031842 |
| eval_no_mixing_s42.json | RelTransformer -Mixing | 42 | 0.031842 |

Two supposedly different ablations (removing gating vs. removing mixing) producing
a bit-for-bit identical accuracy at the same seed is exactly the kind of thing that
should not happen if the two ablations are actually doing different things to the
model, and was one of the observations that prompted the audit which found the
config-forwarding bug described below. We kept this detail here as a concrete
example of what to look for in your own ablation results.

## Where the current, corrected numbers live

- results/eval_k4_s{42,43,44}.json, results/eval_k16_s{42,43,44}.json -- slot-count
  sweep (Table 4). These configs use the default use_gating=true, use_mixing=true,
  so they are unaffected by the bug below and match the paper's reported values
  exactly (k=4: 2.6%+/-0.5; k=16: 2.3%+/-0.5).
- results/verified_2026_revision/ -- everything else: the GSM8K component ablation
  (post-bugfix), the pairing-strategy ablation, COGS, Spider-with-copy-mechanism,
  and the Composed Join Attention feasibility test. See
  results/verified_2026_revision/README.md for the exact file-to-table mapping
  and which comparisons we have individually cross-checked vs. verified only in
  aggregate.
- results/retracted_pre_bugfix/ -- the no_gating/no_mixing files quoted above and
  their siblings, kept for transparency, not to be used.
- results/unverified_needs_audit/ -- results found in this repository that have
  not been through the verification protocol described in the paper's
  Mechanistic Analysis section, most notably an FFN-width-matched Standard
  Transformer control (eval_ffn2900_*.json) whose provenance we have not
  confirmed. **This is a live discrepancy, not a resolved one**: the paper's
  Limitations section currently states this control "was not, however, actually
  run... it remains an unexecuted control," but a result file matching that
  description exists in this repository's history. We have not verified whether
  it reflects a complete, correctly-configured run, so we do not cite it, but the
  paper's wording needs to be reconciled with this file's existence before
  submission -- either by verifying and reporting the run, or by describing
  honestly why it is not trusted, rather than stating flatly that it was never run.

## The config-forwarding bug, precisely

`use_gating` and `use_mixing` were defined in the training config (see any
`configs/*_ablation_no_gating*.yaml` / `*_no_mixing*.yaml` file) and were read
correctly by `scripts/train.py`'s encoder-block construction, but the
decoder-block class used no such parameters at all, so any run intending to
ablate gating or mixing away trained a decoder with **both fully active**
regardless of the flag. The identical bug pattern -- the same two flags, plus
`pairing_strategy` -- existed independently in `scripts/evaluate.py`'s
model-reconstruction-from-checkpoint function. Both are now fixed; the fix
threads `use_gating`, `use_mixing`, and `pairing_strategy` through every model
construction and reconstruction call site (`relational_attention/layers.py`,
`relational_attention/model.py`, `scripts/train.py`, `scripts/evaluate.py`,
`analysis/probing.py`).

## Evaluation command

```
python scripts/evaluate.py \
  --checkpoint <path>/best_model \
  --dataset gsm8k \
  --nas-dir /nas/Dataset/nlp \
  --split dev \
  --output-file eval_result.json
```

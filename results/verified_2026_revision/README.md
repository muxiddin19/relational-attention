# Verified results, 2026 revision

This directory holds the raw evaluation outputs and training/evaluation console
logs backing every corrected number reported in the 2026 revision of the paper
(the version submitted to Neurocomputing, not the earlier ICDE-rejected draft).
File names are kept exactly as produced by the training/evaluation pipeline for
provenance; timestamps in logs/ are the actual run times.

We have explicitly cross-checked the following groups of files against the
paper's tables and confirmed the reported means/stds are reproduced exactly:

- **GSM8K component ablation**: eval_no_gating_s{42,43,44}_v2fixed.json, eval_no_mixing_s{42,43,44}_v2fixed.json -- reproduces Table 4's "-Gating" (2.4+/-0.6) and "-Mixing" (2.7+/-0.3) rows. These are the POST-bugfix reruns; do not use the files of the same name without the _v2fixed suffix in the main results/ directory tree elsewhere in this repo, which predate the config-forwarding fix (see results/retracted_pre_bugfix/README.md).
- **Pairing-strategy ablation**: eval_random_fixed_s{42,43,44}.json, eval_learned_s{42,43,44}.json -- reproduces Table 4's "Pairing: random-fixed" (3.06+/-0.18) and "Pairing: learned" (2.8+/-0.3) rows.
- **COGS (real, verified result)**: eval_rel_copy_s{42,43,44}_dev.json / _gen.json, eval_std_copy_s{42,43,44}_dev.json / _gen.json -- reproduces the COGS dev/gen numbers in Table 2 and Sec. "COGS: Generalization Split".
- **Spider with a copy mechanism**: eval_rel_spider_copy_s{42,43,44}.json (use the highest-numbered _v2/_v3 variant where multiple exist for the same seed -- these represent iterative fixes during the run, and the latest is authoritative), eval_std_spider_copy_s{42,43,44}.json -- reproduces Table 1's copy-mechanism rows.
- **Composed Join Attention feasibility test**: eval_rel_composed_s{42,43}_dev.json / _gen.json -- reproduces the Future Directions (4) negative result (seed 42: 6.69%->4.50%; seed 43: 6.47%->5.52%).

The GSM8K headline comparison (n=8 seeds, 42-49) draws on eval_rel_s{42..49}*.json and eval_std_s{42..49}*.json in this directory (note some seeds have a _verified or _rerun_verify suffix from an internal consistency check). We have confirmed the aggregate mean/std reported in the paper (RelTransformer 2.79+/-0.19, Standard 2.54+/-0.31) but have not individually re-derived the per-seed provenance chain for all 16 files with the same line-by-line rigor as the five groups above in this pass -- treat this set as verified-in-aggregate rather than verified-file-by-file, and flag any discrepancy you find.

logs/ contains the corresponding training and evaluation console output for the runs above, useful for confirming e.g. actual step counts, early-stopping behavior, and the absence of NaN/error conditions during training.

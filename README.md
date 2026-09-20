# Attribute-Decomposed Attention (RelAttn)

[![Status: submitted to Neurocomputing](https://img.shields.io/badge/status-submitted%20to%20Neurocomputing-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4-ee4c2c.svg)](https://pytorch.org/)

**Official implementation of "Attribute-Decomposed Attention: A Relational Inductive Bias for Structured Reasoning."**

This paper was rejected from ICDE 2026. This repository, and the paper in `paper/`, reflect a substantially corrected revision now prepared for submission to **Neurocomputing** (Elsevier) -- see [**Note on this revision**](#note-on-this-revision) below before trusting anything you read elsewhere about this project, including an earlier version of this README.

> Attribute-Decomposed Attention decomposes each token into *k* typed attribute slots and computes attention via slot-to-slot join pairs, a neural analogue of a relational foreign-key join. Balance, path-completeness, and minimum-edge constraints characterize valid head-assignment strategies as the family of directed Hamiltonian cycles on the *k* slots; the cyclic pairing used throughout is the canonical instance, not a uniquely optimal one. The architecture gives a large, statistically robust gain on COGS compositional generalization (5.81%+/-1.34 vs. 2.27%+/-1.94 generalization-split EM, Cohen's d=2.12, 3 seeds) and a modest, real gain on GSM8K (2.79%+/-0.19 vs. 2.54%+/-0.31, d=0.96, 8 seeds, caveated by answer-frequency mode collapse in both architectures), with no measurable advantage on Spider once both architectures have a copy mechanism. Rigorous probing for emergent, role-specific attribute-slot specialization -- the mechanistic story an earlier version of this paper claimed to have found -- turns up **no discriminability signal above an untrained, randomly initialized control**. We report this as an honest negative result, not a mechanism.

---

## Note on this revision

An earlier version of this paper (submitted to ICDE 2026 and rejected) made several claims that a subsequent audit found did not trace to complete, correctly configured, or correctly computed experiments:

- A central mechanistic-interpretability claim (Fisher discriminability F=18-22 for a trained model vs. F=1 for random init) came from a probing script with a flag that echoed hard-coded numbers without computing anything, and whose one real computation path probed the wrong dataset with an ungrounded labeling heuristic. Correctly recomputed (real data, content-based labels, random-init and permutation-null controls), the result is the opposite: F is approximately 0.0001-0.007 across all conditions, with the trained model sometimes *less* discriminative than random initialization. See `analysis/probing.py` (kept, with a retraction notice at the top, for transparency) vs. `analysis/probing_gsm8k_fixed.py` (the corrected implementation actually used for the paper).
- A claimed non-zero SCAN training trajectory had no surviving log evidence, old or new.
- A Spider foreign-key-depth statistic ("138 databases, median 3, p90 7, max 9") was contradicted by direct recomputation from the benchmark's own schema files (166 databases, median 1, p90 3, max 7).
- Two independent instances of a config-forwarding bug (`use_gating` / `use_mixing` defined in the training config but never threaded into the decoder block's constructor, in both `scripts/train.py` and `scripts/evaluate.py`) silently nullified an entire prior ablation study -- see `results/retracted_pre_bugfix/README.md` for a striking, concrete symptom of this bug that we kept in the record.
- The stated hardware (a single A100 80GB GPU) was wrong; the real hardware was 2x NVIDIA RTX 4090 (24GB each).

All of these are corrected in the current paper (`paper/relattn_neurocomputing_submission.pdf`), which reports the real numbers -- including honest nulls -- with full seed-level data. We are not silently fixing and re-releasing; the retracted claims, why they were wrong, and the corrected results are documented explicitly in the paper's Conclusion and Mechanistic Analysis sections, and the pre-correction artifacts are kept in this repository (clearly marked) rather than deleted. One open item from this same audit process has **not** yet been resolved: `results/unverified_needs_audit/` contains a result file for an FFN-width-matched Standard Transformer control that the paper's Limitations section currently describes as "not run in this work" -- we found the file after that text was written and have not yet verified its provenance, so treat that specific claim as pending reconciliation, not settled.

---

## Key Idea

Standard multi-head attention conflates all semantic aspects of a token into one similarity score. RelAttn decomposes every token representation into **k typed attribute slots** and assigns each attention head to a specific *pair* of slots:

```
Standard Attention (1 head shown)               RelAttn (k=8 slots, 1 head shown)
-------------------------------------           --------------------------------------------
                                                 Token x_i decomposed into k slots:
  x_i --[W_Q]--> q_i --.                        x_i --[W_0^e]--> a_i^(0)  [slot 0]
                        |-- q_i . k_j / root(d)  x_i --[W_1^e]--> a_i^(1)  [slot 1]
  x_j --[W_K]--> k_j --'                        x_i --[W_2^e]--> a_i^(2)  [slot 2]
                                                  ...
                                                  x_i --[W_7^e]--> a_i^(7)  [slot 7]

                                                 Head r uses pair (r mod k, (r+1) mod k):
                                                  head 0: a^(0)_i . a^(1)_j / root(d/k)
                                                  head 1: a^(1)_i . a^(2)_j / root(d/k)
                                                  ...  (cyclic, wraps at k)
                                                  head 7: a^(7)_i . a^(0)_j / root(d/k)
```

Balance, path-completeness, and minimum-edge constraints characterize valid head-assignment strategies as the family of directed Hamiltonian cycles on the *k* slots (see the paper's Theorem on head-assignment characterization) -- the cyclic pairing `(j, j+1 mod k)` used throughout is the canonical instance of this family, not a uniquely optimal one; a reviewer-requested ablation (`configs/*_ablation_random_fixed*.yaml`) confirms an arbitrary member of the same family performs statistically indistinguishably. Slot semantics (entity identity / functional-dependency operator / schema structure) were the design *intent* -- whether they actually emerge is an empirical question the paper tests directly and reports honestly did not happen, at least not detectably (see Note on this revision above).

---

## Results (real, verified numbers)

| Benchmark | RelTransformer | Standard Transformer | Effect | Seeds |
|---|---|---|---|---|
| COGS (generalization split, EM) | **5.81% +/- 1.34** | 2.27% +/- 1.94 | d=2.12, one-tailed p=0.034 | 3 |
| GSM8K (exact match) | **2.79% +/- 0.19** | 2.54% +/- 0.31 | d=0.96, one-tailed p=0.040 (caveat: mode collapse) | 8 |
| Spider (execution accuracy, with copy mechanism) | 0.60% +/- 0.60 | 0.90% +/- 0.17 | statistically indistinguishable | 3 |

See `paper/relattn_neurocomputing_submission.pdf` for the full results, ablations (slot count, gating, mixing, pairing strategy), and the mechanistic-analysis negative result, all with figures in `figures/`.

## Repository layout

- `relational_attention/` -- model implementation (attention, layers, model config)
- `scripts/` -- training and evaluation entry points
- `configs/` -- one YAML per experiment/ablation reported in the paper
- `analysis/` -- mechanistic probing, including both the retracted script (kept for transparency) and the corrected one
- `results/` -- raw evaluation outputs; see `results/README.md` for which files back which paper table, which are retracted, and which are still unverified
- `paper/` -- the paper itself (source, PDF, bibliography, figures) and the corrected supplemental material
- `figures/` -- the 6 figures used in the paper

## Reproducing the results

```bash
python scripts/train.py --config configs/<name>.yaml --seed <seed> --dataset <dataset> \
  --nas-dir <path-to-data> --output-dir <output-dir>

python scripts/evaluate.py --checkpoint <output-dir>/best_model --dataset <dataset> \
  --nas-dir <path-to-data> --split <dev|gen> --output-file <result>.json
```

Every config used in the paper is in `configs/`; every corresponding result JSON and training/evaluation log is in `results/` (see `results/README.md` and `results/verified_2026_revision/README.md` for the exact mapping to paper tables).

## Citation

A BibTeX entry will be added once the paper is accepted. In the meantime, please cite this repository directly if you build on it.

## License

MIT (see `LICENSE`).

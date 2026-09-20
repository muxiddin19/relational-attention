# Attribute-Decomposed Attention (RelAttn)

[![Status: submitted to Neurocomputing](https://img.shields.io/badge/status-submitted%20to%20Neurocomputing-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4-ee4c2c.svg)](https://pytorch.org/)

**Official implementation of "Attribute-Decomposed Attention: A Relational Inductive Bias for Structured Reasoning," submitted to Neurocomputing (Elsevier).**

> Attribute-Decomposed Attention decomposes each token into *k* typed attribute slots and computes attention via slot-to-slot join pairs, a neural analogue of a relational foreign-key join. Balance, path-completeness, and minimum-edge constraints characterize valid head-assignment strategies as the family of directed Hamiltonian cycles on the *k* slots; the cyclic pairing used throughout is the canonical instance, not a uniquely optimal one. The architecture gives a large observed gain on COGS compositional generalization (5.81%+/-1.34 vs. 2.27%+/-1.94 generalization-split EM, Cohen's d=2.12, 3 seeds) and a small difference on GSM8K (2.79%+/-0.19 vs. 2.54%+/-0.31, d=0.96, 8 seeds) that is better attributed to answer-frequency behavior than improved reasoning, with no measurable advantage on Spider once both architectures have a copy mechanism. Probing for emergent, role-specific attribute-slot specialization finds no discriminability signal above an untrained, randomly initialized control -- a negative interpretability result we report alongside the behavioral findings.

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

Balance, path-completeness, and minimum-edge constraints characterize valid head-assignment strategies as the family of directed Hamiltonian cycles on the *k* slots (see the paper's Theorem on head-assignment characterization) -- the cyclic pairing `(j, j+1 mod k)` used throughout is the canonical instance of this family, not a uniquely optimal one. A pairing-strategy ablation (`configs/*_ablation_random_fixed*.yaml`) shows a random-fixed assignment -- which does not itself satisfy the theorem's conditions -- performs statistically indistinguishably from cyclic, evidence against cyclic being uniquely advantageous rather than a confirmation of the theorem. Slot semantics (entity identity / functional-dependency operator / schema structure) were the design *intent*; whether they actually emerge is an empirical question the paper tests directly, and probing does not find evidence that they do (see Results below).

---

## Results

| Benchmark | RelTransformer | Standard Transformer | Effect | Seeds |
|---|---|---|---|---|
| COGS (generalization split, EM) | **5.81% +/- 1.34** | 2.27% +/- 1.94 | d=2.12, one-tailed p=0.034 | 3 |
| GSM8K (exact match) | **2.79% +/- 0.19** | 2.54% +/- 0.31 | d=0.96, one-tailed p=0.040, two-tailed p=0.079 | 8 |
| Spider (execution accuracy, with copy mechanism) | 0.60% +/- 0.60 | 0.90% +/- 0.17 | statistically indistinguishable | 3 |

GSM8K's difference should be read with the paper's mode-collapse caveat in mind (Sec. "Mathematical Reasoning"): both architectures' predictions concentrate heavily on a small set of common answers, so this is better characterized as a difference in answer-frequency behavior than a reasoning improvement.

Probing for emergent, role-specific attribute-slot specialization (Fisher discriminability, with an untrained random-init control and a permutation-null control) finds no signal above the untrained baseline -- see the paper's Mechanistic Analysis section.

See `paper/relattn_neurocomputing_submission.pdf` for the full results, ablations (slot count, gating, mixing, pairing strategy), and mechanistic analysis, with figures in `figures/`.

## Repository layout

- `relational_attention/` -- model implementation (attention, layers, model config)
- `scripts/` -- training and evaluation entry points
- `configs/` -- one YAML per experiment/ablation reported in the paper
- `analysis/` -- mechanistic probing; `probing_gsm8k_fixed.py` is the implementation used for the paper's results (see its module docstring and `probing.py`'s header comment for data-provenance notes on an earlier probing script kept in this repository for reference)
- `results/` -- raw evaluation outputs; see `results/README.md` for the file-to-table mapping and provenance notes on a few historical files
- `paper/` -- the paper itself (source, PDF, bibliography, figures) and the supplemental material
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

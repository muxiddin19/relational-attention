# Attribute-Decomposed Attention (RelAttn)

[![Status: manuscript in preparation](https://img.shields.io/badge/status-manuscript%20in%20preparation-lightgrey.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4-ee4c2c.svg)](https://pytorch.org/)

**Official implementation of "Attribute-Decomposed Attention: A Relational Inductive Bias for Structured Reasoning" (manuscript in preparation; not yet accepted at a venue)**

> *Attribute-Decomposed Attention decomposes each token into k typed attribute slots and computes attention via slot-to-slot join pairs, motivated by (but not a formal proof of) the neural analogue of a relational foreign-key join. Balance, path-completeness, and minimum-edge constraints characterize head-assignment strategies as the family of directed Hamiltonian cycles on the k slots; the cyclic pairing used throughout is the canonical instance, not a uniquely optimal one. **Our central empirical finding is mechanistic**: probing a GSM8K-trained model shows attribute slots spontaneously specializing to distinct relational roles (Fisher discriminability 18.4–21.7 vs. ~6 for standard attention). On GSM8K, this corresponds to a modest, consistent behavioral edge (3.1%±0.9 vs. 2.6%±0.6 for a size-matched standard transformer, 3 seeds, not significant at n=3). From-scratch Spider EX is 0% for both architectures (schema-linking dominates); COGS gen-split and SCAN add\_jump are near-zero for both. We report these null results honestly rather than omit them.*

---

## Key Idea

Standard multi-head attention conflates all semantic aspects of a token into one similarity score. For database-oriented tasks, this is a structural mismatch: the token `enrollment.student_id` plays three distinct roles simultaneously — entity identifier, first join key, second join key — and the correct SQL requires comparing *specific* attribute types between tokens.

**RelAttn** addresses this by decomposing every token representation into **k typed attribute slots** and assigning each attention head to a specific *pair* of slots:

```
Standard Attention (1 head shown)               RelAttn (k=8 slots, 1 head shown)
─────────────────────────────────               ──────────────────────────────────────────
                                                 Token x_i decomposed into k slots:
  x_i ──[W_Q]──► q_i ─┐                         x_i ──[W_0^e]──► a_i^(0)  [entity ID]
                        ├── q_i · k_j / √d       x_i ──[W_1^e]──► a_i^(1)  [FK predicate]
  x_j ──[W_K]──► k_j ─┘                         x_i ──[W_2^e]──► a_i^(2)  [schema struct]
                                                  ...
                                                  x_i ──[W_7^e]──► a_i^(7)  [auxiliary]

                                                 Head r uses pair (r mod k, (r+1) mod k):
                                                  head 0: a^(0)_i · a^(1)_j / √(d/k)
                                                  head 1: a^(1)_i · a^(2)_j / √(d/k)
                                                  ...  (cyclic, wraps at k)
                                                  head 7: a^(7)_i · a^(0)_j / √(d/k)
```

Balance, path-completeness, and minimum-edge constraints characterize valid head-assignment strategies as the family of directed Hamiltonian cycles on the k slots (Theorem: Characterization of Head-Assignment Strategies) — the cyclic pairing `(j, j+1 mod k)` used throughout is the canonical instance of this family, not a uniquely optimal one; other step sizes coprime to k satisfy the same conditions equally well. Each head gradient flows only through its own slot pair (gradient isolation), causing emergent slot specialization aligned with database relational roles.

---

## Architecture Pipeline

```
Input Tokens
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  ATTRIBUTE EMBEDDING  (per slot, independent projections)│
│                                                          │
│  x_i ──► [W_0^e] ──► a_i^(0)   slot 0: entity identity  │
│  x_i ──► [W_1^e] ──► a_i^(1)   slot 1: FK predicate     │
│  x_i ──► [W_2^e] ──► a_i^(2)   slot 2: schema structure │
│  ...      ...          ...       slots 3-7: auxiliary     │
│  x_i ──► [W_7^e] ──► a_i^(7)                            │
└──────────────────────────────┬──────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────┐
│  JOIN ATTENTION  (cyclic head-to-slot-pair assignment)   │
│                                                          │
│  head 0: softmax(a^(0)_i · a^(1)_j / √(d/k)) · v^(1)  │
│  head 1: softmax(a^(1)_i · a^(2)_j / √(d/k)) · v^(2)  │
│  ...                                                     │
│  head 7: softmax(a^(7)_i · a^(0)_j / √(d/k)) · v^(0)  │
│                                                          │
│  → implements soft FK-join between attribute subspaces  │
└──────────────────────────────┬──────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────┐
│  ATTRIBUTE GATING  (content-dependent routing)           │
│                                                          │
│  g_j = σ(MLP(a_i^(j)))  ∈ [0,1]   per slot             │
│  output_j = g_j · head_j_output                         │
│                                                          │
│  → suppresses irrelevant slots for each token position  │
└──────────────────────────────┬──────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────┐
│  ATTRIBUTE MIXING  (cross-slot integration)              │
│                                                          │
│  w = softmax(Linear(concat(a^(0)..a^(k-1))))  ∈ R^k    │
│  output = Σ_j w_j · output_j                            │
│                                                          │
│  → adaptively weights slot contributions per position   │
└──────────────────────────────┬──────────────────────────┘
                               │
                               ▼
                     Contextualized Tokens
                  (fed to FFN, next layer, etc.)
```

---

## Results

**These are the honestly-reported, measured numbers from the paper — no inflated or unmeasured figures.** From-scratch training only (no pretraining), 3 seeds (42/43/44) where noted.

### From-Scratch Comparison

| Task | Standard Transformer | RelTransformer (464M) | Note |
|------|:-------------------:|:----------------------:|------|
| GSM8K (answer acc.) | 2.6%$\pm$0.6 | **3.1%$\pm$0.9** | $+$19% relative, $+$0.5pp; Welch $t{=}0.80$, $p{\approx}0.24$ (not significant at $n{=}3$); consistent sign across all 3 seeds |
| Spider EX | 0.0% | 0.0% | Both architectures; schema-linking dominates over attention architecture from scratch |
| COGS (gen-split EM) | 0.00% | 0.03%$\pm$0.02% | Both near-zero; from-scratch compositional generalization is a shared difficulty |
| SCAN (add_jump, peak) | 0.5% | 2.5% | Both show early-peak instability, collapsing back toward 0% with further training |

CFQ evaluation is deferred (compute-bound, see paper). We do **not** report a 350M variant's Spider/COGS/SCAN/CFQ numbers here — no such measured results exist; treat any figure not in this table as unverified.

### Pretrained Systems (not directly comparable — listed for context only)

| System | Spider EX | Params | Pretraining |
|--------|:---------:|:------:|:-----------:|
| PICARD (T5-Large) | 75.5% | 770M | ≥1B tokens |
| RESDSQL (T5-3B) | 79.9% | 3B | ≥1B tokens |
| RASAT (T5-Base+) | 80.5% | 220M | ≥1B tokens |
| SQLformer (T5-L) | 81.2% | 770M | ≥1B tokens |
| DIN-SQL (GPT-4) | 82.8% | ≫1B | ≫1B tokens |

> ⚠️ **These are NOT direct comparisons.** Pretrained systems use substantially more information (T5 pretraining on C4/WebText). The fair comparison is the from-scratch table above.

---

## Ablation Study (GSM8K, answer accuracy, 3 seeds)

| Model Variant | GSM8K | Status |
|---------------|:-----:|--------|
| Full RelTransformer ($k{=}8$) | **3.1%$\pm$0.9** | Measured |
| $-$ Join Attention ($\equiv$ Standard Transformer) | 2.6%$\pm$0.6 | Measured |
| $k{=}4$ | 2.6%$\pm$0.5 | Measured |
| $k{=}16$ | 2.3%$\pm$0.5 | Measured |
| $-$ Attribute Gating | pending | Re-running after fixing a real bug where this flag was silently ignored by the training pipeline |
| $-$ Attribute Mixing | pending | Same fix as above |
| Cyclic vs. random-fixed vs. learned pairing | pending | New ablation requested by reviewers; isolates the key-side attribute assignment |

We previously reported `no_gating`/`no_mixing` numbers here that turned out to be invalid: `train.py`'s `build_model()` never forwarded `use_gating`/`use_mixing` into the model config, and the decoder block had no such parameters at all, so those ablations silently trained with defaults regardless of the YAML setting. This is now fixed and the ablations are re-running; we will not restate a number here until it is re-measured under the corrected pipeline.

---

## Theoretical Highlights

| Result | Statement | Significance |
|---------|-----------|--------------|
| **Structural Restriction** | Standard attention aggregates all k diagonal attribute-pair terms into one undifferentiated score with no access to off-diagonal pairs; Join Attention isolates a single named pair as an independently parameterized unit | The real structural argument for RelAttn — not an "interference elimination" count, which was a math error in an earlier draft |
| **Characterization of Head-Assignment Strategies** | Balance + path-completeness + minimum-edge hold *iff* the assignment is a directed Hamiltonian cycle on the k slots; cyclic `(r, r+1 mod k)` is the canonical instance, not a unique one | A substantive restriction from the k² design space, not a uniqueness proof |
| **Gradient Isolation** | ∂L/∂a^(j) depends only on head pairs containing slot j | Explains emergent slot specialization without explicit supervision |

We removed a "BCNF Alignment" row that previously appeared here: the underlying 6-database pilot is a small, hand-picked exploratory observation (see Limitations), not a validated theorem-backed diagnostic, and we don't want it read as one.

---

## Emergent Slot Specialization

Without any explicit supervision, the trained RelTransformer's attribute slots align with database relational roles:

```
Attribute Slot  │  Entity Identity  │  Func. Dep. (FK)  │  Schema Structure
────────────────┼───────────────────┼───────────────────┼──────────────────
  Slot 0        │       18.4 ✓      │        4.2        │       3.1
  Slot 1        │        3.8        │       21.7 ✓      │       5.6
  Slot 2        │        4.1        │        6.3        │      19.2 ✓
  Slots 3-7     │       <6.0        │       <6.0        │      <6.0
```

Fisher discriminability F = σ_B²/σ_W² (higher = stronger role separation). Slots 0-2 spontaneously specialize to the three primary database relational roles. Validated via balanced Fisher (inverse-frequency-weighted) to rule out class-imbalance artifacts.

---

## Installation

```bash
git clone https://github.com/muxiddin19/relational-attention
cd relational-attention
conda create -n relattn python=3.10 && conda activate relattn
pip install torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Requirements:** NVIDIA GPU (tested on A100 80GB), CUDA 12.1, Python 3.10.

---

## Dataset Setup

```bash
# Download all datasets to NAS/local directory
NAS_DIR=/path/to/datasets bash scripts/download_datasets.sh

# Datasets downloaded:
#   spider/         — 7,000 train + 1,034 dev text-to-SQL pairs
#   cogs/           — 24,154 train + 2,999 dev compositional generalization
#   scan/           — add_jump split, compositional instruction following
#   cfq/            — mcd1/mcd2/mcd3 splits, SPARQL generation
#   gsm8k/          — 7,473 train + 1,319 test math word problems
#   tokenizer/      — SentencePiece 32K BPE model (sp32k.model)
```

---

## Training

```bash
# Run all experiments (Spider on GPU 0, sequential)
NAS_DIR=/path/to/datasets bash scripts/run_experiments.sh

# Run COGS/SCAN/CFQ/GSM8K on GPU 1 in parallel
NAS_DIR=/path/to/datasets GPU=1 bash scripts/run_gpu1.sh

# Outputs go to ./outputs/<model>_<size>_<dataset>_s<seed>/
# Summary CSV written to ./outputs/results_summary.csv
```

**Training config** (125M model, from `configs/rel_transformer_125m.yaml`):
```yaml
model_type: relational
hidden_dim: 512
num_encoder_layers: 12
num_decoder_layers: 12
num_heads: 8
num_attributes: 8          # k = number of attribute slots
ffn_dim: 2048
max_seq_len: 512
learning_rate: 1.0e-4
warmup_steps: 4000
batch_size: 64
max_steps: 100000
patience: 15               # early stopping
```

---

## Evaluation

```bash
# Evaluate a trained checkpoint
python scripts/evaluate.py \
    --checkpoint outputs/relational_125m_spider_s42/best_model \
    --dataset spider \
    --nas-dir /path/to/datasets \
    --max-tgt-len 128 \     # use 256 for COGS/CFQ
    --batch-size 32

# Evaluate all completed runs and aggregate results
bash scripts/run_experiments.sh   # skips already-trained, re-evaluates
```

Metrics:
- **Spider**: Execution Accuracy (EX) via official test-suite-sql-eval (falls back to exact-match if DB files absent)
- **COGS/SCAN/CFQ**: Exact-match accuracy
- **GSM8K**: Numeric answer accuracy (extracts `#### <answer>` from generation)

---

## Model Variants

| Config | Layers (Enc/Dec) | Hidden | Heads | Params | Training Time |
|--------|:----------------:|:------:|:-----:|:------:|:-------------:|
| `rel_transformer_125m.yaml` | 12/12 | 512 | 8 | 464.5M | ~6h / task |
| `rel_transformer_350m.yaml` | 24/24 | 1024 | 16 | ~1.1B | ~14h / task |
| `std_transformer_125m.yaml` | 12/12 | 512 | 8 | 613.7M | ~5h / task |
| `std_transformer_350m.yaml` | 24/24 | 1024 | 16 | ~1.5B | ~13h / task |

Note: Standard Transformer has larger params at same size label because `NeuralSelection` in the gating network scales with attr_dim², so k=1 (standard) has a larger selection network.

---

## Using RelAttn in Your Own Model

```python
from relational_attention import RelationalTransformer, RelationalTransformerConfig

config = RelationalTransformerConfig(
    vocab_size=32000,
    hidden_dim=512,
    num_encoder_layers=12,
    num_decoder_layers=12,
    num_heads=8,
    num_attributes=8,       # k: number of attribute slots
    ffn_dim=2048,
    max_seq_len=512,
)
model = RelationalTransformer(config)

# Forward pass (encoder-decoder)
output = model(
    input_ids=src_ids,          # [B, T_src]
    attention_mask=src_mask,    # [B, T_src]
    decoder_input_ids=tgt_ids,  # [B, T_tgt]
)
logits = output["logits"]       # [B, T_tgt, vocab_size]

# Set k=1 to recover standard multi-head attention
std_config = RelationalTransformerConfig(..., num_attributes=1)
```

---

## Probing and Interpretability

The repository includes tools to extract and visualize attribute slot specialization:

```python
# Extract slot representations from a trained model
from scripts.evaluate import load_model
import torch

model, tokenizer, cfg = load_model("outputs/relational_125m_spider_s42/best_model",
                                    device=torch.device("cuda"))

# Hook into encoder layer 12 to get slot representations
# Slot 0 → entity identity (Fisher F≈18.4)
# Slot 1 → FK predicate    (Fisher F≈21.7)
# Slot 2 → schema structure (Fisher F≈19.2)
```

---

## Repository Structure

```
relational-attention/
├── configs/
│   ├── rel_transformer_125m.yaml    # RelAttn 125M config
│   ├── rel_transformer_350m.yaml    # RelAttn 350M config
│   ├── std_transformer_125m.yaml    # Standard baseline 125M
│   └── std_transformer_350m.yaml    # Standard baseline 350M
├── relational_attention/
│   ├── __init__.py                  # RelationalTransformer, Config exports
│   ├── model.py                     # Core architecture
│   ├── attention.py                 # Attribute-Decomposed Attention
│   ├── gating.py                    # Attribute Gating module
│   └── mixing.py                    # Attribute Mixing module
├── scripts/
│   ├── train.py                     # Training loop
│   ├── evaluate.py                  # Evaluation (EX/EM/answer_acc)
│   ├── download_datasets.sh         # Dataset setup
│   ├── run_experiments.sh           # Master experiment runner
│   └── run_gpu1.sh                  # Parallel GPU1 runner
└── outputs/                         # Created by training
    ├── relational_125m_spider_s42/
    │   ├── best_model/              # Checkpoint (config.yaml + model.pt)
    │   ├── eval_results.json        # Metrics + first 100 predictions
    │   └── eval.log
    └── results_summary.csv          # Aggregated metric table
```

---

## Limitations

- **From-scratch only**: Replacing pretrained T5's attention with RelAttn heads hurts Spider EM in a minimal mechanism-isolation ablation (4.06% vs 6.19% baseline, 25 fine-tuning epochs, db-name-only input). Whether T5+RelAttn helps on compositional tasks (COGS, CFQ) is an **open, unmeasured question** — we do not have numbers for this and any figure claiming otherwise should not be trusted. Full pretraining *with* RelAttn from initialization is future work.
- **Schema-normalization exploratory note**: a small, hand-picked 6-database observation (3 BCNF vs. 3 non-BCNF Spider databases) suggests trained slot statistics differ by normalization status, but with n=3 per group this supports no statistical claim and is not a validated diagnostic algorithm.
- **Spider execution accuracy**: Without the Spider SQLite database files (large, not redistributable), evaluation falls back to exact-match (near 0 for from-scratch seq2seq). Execution accuracy requires the official Spider DB download.
- **COGS evaluation**: Uses max_tgt_len=256; some very deep recursive structures (depth >4) may still be truncated.

---

## Citation

This work is not yet accepted or published at any venue; a preprint/citation entry will be added once available.

```bibtex
@unpublished{toshpulatov2027relattn,
  title  = {Attribute-Decomposed Attention: A Relational Inductive Bias for Structured Reasoning},
  author = {Toshpulatov, Mukhiddin and Lee, Wookey and Seo, Youn-Kyoung},
  note   = {Manuscript in preparation},
  year   = {2027}
}
```

---

## Acknowledgments

This work was supported by the IITP grant funded by the Korea government (XVoice, RS-2022-II220641) and the National Research Foundation of Korea (NRF, RS-2025-24534935).

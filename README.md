# Attribute-Decomposed Attention

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **"Attribute-Decomposed Attention: A Relational Inductive Bias for Structured Reasoning"** (ICDE 2027).

**Authors:** Mukhiddin Toshpulatov, Wookey Lee, Youn-Kyoung Seo — Gachon University / Inha University

## Abstract

We introduce **Attribute-Decomposed Attention** (RelAttn), an attention mechanism grounded in the relational model that treats token representations as typed attribute tuples. The design implements database operations—selection (σ), projection (π), and join (⋈)—as differentiable attention operations, providing a principled relational inductive bias for structured reasoning.

The key innovation is **Join Attention**, which computes attention based on specific attribute pairs rather than full representations: each head implements a soft foreign-key join between typed attribute subspaces, enabling targeted structured reasoning with systematic compositional generalization.

## Key Features

- **Tuple Embeddings**: Each token is represented as a tuple of K typed attributes
- **Attribute-Specific Attention**: Different heads attend based on different attribute pairs (join conditions)
- **Schema-Aware Encoding**: Optional relation/table and column embeddings for structured inputs
- **Type-Constrained Decoding**: Mask invalid tokens during generation for structured outputs

## Installation

```bash
git clone https://github.com/muxiddin19/relational-attention.git
cd relational-attention
pip install -e ".[experiments]"
```

**Requirements:** Python ≥ 3.8, PyTorch ≥ 2.0

## Quick Start

```python
from relational_attention import RelationalTransformer, RelationalTransformerConfig

# Configure model
config = RelationalTransformerConfig(
    vocab_size=32000,
    hidden_dim=768,
    num_encoder_layers=12,
    num_decoder_layers=12,
    num_heads=12,
    num_attributes=8,  # K attributes per token
)

# Create model
model = RelationalTransformer(config)

# Forward pass
output = model(
    input_ids=src_tokens,
    decoder_input_ids=tgt_tokens,
    labels=labels
)
loss = output['loss']
```

## Architecture Overview

<p align="center">
  <img src="docs/architecture.png" alt="Relational Attention Architecture" width="700"/>
</p>

### Core Components

| Component | Description | Equation |
|-----------|-------------|----------|
| **Neural Selection (σ̃)** | Attribute-conditional gating | g = σ(MLP(a)) |
| **Neural Projection (π̃)** | Learned attribute combination | Σᵢ wᵢ · aᵢ |
| **Join Attention (⋈̃)** | Attribute-specific similarity | α = softmax(aᵢ⁽ʲ⁾ · bₜ⁽ˡ⁾ / τ) |

### Join Attention

The key innovation—attention based on specific attribute pairs:

```
α_it = softmax_t( (a_i^{(j)} · b_t^{(l)}) / τ )
output_i = Σ_t α_it · v_t
```

Where `a_i^{(j)}` is the j-th attribute of query token i, and `b_t^{(l)}` is the l-th attribute of key token t.

## Documentation

For detailed documentation, see [`relational_attention/README.md`](relational_attention/README.md), including:

- Complete API reference
- Tensor shape conventions
- Configuration options
- Usage examples (Text-to-SQL, classification, etc.)

## Examples

```bash
# Run all examples
python relational_attention/examples/basic_usage.py

# Run tests
python relational_attention/tests/test_attention.py
```

## Model Configurations

| Model | Parameters | Hidden | Layers | Heads | Attributes |
|-------|------------|--------|--------|-------|------------|
| Small | ~125M | 768 | 12+12 | 12 | 8 |
| Medium | ~350M | 1024 | 24+24 | 16 | 8 |
| Large | ~1.3B | 2048 | 24+24 | 16 | 8 |

## Complexity

Relational Attention maintains **O(n²d)** complexity, matching standard attention, with ~1.3× overhead due to attribute projections.

## Citation

```bibtex
@inproceedings{toshpulatov2027relational,
  title     = {Attribute-Decomposed Attention: A Relational Inductive Bias for Structured Reasoning},
  author    = {Toshpulatov, Mukhiddin and Lee, Wookey and Seo, Youn-Kyoung},
  booktitle = {Proceedings of the 43rd IEEE International Conference on Data Engineering (ICDE)},
  year      = {2027},
  address   = {Copenhagen, Denmark}
}
```

## Reproducing Experiments

See [`scripts/`](scripts/) for the full experiment pipeline:

```bash
# 1. Server environment
bash scripts/setup_env.sh

# 2. Download all datasets to NAS
NAS_DIR=/nas/Dataset bash scripts/download_datasets.sh

# 3. Run all experiments (3 seeds × 2 sizes × 5 datasets)
NAS_DIR=/nas/Dataset bash scripts/run_experiments.sh
```

See [Fairness of Comparisons](#fairness-of-comparisons) below for details on ensuring identical conditions across models.

## Fairness of Comparisons

All from-scratch comparisons (RelTransformer vs Standard Transformer) use:
- Identical architecture depth, width, and parameter count
- Identical training data, tokenizer (SentencePiece, 32k vocab), and batch schedule
- Identical constrained decoding applied uniformly to all models
- 3 seeds {42, 43, 44}; results reported as mean ± std
- Spider: official `evaluation.py` from the Spider repository (EX + EM)

PICARD and RESDSQL use pretrained T5-Large/3B — not size-matched; presented as reference only (paper §V).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

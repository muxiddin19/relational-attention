# Relational Attention

A PyTorch implementation of Relational Attention, an attention mechanism grounded in relational algebra that treats token representations as tuples with typed attributes.

> **Paper:** *Relational Attention Is All You Need: A Set-Theoretic Foundation for Neural Structured Reasoning*
> **Venue:** EMNLP 2026 (Submitted: January 10, 2026)

## Overview

Relational Attention reformulates the attention mechanism through the lens of relational algebra, providing:

- **Structured token representations**: Each token is represented as a tuple of K typed attributes
- **Attribute-specific attention**: Different attention heads attend based on different attribute pairs (join conditions)
- **Schema-aware encoding**: Optional relation/table and column embeddings for structured inputs
- **Type-constrained decoding**: Mask invalid tokens during generation for structured outputs (e.g., SQL)

The key innovation is the **Join Attention** mechanism, which computes attention scores based on specific attribute pairs rather than the full representation, enabling more targeted relational reasoning.

## Installation

```bash
# Ensure the package is in your Python path
cd /path/to/relational_attention
pip install torch  # Requires PyTorch >= 1.9

# Or install in development mode
pip install -e .
```

**Requirements:**
- Python >= 3.8
- PyTorch >= 1.9

## Quick Start

```python
from relational_attention import RelationalTransformer, RelationalTransformerConfig

# Create configuration
config = RelationalTransformerConfig(
    vocab_size=32000,
    hidden_dim=768,
    num_encoder_layers=12,
    num_decoder_layers=12,
    num_heads=12,
    num_attributes=8,  # K=8 attributes per token
    max_seq_len=512
)

# Create model
model = RelationalTransformer(config)

# Forward pass
output = model(
    input_ids=input_ids,           # (B, S) source tokens
    decoder_input_ids=decoder_ids, # (B, T) target tokens
    labels=labels                  # (B, T) for computing loss
)

loss = output['loss']
logits = output['logits']  # (B, T, V)
```

## Architecture

### Tensor Shape Conventions

Throughout this implementation, we use consistent shape conventions:

| Symbol | Meaning |
|--------|---------|
| B | Batch size |
| S | Source sequence length |
| T | Target sequence length |
| D | Hidden dimension |
| K | Number of attributes |
| A | Attribute dimension (D // K) |
| H | Number of attention heads |
| V | Vocabulary size |

### Core Components

#### 1. Tuple Embedding

Each token is embedded as a tuple of K typed attributes. The hidden dimension D is divided into K attribute slots, each of dimension A = D/K.

```python
from relational_attention import TupleEmbedding

embedding = TupleEmbedding(
    vocab_size=32000,
    hidden_dim=768,      # D = 768
    num_attributes=8     # K = 8, so A = 96 per attribute
)

# Input: (B, S) token indices
# Output: (B, S, D) where D is organized as K concatenated attributes
output = embedding(input_ids)

# Access individual attributes
attributes = embedding.get_attributes(output)  # List of K tensors, each (B, S, A)
```

**How attributes are determined:** Attributes are learned end-to-end via K separate linear projections from the base token embedding. Each projection `W_j^{(e)}` learns to extract different semantic aspects (e.g., entity type, syntactic role, relational context). No explicit supervision is required—the model learns attribute semantics through task gradients.

#### 2. Neural Selection (σ̃)

Attribute-conditional gating that filters representations based on learned predicates:

```python
from relational_attention import NeuralSelection

selection = NeuralSelection(attr_dim=96)

# x: (B, S, D) - input representations
# attribute: (B, S, A) - conditioning attribute
# output: (B, S, D) - gated output (same shape as x)
output = selection(x, attribute)
```

The selection gate `g = σ(MLP(attribute))` produces values in [0, 1] that modulate the input.

#### 3. Neural Projection (π̃)

Learned weighted combination of K attributes:

```python
from relational_attention import NeuralProjection

projection = NeuralProjection(
    num_attributes=8,   # K
    attr_dim=96,        # A
    output_dim=768      # D
)

# attributes: List of K tensors, each (B, S, A)
# output: (B, S, D)
output = projection(attributes)
```

Uses softmax-normalized learnable weights to combine attributes.

#### 4. Join Attention (⋈̃) — The Key Innovation

Computes attention based on specific attribute pairs, not the full representation:

```python
from relational_attention import JoinAttention

join_attn = JoinAttention(
    attr_dim=96,        # A
    temperature=1.0,    # τ for scaling
    dropout=0.1
)

# query_attr: (B, S_q, A) - query's j-th attribute
# key_attr: (B, S_k, A) - key's l-th attribute
# value: (B, S_k, D_v) - full value representation
# output: (B, S_q, D_v), attn_weights: (B, S_q, S_k)
output, attn_weights = join_attn(query_attr, key_attr, value)
```

**Equation (from paper):**
```
α_it = softmax_t( (a_i^{(j)} · b_t^{(l)}) / τ )
output_i = Σ_t α_it · v_t
```

Where:
- `a_i^{(j)}` is the j-th attribute of the i-th query token (shape: A)
- `b_t^{(l)}` is the l-th attribute of the t-th key token (shape: A)
- `v_t` is the full value vector (shape: D_v)
- τ is a learnable temperature parameter

#### 5. Relational Attention Head

Combines join attention with selection and projection:

```python
from relational_attention import RelationalAttentionHead

head = RelationalAttentionHead(
    hidden_dim=768,
    num_attributes=8,
    query_attr_idx=0,  # j - which query attribute to use
    key_attr_idx=1,    # l - which key attribute to use
    dropout=0.1
)

# query, key, value: (B, S, D)
# output: (B, S, D), attn: (B, S, S)
output, attn = head(query, key, value)
```

**Operation:** `RelAttn(Q, K, V) = W_o · σ̃(⋈̃^{(j,l)}(W_q·Q, W_k·K, W_v·V))`

#### 6. Multi-Relation Attention

Multiple heads with different attribute pairs:

```python
from relational_attention import MultiRelationAttention

multi_attn = MultiRelationAttention(
    hidden_dim=768,
    num_heads=12,       # H heads
    num_attributes=8,   # K attributes
    dropout=0.1
)

# Each head r uses attribute pair (j_r, l_r) = (r % K, (r+1) % K)
# query, key, value: (B, S, D)
# output: (B, S, D)
output, attn = multi_attn(query, key, value, return_attention=True)
```

### Schema-Aware Positional Encoding

For structured inputs like database schemas:

```python
from relational_attention import SchemaAwarePositionalEncoding

pos_enc = SchemaAwarePositionalEncoding(
    hidden_dim=768,
    max_seq_len=2048,
    max_relations=64,   # Max number of tables
    max_columns=128     # Max number of column types
)

# x: (B, S, D)
# relation_ids: (B, S) - which table each token belongs to
# column_ids: (B, S) - which column type
output = pos_enc(x, relation_ids, column_ids)
```

Position encoding: `p_i = p_i^{(seq)} + p_{r(i)}^{(rel)} + p_{c(i)}^{(col)}`

### Type-Constrained Decoding

For structured generation (SQL, code, JSON):

```python
# Create mask for valid tokens at each position
valid_token_mask = torch.ones(batch, vocab_size, dtype=torch.bool)
valid_token_mask[:, invalid_token_ids] = False

output = model(
    input_ids=input_ids,
    decoder_input_ids=decoder_ids,
    valid_token_mask=valid_token_mask  # (B, V) or (B, T, V)
)
```

Invalid tokens receive `-inf` logits before softmax.

## Model Configurations

```python
RelationalTransformerConfig(
    # Architecture
    vocab_size=32000,
    hidden_dim=768,           # D
    num_encoder_layers=12,
    num_decoder_layers=12,
    num_heads=12,             # H
    num_attributes=8,         # K (key hyperparameter)
    ffn_dim=None,             # Defaults to 4 * hidden_dim
    max_seq_len=2048,

    # Regularization
    dropout=0.1,

    # Embedding
    padding_idx=0,
    tie_embeddings=True,

    # Attribute learning
    attribute_mode='learned',  # Attributes learned end-to-end

    # Join attention
    join_temperature=1.0,      # τ (key hyperparameter)
    learnable_temperature=True # Whether τ is trainable
)
```

### Key Hyperparameters

| Parameter | Symbol | Default | Notes |
|-----------|--------|---------|-------|
| `num_attributes` | K | 8 | Number of attribute slots per token. Higher K = more fine-grained decomposition but smaller per-attribute dimension A=D/K |
| `join_temperature` | τ | 1.0 | Temperature for attention softmax. Lower = sharper attention |
| `learnable_temperature` | - | True | If True, τ is learned per head |

### Model Sizes

| Config | Params | hidden_dim | layers | heads | attributes |
|--------|--------|------------|--------|-------|------------|
| Small  | ~125M  | 768        | 12+12  | 12    | 8          |
| Medium | ~350M  | 1024       | 24+24  | 16    | 8          |
| Large  | ~1.3B  | 2048       | 24+24  | 16    | 8          |

## Complexity

Relational Attention maintains **O(n²d)** complexity, matching standard attention:
- Join attention: O(n² · A) per head for attribute comparison
- With H heads and K attributes: O(H · n² · D/K) = O(n²d)
- Overhead vs standard attention: ~1.3× due to attribute projections

## Usage Examples

### Basic Encoder-Decoder

```python
from relational_attention import RelationalTransformer, RelationalTransformerConfig

config = RelationalTransformerConfig(
    vocab_size=10000,
    hidden_dim=256,
    num_encoder_layers=4,
    num_decoder_layers=4,
    num_heads=8,
    num_attributes=8
)

model = RelationalTransformer(config)

# Training
output = model(
    input_ids=src_tokens,        # (B, S)
    decoder_input_ids=tgt_tokens,# (B, T)
    labels=tgt_labels            # (B, T)
)
loss = output['loss']
loss.backward()

# Generation
model.eval()
generated = model.generate(
    input_ids=src_tokens,
    max_length=100,
    do_sample=True,
    temperature=0.8,
    top_k=50
)
```

### Text-to-SQL with Schema Encoding

```python
# Assign relation IDs based on table membership
relation_ids = torch.zeros(batch, seq_len, dtype=torch.long)
relation_ids[:, schema_start:table1_end] = 1  # Table 1 tokens
relation_ids[:, table1_end:table2_end] = 2    # Table 2 tokens

# Assign column type IDs
column_ids = torch.zeros(batch, seq_len, dtype=torch.long)
column_ids[:, name_col_positions] = 1
column_ids[:, id_col_positions] = 2

output = model(
    input_ids=input_ids,
    decoder_input_ids=decoder_ids,
    relation_ids=relation_ids,
    column_ids=column_ids,
    valid_token_mask=sql_grammar_mask  # Constrain to valid SQL
)
```

### Sequence Classification

```python
from relational_attention import RelationalTransformerForSequenceClassification

model = RelationalTransformerForSequenceClassification(config, num_classes=3)

output = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)

predictions = output['logits'].argmax(dim=-1)
```

### Analyzing Attention Patterns

```python
# Get attention weights for interpretability
output, attn = model.encoder(input_ids, return_attention=True)

# attn is a list of (H, B, S, S) tensors per layer
# Each head attends based on different attribute pairs
for layer_idx, layer_attn in enumerate(attn):
    for head_idx in range(layer_attn.size(0)):
        # Head h joins attributes (h % K) and ((h+1) % K)
        head_weights = layer_attn[head_idx]  # (B, S, S)
```

## Running Tests

```bash
cd relational_attention

# Run unit tests (without pytest)
python tests/test_attention.py

# Run examples
python examples/basic_usage.py
```

## File Structure

```
relational_attention/
├── __init__.py              # Package exports
├── attention.py             # Core attention mechanisms
│   ├── NeuralSelection      # σ̃ - attribute gating
│   ├── NeuralProjection     # π̃ - attribute combination
│   ├── JoinAttention        # ⋈̃ - attribute-specific attention
│   ├── RelationalAttentionHead
│   └── MultiRelationAttention
├── layers.py                # Transformer building blocks
│   ├── TupleEmbedding
│   ├── SchemaAwarePositionalEncoding
│   ├── FeedForward
│   ├── RelationalTransformerBlock
│   └── RelationalTransformerEncoderBlock
├── model.py                 # Complete models
│   ├── RelationalTransformerConfig
│   ├── RelationalTransformerEncoder
│   ├── RelationalTransformerDecoder
│   ├── TypeConstrainedDecoder
│   ├── RelationalTransformer
│   └── RelationalTransformerForSequenceClassification
├── examples/
│   └── basic_usage.py       # Usage examples
└── tests/
    └── test_attention.py    # Unit tests
```

## Relationship to Standard Attention

When `num_attributes=1` (K=1), Relational Attention reduces to standard multi-head attention:
- Single attribute = full representation
- Join attention = standard scaled dot-product attention
- No attribute decomposition overhead

This provides a smooth interpolation between standard attention (K=1) and fully factorized relational attention (K=D).

## Citation

```bibtex
@inproceedings{relational-attention-2026,
  title={Relational Attention Is All You Need: A Set-Theoretic Foundation
         for Neural Structured Reasoning},
  author={Anonymous},
  booktitle={Proceedings of EMNLP},
  year={2026}
}
```

## License

MIT License

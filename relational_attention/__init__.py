"""
Relational Attention: A Set-Theoretic Foundation for Neural Structured Reasoning

This package provides a PyTorch implementation of Relational Attention,
an attention mechanism grounded in relational algebra that treats token
representations as tuples with typed attributes.

Key Components:
- RelationalAttention: Drop-in replacement for standard attention
- RelationalTransformer: Complete encoder-decoder model
- TupleEmbedding: Embeds tokens as tuples with K attributes
- JoinAttention: Attribute-specific attention mechanism

Example:
    >>> from relational_attention import RelationalTransformer, RelationalTransformerConfig
    >>>
    >>> config = RelationalTransformerConfig(
    ...     vocab_size=32000,
    ...     hidden_dim=768,
    ...     num_encoder_layers=12,
    ...     num_decoder_layers=12,
    ...     num_heads=12,
    ...     num_attributes=8
    ... )
    >>> model = RelationalTransformer(config)
    >>>
    >>> # Forward pass
    >>> output = model(input_ids, decoder_input_ids, labels=labels)
    >>> loss = output['loss']
    >>> logits = output['logits']

Reference:
    "Relational Attention Is All You Need: A Set-Theoretic Foundation
    for Neural Structured Reasoning"
"""

__version__ = "0.1.0"
__author__ = "Anonymous"

# Core attention components
from .attention import (
    NeuralSelection,
    NeuralProjection,
    JoinAttention,
    RelationalAttentionHead,
    MultiRelationAttention,
    RelationalAttention,
)

# Layers
from .layers import (
    TupleEmbedding,
    SchemaAwarePositionalEncoding,
    FeedForward,
    RelationalTransformerBlock,
    RelationalTransformerEncoderBlock,
)

# Models
from .model import (
    RelationalTransformerConfig,
    RelationalTransformerEncoder,
    RelationalTransformerDecoder,
    TypeConstrainedDecoder,
    RelationalTransformer,
    RelationalTransformerForSequenceClassification,
)

__all__ = [
    # Version
    "__version__",

    # Attention
    "NeuralSelection",
    "NeuralProjection",
    "JoinAttention",
    "RelationalAttentionHead",
    "MultiRelationAttention",
    "RelationalAttention",

    # Layers
    "TupleEmbedding",
    "SchemaAwarePositionalEncoding",
    "FeedForward",
    "RelationalTransformerBlock",
    "RelationalTransformerEncoderBlock",

    # Models
    "RelationalTransformerConfig",
    "RelationalTransformerEncoder",
    "RelationalTransformerDecoder",
    "TypeConstrainedDecoder",
    "RelationalTransformer",
    "RelationalTransformerForSequenceClassification",
]

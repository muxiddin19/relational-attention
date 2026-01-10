#!/usr/bin/env python3
"""
Basic Usage Examples for Relational Attention

This script demonstrates how to use the Relational Attention implementation
for various tasks including:
1. Basic forward pass
2. Text-to-SQL generation with type constraints
3. Sequence classification
4. Examining attention patterns
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '..')

from relational_attention import (
    RelationalTransformer,
    RelationalTransformerConfig,
    RelationalTransformerForSequenceClassification,
    JoinAttention,
    MultiRelationAttention,
)


def example_1_basic_forward():
    """Example 1: Basic forward pass with Relational Transformer."""
    print("=" * 60)
    print("Example 1: Basic Forward Pass")
    print("=" * 60)

    # Create configuration
    config = RelationalTransformerConfig(
        vocab_size=10000,
        hidden_dim=256,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_heads=8,
        num_attributes=8,  # k=8 attributes per token
        max_seq_len=512,
        dropout=0.1
    )

    # Create model
    model = RelationalTransformer(config)
    print(f"Model parameters: {model.get_num_parameters():,}")
    print(f"Trainable parameters: {model.get_num_parameters(trainable_only=True):,}")

    # Create dummy inputs
    batch_size = 2
    src_len = 32
    tgt_len = 16

    input_ids = torch.randint(1, config.vocab_size, (batch_size, src_len))
    decoder_input_ids = torch.randint(1, config.vocab_size, (batch_size, tgt_len))
    attention_mask = torch.ones(batch_size, src_len)
    labels = torch.randint(1, config.vocab_size, (batch_size, tgt_len))

    # Forward pass
    output = model(
        input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    print(f"Logits shape: {output['logits'].shape}")  # (B, T, V)
    print(f"Loss: {output['loss'].item():.4f}")
    print()


def example_2_text_to_sql():
    """Example 2: Text-to-SQL with schema-aware encoding and type constraints."""
    print("=" * 60)
    print("Example 2: Text-to-SQL Generation")
    print("=" * 60)

    # Smaller model for demo
    config = RelationalTransformerConfig(
        vocab_size=5000,
        hidden_dim=256,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_heads=8,
        num_attributes=8,
        max_seq_len=256
    )

    model = RelationalTransformer(config)

    # Simulate text-to-SQL input
    # Input: "Find employees in Sales department earning > 50k"
    batch_size = 1
    src_len = 20
    tgt_len = 15

    # Token IDs (simulated)
    input_ids = torch.randint(1, 1000, (batch_size, src_len))

    # Schema-aware encoding: assign relation IDs (table membership)
    # e.g., tokens belonging to different tables in the schema
    relation_ids = torch.zeros(batch_size, src_len, dtype=torch.long)
    relation_ids[:, 5:10] = 1  # "employees" table tokens
    relation_ids[:, 10:15] = 2  # "departments" table tokens

    # Column type IDs
    column_ids = torch.zeros(batch_size, src_len, dtype=torch.long)
    column_ids[:, 5:7] = 1  # name column
    column_ids[:, 7:9] = 2  # salary column

    # Decoder input (shifted target)
    decoder_input_ids = torch.randint(1, 500, (batch_size, tgt_len))

    # Type constraints: only allow SQL keywords at certain positions
    # This is a simplified example - in practice, this would be computed
    # based on grammar rules and current generation state
    valid_token_mask = torch.ones(batch_size, config.vocab_size, dtype=torch.bool)
    # Mask out non-SQL tokens (simplified)
    valid_token_mask[:, 500:] = False

    # Forward pass with schema encoding
    output = model(
        input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        relation_ids=relation_ids,
        column_ids=column_ids,
        valid_token_mask=valid_token_mask
    )

    print(f"Logits shape: {output['logits'].shape}")

    # Generation example
    print("\nGenerating SQL...")
    generated = model.generate(
        input_ids=input_ids,
        relation_ids=relation_ids,
        column_ids=column_ids,
        max_length=30,
        temperature=0.8,
        do_sample=True
    )
    print(f"Generated shape: {generated.shape}")
    print()


def example_3_sequence_classification():
    """Example 3: Sequence classification (e.g., sentiment, NLI)."""
    print("=" * 60)
    print("Example 3: Sequence Classification")
    print("=" * 60)

    config = RelationalTransformerConfig(
        vocab_size=10000,
        hidden_dim=256,
        num_encoder_layers=4,
        num_decoder_layers=4,  # Not used in encoder-only
        num_heads=8,
        num_attributes=8
    )

    # Classification model
    num_classes = 3  # e.g., positive, negative, neutral
    model = RelationalTransformerForSequenceClassification(config, num_classes)

    # Dummy input
    batch_size = 4
    seq_len = 64
    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, num_classes, (batch_size,))

    # Forward pass
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    print(f"Logits shape: {output['logits'].shape}")  # (B, num_classes)
    print(f"Loss: {output['loss'].item():.4f}")

    # Get predictions
    predictions = output['logits'].argmax(dim=-1)
    accuracy = (predictions == labels).float().mean()
    print(f"Accuracy: {accuracy.item():.2%}")
    print()


def example_4_attention_patterns():
    """Example 4: Examining attention patterns for interpretability."""
    print("=" * 60)
    print("Example 4: Attention Pattern Analysis")
    print("=" * 60)

    # Create a single Join Attention layer
    attr_dim = 64
    join_attn = JoinAttention(attr_dim=attr_dim, dropout=0.0)

    # Simulate two sequences with attributes
    batch_size = 1
    seq_len = 8

    # Query and key attributes (imagine these are entity type attributes)
    query_attr = torch.randn(batch_size, seq_len, attr_dim)
    key_attr = torch.randn(batch_size, seq_len, attr_dim)
    value = torch.randn(batch_size, seq_len, attr_dim * 4)

    # Make some positions similar (simulating entity matching)
    # Positions 2 and 5 have similar attributes
    key_attr[:, 5] = query_attr[:, 2] + torch.randn_like(query_attr[:, 2]) * 0.1

    # Compute attention
    output, attn_weights = join_attn(query_attr, key_attr, value)

    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"\nAttention from position 2 (query) to all keys:")
    print(f"  Position 5 (similar): {attn_weights[0, 2, 5].item():.4f}")
    print(f"  Other positions: {attn_weights[0, 2, [0,1,3,4,6,7]].mean().item():.4f} (avg)")
    print()

    # Multi-relation attention shows different heads learning different patterns
    hidden_dim = 256
    num_heads = 4
    num_attributes = 8

    multi_attn = MultiRelationAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_attributes=num_attributes
    )

    x = torch.randn(batch_size, seq_len, hidden_dim)
    output, attn = multi_attn(x, x, x, return_attention=True)

    if attn is not None:
        print(f"Multi-head attention shape: {attn.shape}")
        print(f"Each head attends on different attribute pairs:")
        for h in range(num_heads):
            q_attr = h % num_attributes
            k_attr = (h + 1) % num_attributes
            print(f"  Head {h}: joins attributes {q_attr} and {k_attr}")
    print()


def example_5_model_configurations():
    """Example 5: Different model sizes (125M, 350M, 1.3B configurations)."""
    print("=" * 60)
    print("Example 5: Model Size Configurations")
    print("=" * 60)

    configs = {
        "125M": RelationalTransformerConfig(
            vocab_size=32000,
            hidden_dim=768,
            num_encoder_layers=12,
            num_decoder_layers=12,
            num_heads=12,
            num_attributes=8
        ),
        "350M": RelationalTransformerConfig(
            vocab_size=32000,
            hidden_dim=1024,
            num_encoder_layers=24,
            num_decoder_layers=24,
            num_heads=16,
            num_attributes=8
        ),
        "1.3B": RelationalTransformerConfig(
            vocab_size=32000,
            hidden_dim=2048,
            num_encoder_layers=24,
            num_decoder_layers=24,
            num_heads=16,
            num_attributes=8
        ),
    }

    for name, config in configs.items():
        # Just count parameters without instantiating large models
        # Approximate parameter count
        d = config.hidden_dim
        L_enc = config.num_encoder_layers
        L_dec = config.num_decoder_layers
        V = config.vocab_size
        ffn = config.ffn_dim or 4 * d

        # Rough estimate: embeddings + encoder + decoder + output
        embedding_params = V * d
        layer_params = (4 * d * d + 2 * d * ffn) * (L_enc + L_dec)  # attention + ffn
        output_params = d * V

        total = embedding_params + layer_params + output_params
        print(f"{name}: ~{total / 1e6:.1f}M parameters (hidden={d}, layers={L_enc}+{L_dec})")

    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Relational Attention - Usage Examples")
    print("=" * 60 + "\n")

    example_1_basic_forward()
    example_2_text_to_sql()
    example_3_sequence_classification()
    example_4_attention_patterns()
    example_5_model_configurations()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

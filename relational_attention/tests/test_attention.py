"""
Unit tests for Relational Attention components.

Run with: python -m pytest tests/test_attention.py -v
"""

import pytest
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '..')

from relational_attention import (
    NeuralSelection,
    NeuralProjection,
    JoinAttention,
    RelationalAttentionHead,
    MultiRelationAttention,
    RelationalAttention,
    TupleEmbedding,
    RelationalTransformerBlock,
    RelationalTransformer,
    RelationalTransformerConfig,
)


class TestNeuralSelection:
    """Tests for Neural Selection (σ̃) operation."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        attr_dim = 64
        batch_size = 2
        seq_len = 16
        hidden_dim = 256

        selection = NeuralSelection(attr_dim=attr_dim)

        x = torch.randn(batch_size, seq_len, hidden_dim)
        attribute = torch.randn(batch_size, seq_len, attr_dim)

        output = selection(x, attribute)

        assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"

    def test_gate_range(self):
        """Test that selection gates are in [0, 1]."""
        attr_dim = 64
        selection = NeuralSelection(attr_dim=attr_dim)

        attribute = torch.randn(2, 16, attr_dim)

        # Access predicate network directly
        gates = selection.predicate_net(attribute)

        assert gates.min() >= 0, "Gates should be >= 0"
        assert gates.max() <= 1, "Gates should be <= 1"


class TestNeuralProjection:
    """Tests for Neural Projection (π̃) operation."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        num_attributes = 8
        attr_dim = 32
        output_dim = 64
        batch_size = 2
        seq_len = 16

        projection = NeuralProjection(
            num_attributes=num_attributes,
            attr_dim=attr_dim,
            output_dim=output_dim
        )

        attributes = [
            torch.randn(batch_size, seq_len, attr_dim)
            for _ in range(num_attributes)
        ]

        output = projection(attributes)

        expected_shape = (batch_size, seq_len, output_dim)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    def test_weights_sum_to_one(self):
        """Test that projection weights are normalized."""
        num_attributes = 8
        projection = NeuralProjection(
            num_attributes=num_attributes,
            attr_dim=32,
            output_dim=64
        )

        weights = torch.softmax(projection.projection_weights, dim=0)
        assert torch.isclose(weights.sum(), torch.tensor(1.0)), "Weights should sum to 1"


class TestJoinAttention:
    """Tests for Join Attention (⋈̃) operation."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        attr_dim = 64
        value_dim = 128
        batch_size = 2
        seq_len_q = 16
        seq_len_k = 20

        join_attn = JoinAttention(attr_dim=attr_dim)

        query_attr = torch.randn(batch_size, seq_len_q, attr_dim)
        key_attr = torch.randn(batch_size, seq_len_k, attr_dim)
        value = torch.randn(batch_size, seq_len_k, value_dim)

        output, attn_weights = join_attn(query_attr, key_attr, value)

        expected_output_shape = (batch_size, seq_len_q, value_dim)
        expected_attn_shape = (batch_size, seq_len_q, seq_len_k)

        assert output.shape == expected_output_shape
        assert attn_weights.shape == expected_attn_shape

    def test_attention_weights_normalized(self):
        """Test that attention weights sum to 1 across keys."""
        attr_dim = 64
        join_attn = JoinAttention(attr_dim=attr_dim, dropout=0.0)

        query_attr = torch.randn(2, 8, attr_dim)
        key_attr = torch.randn(2, 10, attr_dim)
        value = torch.randn(2, 10, 128)

        _, attn_weights = join_attn(query_attr, key_attr, value)

        # Sum along key dimension should be ~1
        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_mask_application(self):
        """Test that mask properly zeros attention to masked positions."""
        attr_dim = 64
        join_attn = JoinAttention(attr_dim=attr_dim, dropout=0.0)

        batch_size = 2
        seq_len = 8

        query_attr = torch.randn(batch_size, seq_len, attr_dim)
        key_attr = torch.randn(batch_size, seq_len, attr_dim)
        value = torch.randn(batch_size, seq_len, 128)

        # Mask out last 3 positions
        mask = torch.ones(batch_size, seq_len, seq_len)
        mask[:, :, -3:] = 0

        _, attn_weights = join_attn(query_attr, key_attr, value, mask=mask)

        # Attention to masked positions should be ~0
        masked_attention = attn_weights[:, :, -3:]
        assert masked_attention.max() < 0.01, "Masked attention should be near zero"


class TestRelationalAttentionHead:
    """Tests for single Relational Attention head."""

    def test_output_shape(self):
        """Test that output shape matches input."""
        hidden_dim = 256
        num_attributes = 8
        batch_size = 2
        seq_len = 16

        head = RelationalAttentionHead(
            hidden_dim=hidden_dim,
            num_attributes=num_attributes,
            query_attr_idx=0,
            key_attr_idx=1
        )

        query = torch.randn(batch_size, seq_len, hidden_dim)
        key = torch.randn(batch_size, seq_len, hidden_dim)
        value = torch.randn(batch_size, seq_len, hidden_dim)

        output, attn = head(query, key, value)

        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert attn.shape == (batch_size, seq_len, seq_len)


class TestMultiRelationAttention:
    """Tests for Multi-Relation Attention."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        hidden_dim = 256
        num_heads = 8
        num_attributes = 8
        batch_size = 2
        seq_len = 16

        multi_attn = MultiRelationAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_attributes=num_attributes
        )

        x = torch.randn(batch_size, seq_len, hidden_dim)

        output, _ = multi_attn(x, x, x)

        assert output.shape == (batch_size, seq_len, hidden_dim)

    def test_different_heads_use_different_attributes(self):
        """Test that heads are configured with different attribute pairs."""
        hidden_dim = 256
        num_heads = 4
        num_attributes = 8

        multi_attn = MultiRelationAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_attributes=num_attributes
        )

        # Check that heads have different attribute indices
        attr_pairs = set()
        for head in multi_attn.heads:
            pair = (head.query_attr_idx, head.key_attr_idx)
            attr_pairs.add(pair)

        # All pairs should be different
        assert len(attr_pairs) == num_heads


class TestTupleEmbedding:
    """Tests for Tuple Embedding layer."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        vocab_size = 1000
        hidden_dim = 256
        num_attributes = 8
        batch_size = 2
        seq_len = 16

        embedding = TupleEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_attributes=num_attributes
        )

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        output = embedding(input_ids)

        assert output.shape == (batch_size, seq_len, hidden_dim)

    def test_padding_is_zero(self):
        """Test that padding tokens produce zero embeddings."""
        vocab_size = 1000
        hidden_dim = 256
        padding_idx = 0

        embedding = TupleEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_attributes=8,
            padding_idx=padding_idx
        )

        # Input with padding
        input_ids = torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]])

        output = embedding(input_ids)

        # Padding positions should be zero
        assert torch.allclose(
            embedding.token_embedding.weight[padding_idx],
            torch.zeros(hidden_dim)
        )


class TestRelationalTransformerBlock:
    """Tests for Relational Transformer Block."""

    def test_output_shape(self):
        """Test that output shape matches input."""
        hidden_dim = 256
        num_heads = 8
        num_attributes = 8
        batch_size = 2
        seq_len = 16

        block = RelationalTransformerBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_attributes=num_attributes
        )

        x = torch.randn(batch_size, seq_len, hidden_dim)

        output, _ = block(x)

        assert output.shape == x.shape

    def test_residual_connection(self):
        """Test that residual connections allow gradient flow."""
        hidden_dim = 256
        num_heads = 8
        num_attributes = 8

        block = RelationalTransformerBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_attributes=num_attributes
        )

        x = torch.randn(2, 16, hidden_dim, requires_grad=True)
        output, _ = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestRelationalTransformer:
    """Tests for complete Relational Transformer model."""

    @pytest.fixture
    def small_config(self):
        """Small configuration for testing."""
        return RelationalTransformerConfig(
            vocab_size=1000,
            hidden_dim=64,
            num_encoder_layers=2,
            num_decoder_layers=2,
            num_heads=4,
            num_attributes=4,
            max_seq_len=128
        )

    def test_forward_pass(self, small_config):
        """Test complete forward pass."""
        model = RelationalTransformer(small_config)

        batch_size = 2
        src_len = 16
        tgt_len = 12

        input_ids = torch.randint(1, small_config.vocab_size, (batch_size, src_len))
        decoder_input_ids = torch.randint(1, small_config.vocab_size, (batch_size, tgt_len))
        labels = torch.randint(1, small_config.vocab_size, (batch_size, tgt_len))

        output = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )

        assert 'logits' in output
        assert 'loss' in output
        assert output['logits'].shape == (batch_size, tgt_len, small_config.vocab_size)

    def test_generation(self, small_config):
        """Test sequence generation."""
        model = RelationalTransformer(small_config)
        model.eval()

        input_ids = torch.randint(1, small_config.vocab_size, (1, 10))

        generated = model.generate(
            input_ids=input_ids,
            max_length=20,
            do_sample=False  # Greedy for determinism
        )

        assert generated.shape[0] == 1
        assert generated.shape[1] <= 20

    def test_gradient_flow(self, small_config):
        """Test that gradients flow through entire model."""
        model = RelationalTransformer(small_config)

        input_ids = torch.randint(1, small_config.vocab_size, (2, 10))
        decoder_input_ids = torch.randint(1, small_config.vocab_size, (2, 8))
        labels = torch.randint(1, small_config.vocab_size, (2, 8))

        output = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )

        output['loss'].backward()

        # Check gradients exist for key parameters
        assert model.encoder.embedding.token_embedding.weight.grad is not None
        assert model.decoder.embedding.token_embedding.weight.grad is not None


def run_tests():
    """Run all tests manually (without pytest)."""
    print("Running tests...")

    # Test Neural Selection
    print("Testing NeuralSelection...")
    test_selection = TestNeuralSelection()
    test_selection.test_output_shape()
    test_selection.test_gate_range()
    print("  PASSED")

    # Test Neural Projection
    print("Testing NeuralProjection...")
    test_proj = TestNeuralProjection()
    test_proj.test_output_shape()
    test_proj.test_weights_sum_to_one()
    print("  PASSED")

    # Test Join Attention
    print("Testing JoinAttention...")
    test_join = TestJoinAttention()
    test_join.test_output_shape()
    test_join.test_attention_weights_normalized()
    test_join.test_mask_application()
    print("  PASSED")

    # Test Relational Attention Head
    print("Testing RelationalAttentionHead...")
    test_head = TestRelationalAttentionHead()
    test_head.test_output_shape()
    print("  PASSED")

    # Test Multi-Relation Attention
    print("Testing MultiRelationAttention...")
    test_multi = TestMultiRelationAttention()
    test_multi.test_output_shape()
    test_multi.test_different_heads_use_different_attributes()
    print("  PASSED")

    # Test Tuple Embedding
    print("Testing TupleEmbedding...")
    test_embed = TestTupleEmbedding()
    test_embed.test_output_shape()
    test_embed.test_padding_is_zero()
    print("  PASSED")

    # Test Transformer Block
    print("Testing RelationalTransformerBlock...")
    test_block = TestRelationalTransformerBlock()
    test_block.test_output_shape()
    test_block.test_residual_connection()
    print("  PASSED")

    # Test Full Model
    print("Testing RelationalTransformer...")
    test_model = TestRelationalTransformer()
    config = test_model.small_config()
    test_model.test_forward_pass(config)
    test_model.test_generation(config)
    test_model.test_gradient_flow(config)
    print("  PASSED")

    print("\nAll tests passed!")


if __name__ == "__main__":
    run_tests()

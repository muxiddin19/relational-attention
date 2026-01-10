"""
Relational Attention: A Set-Theoretic Foundation for Neural Structured Reasoning

This module implements the core Relational Attention mechanism as described in:
"Relational Attention Is All You Need: A Set-Theoretic Foundation for Neural Structured Reasoning"

The key innovation is treating token representations as tuples with typed attributes,
enabling differentiable analogs of relational algebra operations:
- Neural Selection (σ̃): Attribute-conditional gating
- Neural Projection (π̃): Learned attribute attention
- Neural Join (⋈̃): Attribute-wise similarity for combining tuples
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class NeuralSelection(nn.Module):
    """
    Neural Selection (σ̃): Filters tuples based on learned predicates.

    Implements attribute-conditional gating where a predicate network
    learns to identify relevant tuples based on specific attribute values.

    σ̃_{f_φ}^{(j)}(X) = [g_1 · x_1, ..., g_n · x_n]
    where g_i = f_φ(a_i^{(j)}) is the selection gate for position i.

    Args:
        attr_dim: Dimension of each attribute (d/k)
        hidden_dim: Hidden dimension for the predicate network
        dropout: Dropout probability
    """

    def __init__(
        self,
        attr_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        hidden_dim = hidden_dim or attr_dim * 2

        # Predicate network f_φ: R^{d/k} -> [0, 1]
        self.predicate_net = nn.Sequential(
            nn.Linear(attr_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        attribute: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply neural selection based on a specific attribute.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            attribute: Attribute tensor of shape (batch, seq_len, attr_dim)

        Returns:
            Gated output of shape (batch, seq_len, hidden_dim)
        """
        # Compute selection gates
        gates = self.predicate_net(attribute)  # (batch, seq_len, 1)

        # Apply soft gating
        return x * gates


class NeuralProjection(nn.Module):
    """
    Neural Projection (π̃): Extracts and combines specific attributes.

    Implements learned attribute attention that dynamically focuses
    on relevant attributes and combines them through learned transformations.

    π̃_w(x) = Σ_i softmax(w)_i · W_i · a_i

    Args:
        num_attributes: Number of attributes (k)
        attr_dim: Dimension of each attribute (d/k)
        output_dim: Output dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_attributes: int,
        attr_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_attributes = num_attributes
        self.attr_dim = attr_dim

        # Learnable projection weights w ∈ R^k
        self.projection_weights = nn.Parameter(torch.zeros(num_attributes))

        # Projection matrices W_i ∈ R^{d' × d/k} for each attribute
        self.projection_matrices = nn.ModuleList([
            nn.Linear(attr_dim, output_dim, bias=False)
            for _ in range(num_attributes)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, attributes: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply neural projection to combine attributes.

        Args:
            attributes: List of k attribute tensors, each of shape (batch, seq_len, attr_dim)

        Returns:
            Projected output of shape (batch, seq_len, output_dim)
        """
        # Compute attention weights over attributes
        weights = F.softmax(self.projection_weights, dim=0)

        # Project and combine attributes
        output = None
        for i, (attr, proj) in enumerate(zip(attributes, self.projection_matrices)):
            projected = proj(attr)  # (batch, seq_len, output_dim)
            if output is None:
                output = weights[i] * projected
            else:
                output = output + weights[i] * projected

        return self.dropout(output)


class JoinAttention(nn.Module):
    """
    Neural Join (⋈̃): Combines tuples based on attribute matching.

    The key innovation of Relational Attention - computes similarities
    over specific attributes rather than entire representations.

    ⋈̃^{(j,l)}(X, Y)_i = Σ_t α_it · [x_i; y_t]
    where α_it = softmax(a_i^{(j)} · b_t^{(l)} / τ)

    Args:
        attr_dim: Dimension of each attribute (d/k)
        temperature: Temperature for softmax (τ), learnable if None
        dropout: Dropout probability
    """

    def __init__(
        self,
        attr_dim: int,
        temperature: Optional[float] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attr_dim = attr_dim
        self.scale = 1.0 / math.sqrt(attr_dim)

        # Learnable temperature if not specified
        if temperature is None:
            self.temperature = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_attr: torch.Tensor,
        key_attr: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute join attention based on attribute matching.

        Args:
            query_attr: Query attribute of shape (batch, seq_len_q, attr_dim)
            key_attr: Key attribute of shape (batch, seq_len_k, attr_dim)
            value: Value tensor of shape (batch, seq_len_k, value_dim)
            mask: Optional attention mask of shape (batch, seq_len_q, seq_len_k)

        Returns:
            Tuple of:
                - Output tensor of shape (batch, seq_len_q, value_dim)
                - Attention weights of shape (batch, seq_len_q, seq_len_k)
        """
        # Compute attribute-wise similarity
        # α_it = (a_i^{(j)} · b_t^{(l)}) / τ
        scores = torch.matmul(query_attr, key_attr.transpose(-2, -1))
        scores = scores * self.scale / self.temperature

        # Apply mask if provided
        # Mask can be: (S_q, S_k) causal, (B, S_k) padding, (B, S_q, S_k), or (B, 1, 1, S_k)
        if mask is not None:
            batch_size = scores.size(0)
            # Ensure mask is broadcastable to scores shape (B, S_q, S_k)
            if mask.dim() == 2:
                # Check if it's a causal mask (S_q, S_k) or padding mask (B, S_k)
                if mask.size(0) == scores.size(1) and mask.size(1) == scores.size(2):
                    # Causal mask (S_q, S_k) -> (1, S_q, S_k)
                    mask = mask.unsqueeze(0)
                else:
                    # Padding mask (B, S_k) -> (B, 1, S_k)
                    mask = mask.unsqueeze(1)
            elif mask.dim() == 4:
                # (B, 1, 1, S_k) -> (B, 1, S_k)
                mask = mask.squeeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute weighted sum of values
        output = torch.matmul(attn_weights, value)

        return output, attn_weights


class RelationalAttentionHead(nn.Module):
    """
    Single head of Relational Attention.

    Combines Join Attention, Neural Selection, and Neural Projection
    into a single attention head that operates on specific attribute pairs.

    RelAttn(Q, K, V) = π̃_{w_o}(σ̃_{f_φ}(⋈̃^{(j,l)}(Q, K)) ⊙ V)

    Args:
        hidden_dim: Model hidden dimension (d)
        num_attributes: Number of attributes (k)
        query_attr_idx: Index of query attribute for joining (j)
        key_attr_idx: Index of key attribute for joining (l)
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_dim: int,
        num_attributes: int,
        query_attr_idx: int,
        key_attr_idx: int,
        dropout: float = 0.1
    ):
        super().__init__()

        assert hidden_dim % num_attributes == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_attributes ({num_attributes})"

        self.hidden_dim = hidden_dim
        self.num_attributes = num_attributes
        self.attr_dim = hidden_dim // num_attributes
        self.query_attr_idx = query_attr_idx
        self.key_attr_idx = key_attr_idx

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Join Attention
        self.join_attention = JoinAttention(
            attr_dim=self.attr_dim,
            dropout=dropout
        )

        # Neural Selection (operates on joined representation)
        self.selection = NeuralSelection(
            attr_dim=self.attr_dim,
            dropout=dropout
        )

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def _split_attributes(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Split tensor into k attribute embeddings."""
        batch, seq_len, _ = x.shape
        # Reshape to (batch, seq_len, num_attributes, attr_dim)
        x = x.view(batch, seq_len, self.num_attributes, self.attr_dim)
        # Return list of attributes
        return [x[:, :, i, :] for i in range(self.num_attributes)]

    def _get_attribute(self, x: torch.Tensor, attr_idx: int) -> torch.Tensor:
        """Extract specific attribute from tensor."""
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_attributes, self.attr_dim)
        return x[:, :, attr_idx, :]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Relational Attention head.

        Args:
            query: Query tensor of shape (batch, seq_len_q, hidden_dim)
            key: Key tensor of shape (batch, seq_len_k, hidden_dim)
            value: Value tensor of shape (batch, seq_len_k, hidden_dim)
            mask: Optional attention mask

        Returns:
            Tuple of output tensor and attention weights
        """
        # Project Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Extract attributes for joining
        q_attr = self._get_attribute(Q, self.query_attr_idx)
        k_attr = self._get_attribute(K, self.key_attr_idx)

        # Apply Join Attention
        joined, attn_weights = self.join_attention(q_attr, k_attr, V, mask)

        # Apply Neural Selection using the query attribute
        selection_attr = self._get_attribute(Q, self.query_attr_idx)
        selected = self.selection(joined, selection_attr)

        # Output projection
        output = self.out_proj(selected)

        return output, attn_weights


class MultiRelationAttention(nn.Module):
    """
    Multi-Relation Attention: Multiple heads with different attribute pairs.

    Analogous to multi-head attention, but each head uses different
    attribute pairs (j_r, l_r) for joining, enabling the model to
    learn different types of relational patterns.

    MultiRelAttn(Q, K, V) = Concat(head_1, ..., head_h) W^O

    Args:
        hidden_dim: Model hidden dimension (d)
        num_heads: Number of attention heads (h)
        num_attributes: Number of attributes (k)
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_attributes: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_attributes = num_attributes
        self.head_dim = hidden_dim // num_heads

        # Create heads with different attribute pairs
        # Each head learns to join on different attribute combinations
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            # Assign different attribute pairs to different heads
            query_attr = i % num_attributes
            key_attr = (i + 1) % num_attributes

            self.heads.append(
                RelationalAttentionHead(
                    hidden_dim=hidden_dim,
                    num_attributes=num_attributes,
                    query_attr_idx=query_attr,
                    key_attr_idx=key_attr,
                    dropout=dropout
                )
            )

        # Output projection
        self.out_proj = nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of Multi-Relation Attention.

        Args:
            query: Query tensor of shape (batch, seq_len_q, hidden_dim)
            key: Key tensor of shape (batch, seq_len_k, hidden_dim)
            value: Value tensor of shape (batch, seq_len_k, hidden_dim)
            mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            Tuple of output tensor and optional attention weights
        """
        head_outputs = []
        attention_weights = []

        for head in self.heads:
            out, attn = head(query, key, value, mask)
            head_outputs.append(out)
            if return_attention:
                attention_weights.append(attn)

        # Concatenate heads and project
        concat = torch.cat(head_outputs, dim=-1)
        output = self.out_proj(concat)
        output = self.dropout(output)

        if return_attention:
            # Stack attention weights: (num_heads, batch, seq_q, seq_k)
            attn = torch.stack(attention_weights, dim=0)
            return output, attn

        return output, None


class RelationalAttention(nn.Module):
    """
    Complete Relational Attention module.

    This is the main interface for Relational Attention, providing
    a drop-in replacement for standard multi-head attention with
    relational algebra-inspired operations.

    Args:
        hidden_dim: Model hidden dimension
        num_heads: Number of attention heads
        num_attributes: Number of tuple attributes (default: 8)
        dropout: Dropout probability
        bias: Whether to use bias in projections
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_attributes: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_attributes = num_attributes

        self.attention = MultiRelationAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_attributes=num_attributes,
            dropout=dropout
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of Relational Attention.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            context: Optional context for cross-attention
            mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            Tuple of output tensor and optional attention weights
        """
        # Self-attention or cross-attention
        if context is None:
            query = key = value = x
        else:
            query = x
            key = value = context

        output, attn_weights = self.attention(
            query, key, value, mask, return_attention
        )

        return output, attn_weights

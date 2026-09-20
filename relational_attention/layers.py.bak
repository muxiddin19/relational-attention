"""
Relational Transformer Layers

This module implements the Relational Transformer building blocks:
- Tuple Embedding Layer
- Schema-Aware Positional Encoding
- Relational Transformer Block
- Feed-Forward Network
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .attention import RelationalAttention, MultiRelationAttention


class TupleEmbedding(nn.Module):
    """
    Tuple Embedding Layer: Decomposes input embeddings into k typed attributes.

    Instead of a single embedding vector, each token is represented as a tuple
    of k attribute embeddings, each capturing different semantic dimensions.

    e_i = [W_1^{(e)} x_i; W_2^{(e)} x_i; ...; W_k^{(e)} x_i]

    Args:
        vocab_size: Size of vocabulary
        hidden_dim: Model hidden dimension (d)
        num_attributes: Number of attributes (k)
        padding_idx: Index of padding token
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_attributes: int = 8,
        padding_idx: int = 0,
        dropout: float = 0.1
    ):
        super().__init__()

        assert hidden_dim % num_attributes == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_attributes ({num_attributes})"

        self.hidden_dim = hidden_dim
        self.num_attributes = num_attributes
        self.attr_dim = hidden_dim // num_attributes

        # Base token embedding
        self.token_embedding = nn.Embedding(
            vocab_size, hidden_dim, padding_idx=padding_idx
        )

        # Attribute-specific projection matrices W_j^{(e)}
        self.attribute_projections = nn.ModuleList([
            nn.Linear(hidden_dim, self.attr_dim, bias=False)
            for _ in range(num_attributes)
        ])

        # Layer norm for each attribute
        self.attr_layer_norms = nn.ModuleList([
            nn.LayerNorm(self.attr_dim)
            for _ in range(num_attributes)
        ])

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed tokens as tuples.

        Args:
            x: Token indices of shape (batch, seq_len)

        Returns:
            Tuple embeddings of shape (batch, seq_len, hidden_dim)
            where hidden_dim is organized as (num_attributes, attr_dim)
        """
        # Get base embeddings
        embeddings = self.token_embedding(x) * self.scale

        # Project to each attribute
        attributes = []
        for proj, norm in zip(self.attribute_projections, self.attr_layer_norms):
            attr = proj(embeddings)
            attr = norm(attr)
            attributes.append(attr)

        # Concatenate attributes: (batch, seq_len, num_attributes * attr_dim)
        output = torch.cat(attributes, dim=-1)
        return self.dropout(output)

    def get_attributes(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get individual attribute embeddings.

        Args:
            x: Embedded tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            List of k attribute tensors, each of shape (batch, seq_len, attr_dim)
        """
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_attributes, self.attr_dim)
        return [x[:, :, i, :] for i in range(self.num_attributes)]


class SchemaAwarePositionalEncoding(nn.Module):
    """
    Schema-Aware Positional Encoding for structured inputs.

    Augments standard positional encodings with relational position embeddings
    that capture structural relationships like table membership and column types.

    p_i = p_i^{(seq)} + p_{r(i)}^{(rel)} + p_{c(i)}^{(col)}

    Args:
        hidden_dim: Model hidden dimension
        max_seq_len: Maximum sequence length
        max_relations: Maximum number of relations/tables
        max_columns: Maximum number of column types
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_dim: int,
        max_seq_len: int = 2048,
        max_relations: int = 64,
        max_columns: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Standard sinusoidal positional encoding
        pe = torch.zeros(max_seq_len, hidden_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        # Learnable relation (table) embeddings
        self.relation_embeddings = nn.Embedding(max_relations, hidden_dim)

        # Learnable column type embeddings
        self.column_embeddings = nn.Embedding(max_columns, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        relation_ids: Optional[torch.Tensor] = None,
        column_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Add positional encodings to input.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            relation_ids: Optional relation IDs of shape (batch, seq_len)
            column_ids: Optional column IDs of shape (batch, seq_len)

        Returns:
            Position-encoded tensor of shape (batch, seq_len, hidden_dim)
        """
        seq_len = x.size(1)

        # Add sequential positional encoding
        output = x + self.pe[:, :seq_len]

        # Add relation embeddings if provided
        if relation_ids is not None:
            output = output + self.relation_embeddings(relation_ids)

        # Add column embeddings if provided
        if column_ids is not None:
            output = output + self.column_embeddings(column_ids)

        return self.dropout(output)


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Standard FFN that operates on concatenated attribute representations.

    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

    Args:
        hidden_dim: Model hidden dimension
        ffn_dim: Feed-forward intermediate dimension (typically 4 * hidden_dim)
        dropout: Dropout probability
        activation: Activation function ('relu' or 'gelu')
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()

        ffn_dim = ffn_dim or hidden_dim * 4

        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of FFN.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class RelationalTransformerBlock(nn.Module):
    """
    Relational Transformer Block.

    Combines Multi-Relation Attention with Feed-Forward Network,
    using pre-norm residual connections.

    h' = LayerNorm(h + MultiRelAttn(h, h, h))
    h'' = LayerNorm(h' + FFN(h'))

    Args:
        hidden_dim: Model hidden dimension
        num_heads: Number of attention heads
        num_attributes: Number of tuple attributes
        ffn_dim: Feed-forward intermediate dimension
        dropout: Dropout probability
        activation: Activation function for FFN
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_attributes: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()

        # Multi-Relation Self-Attention
        self.self_attention = MultiRelationAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_attributes=num_attributes,
            dropout=dropout
        )

        # Feed-Forward Network
        self.ffn = FeedForward(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
            activation=activation
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of Relational Transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            Tuple of output tensor and optional attention weights
        """
        # Self-attention with residual connection
        normed = self.norm1(x)
        attn_out, attn_weights = self.self_attention(
            normed, normed, normed, mask, return_attention
        )
        x = x + self.dropout(attn_out)

        # FFN with residual connection
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out

        return x, attn_weights


class RelationalTransformerEncoderBlock(nn.Module):
    """
    Relational Transformer Encoder Block with cross-attention.

    Used in encoder-decoder architectures for attending to encoder outputs.

    Args:
        hidden_dim: Model hidden dimension
        num_heads: Number of attention heads
        num_attributes: Number of tuple attributes
        ffn_dim: Feed-forward intermediate dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_attributes: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()

        # Self-attention
        self.self_attention = MultiRelationAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_attributes=num_attributes,
            dropout=dropout
        )

        # Cross-attention
        self.cross_attention = MultiRelationAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_attributes=num_attributes,
            dropout=dropout
        )

        # Feed-Forward Network
        self.ffn = FeedForward(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            dropout=dropout
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of encoder block with cross-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            encoder_output: Encoder output of shape (batch, src_len, hidden_dim)
            self_mask: Mask for self-attention
            cross_mask: Mask for cross-attention

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim)
        """
        # Self-attention
        normed = self.norm1(x)
        attn_out, _ = self.self_attention(normed, normed, normed, self_mask)
        x = x + self.dropout(attn_out)

        # Cross-attention
        normed = self.norm2(x)
        cross_out, _ = self.cross_attention(normed, encoder_output, encoder_output, cross_mask)
        x = x + self.dropout(cross_out)

        # FFN
        normed = self.norm3(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out

        return x

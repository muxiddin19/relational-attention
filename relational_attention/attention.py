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

import random
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
        mask: Optional[torch.Tensor] = None,
        key_attr2: Optional[torch.Tensor] = None,
        chain_gate: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute join attention based on attribute matching.

        Args:
            query_attr: Query attribute of shape (batch, seq_len_q, attr_dim)
            key_attr: Key attribute of shape (batch, seq_len_k, attr_dim)
            value: Value tensor of shape (batch, seq_len_k, value_dim)
            mask: Optional attention mask of shape (batch, seq_len_q, seq_len_k)
            key_attr2: Optional secondary (composed/next-hop) key attribute,
                same shape as key_attr. When given (with chain_gate), the
                join score blends the primary and composed-hop similarities
                pre-softmax: Composed Join Attention (see module docstring).
            chain_gate: Optional scalar (pre-sigmoid) blend weight; sigmoid
                near 1 keeps behavior close to the unmodified primary-only
                score, near 0 favors the composed (next-hop) term.

        Returns:
            Tuple of:
                - Output tensor of shape (batch, seq_len_q, value_dim)
                - Attention weights of shape (batch, seq_len_q, seq_len_k)
        """
        # Compute attribute-wise similarity
        # α_it = (a_i^{(j)} · b_t^{(l)}) / τ
        scores = torch.matmul(query_attr, key_attr.transpose(-2, -1))
        if key_attr2 is not None and chain_gate is not None:
            scores2 = torch.matmul(query_attr, key_attr2.transpose(-2, -1))
            beta = torch.sigmoid(chain_gate)
            scores = beta * scores + (1.0 - beta) * scores2
        # Clamp temperature away from zero to prevent fp16 overflow → NaN
        temp = self.temperature.clamp(min=1e-2)
        scores = scores * self.scale / temp
        # Clamp scores before softmax to prevent fp16 overflow
        scores = scores.clamp(min=-1e4, max=1e4)

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

        # Compute attention weights — use float32 for softmax stability
        attn_weights = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        # Guard against NaN (e.g. all-masked rows produce -inf → NaN softmax)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
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
        dropout: float = 0.1,
        use_gating: bool = True,
        pairing_strategy: str = "cyclic",
        use_composed_join: bool = False
    ):
        super().__init__()

        assert hidden_dim % num_attributes == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_attributes ({num_attributes})"
        assert pairing_strategy in ("cyclic", "random_fixed", "learned"), \
            f"unknown pairing_strategy: {pairing_strategy}"

        self.hidden_dim = hidden_dim
        self.num_attributes = num_attributes
        self.attr_dim = hidden_dim // num_attributes
        self.query_attr_idx = query_attr_idx
        self.key_attr_idx = key_attr_idx
        self.use_gating = use_gating
        self.pairing_strategy = pairing_strategy
        self.use_composed_join = use_composed_join

        # Composed Join Attention (scoped, single-layer 2-hop relational
        # composition; see module docstring). chain_gate initialized at 2.0
        # -> sigmoid(2.0) ~= 0.88, so training starts close to the
        # unmodified primary-only score and can learn to lean on the
        # composed (next-hop) term.
        if use_composed_join:
            self.chain_gate = nn.Parameter(torch.tensor(2.0))

        # Ablation: "learned" pairing replaces the fixed key_attr_idx with a
        # learned softmax mixture over all k attribute slots, initialized
        # uniformly. query_attr_idx stays fixed/cyclic in every strategy so
        # the comparison isolates the key-side assignment only.
        if pairing_strategy == "learned":
            self.key_attr_logits = nn.Parameter(torch.zeros(num_attributes))

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
        if use_gating:
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

    def _get_key_attribute(self, x: torch.Tensor) -> torch.Tensor:
        """
        Key-side attribute extraction, dispatching on pairing_strategy.
        cyclic / random_fixed: hard index self.key_attr_idx (identical code
        path; the two strategies differ only in how key_attr_idx was chosen
        at __init__ time by MultiRelationAttention).
        learned: soft mixture over all k slots via a learned softmax weight,
        initialized uniform (torch.zeros -> uniform softmax at step 0).
        """
        if self.pairing_strategy == "learned":
            batch, seq_len, _ = x.shape
            x = x.view(batch, seq_len, self.num_attributes, self.attr_dim)
            w = torch.softmax(self.key_attr_logits, dim=0)  # (k,)
            return torch.einsum("k,bslk->bsl", w, x.permute(0, 1, 3, 2)) \
                if False else (x * w.view(1, 1, -1, 1)).sum(dim=2)
        return self._get_attribute(x, self.key_attr_idx)

    def _get_composed_key_attribute(self, x: torch.Tensor) -> torch.Tensor:
        """
        Secondary key attribute for Composed Join Attention: the next slot
        after key_attr_idx in the cyclic chain, i.e. one additional FD hop
        (j -> l -> l+1) computed within this single layer rather than
        requiring a second RelAttn layer to reach it.
        """
        next_idx = (self.key_attr_idx + 1) % self.num_attributes
        return self._get_attribute(x, next_idx)

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
        k_attr = self._get_key_attribute(K)

        # Apply Join Attention (optionally Composed Join Attention: blend in
        # a second, next-hop key attribute pre-softmax)
        if self.use_composed_join:
            k_attr2 = self._get_composed_key_attribute(K)
            joined, attn_weights = self.join_attention(
                q_attr, k_attr, V, mask,
                key_attr2=k_attr2, chain_gate=self.chain_gate)
        else:
            joined, attn_weights = self.join_attention(q_attr, k_attr, V, mask)

        # Apply Neural Selection using the query attribute
        if self.use_gating:
            selection_attr = self._get_attribute(Q, self.query_attr_idx)
            selected = self.selection(joined, selection_attr)
        else:
            selected = joined

        # Output projection
        output = self.out_proj(selected)
        # Safety guard — propagating NaN crashes training
        output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)

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
        dropout: float = 0.1,
        use_gating: bool = True,
        use_mixing: bool = True,
        pairing_strategy: str = "cyclic",
        pairing_seed: int = 1234,
        use_composed_join: bool = False
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        assert pairing_strategy in ("cyclic", "random_fixed", "learned"), \
            f"unknown pairing_strategy: {pairing_strategy}"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_attributes = num_attributes
        self.head_dim = hidden_dim // num_heads
        self.use_mixing = use_mixing
        self.pairing_strategy = pairing_strategy
        self.use_composed_join = use_composed_join

        # Ablation (reviewer-requested, ICDE 2027 review D4/D7 & D2):
        # query_attr is i % k in every strategy, isolating the key-side
        # assignment as the sole varying factor across pairing strategies.
        #   cyclic:       key_attr = (i + 1) % k                 [original]
        #   random_fixed: key_attr ~ Uniform({0..k-1} \ {query_attr}),
        #                 drawn once from a strategy-dedicated RNG seeded by
        #                 pairing_seed (NOT the training seed), so the same
        #                 random pairing structure is shared across the 3
        #                 training seeds -- isolating pairing choice from
        #                 training-seed variance.
        #   learned:      key attribute is a learned softmax mixture over
        #                 all k slots (see RelationalAttentionHead); the
        #                 key_attr passed here is unused as an index and
        #                 only kept for logging/debugging.
        rng = random.Random(pairing_seed) if pairing_strategy == "random_fixed" else None

        # Create heads with different attribute pairs
        # Each head learns to join on different attribute combinations
        self.heads = nn.ModuleList()
        self.head_pairs = []  # for logging/reproducibility checks
        for i in range(num_heads):
            query_attr = i % num_attributes
            if pairing_strategy == "cyclic":
                key_attr = (i + 1) % num_attributes
            elif pairing_strategy == "random_fixed":
                choices = [a for a in range(num_attributes) if a != query_attr]
                key_attr = rng.choice(choices)
            else:  # "learned" -- placeholder index, unused for indexing
                key_attr = (i + 1) % num_attributes
            self.head_pairs.append((query_attr, key_attr))

            self.heads.append(
                RelationalAttentionHead(
                    hidden_dim=hidden_dim,
                    num_attributes=num_attributes,
                    query_attr_idx=query_attr,
                    key_attr_idx=key_attr,
                    dropout=dropout,
                    use_gating=use_gating,
                    pairing_strategy=pairing_strategy,
                    use_composed_join=use_composed_join
                )
            )

        # Output projection (W^O mixes attribute-head outputs; disabled in -mixing ablation)
        if use_mixing:
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

        # Concatenate heads and project (or average for -mixing ablation)
        if self.use_mixing:
            concat = torch.cat(head_outputs, dim=-1)
            output = self.out_proj(concat)
        else:
            output = torch.stack(head_outputs, dim=0).sum(0) / len(head_outputs)
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


class StandardMultiHeadAttention(nn.Module):
    """
    Genuine standard multi-head scaled dot-product attention (Vaswani et al.,
    2017), used as the true architectural baseline. A single (hidden_dim,
    hidden_dim) projection each for Q, K, V is split across num_heads heads
    of hidden_dim/num_heads dimensions (the standard convention), followed by
    a single (hidden_dim, hidden_dim) output projection. No attribute
    decomposition, no per-slot gating, no attribute mixing -- this is
    deliberately the plain baseline Eq. (1) describes, not a degenerate case
    of MultiRelationAttention (see the paper's corrected Remark on why k=1 in
    MultiRelationAttention does not reduce to this).

    Matches MultiRelationAttention's forward signature exactly so it is a
    drop-in replacement at the block level.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        **_ignored_kwargs,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, seq_q, _ = query.shape

        Q = self._split_heads(self.q_proj(query))
        K = self._split_heads(self.k_proj(key))
        V = self._split_heads(self.v_proj(value))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, seq_q, self.hidden_dim)
        output = self.dropout(self.out_proj(out))

        if return_attention:
            return output, attn
        return output, None

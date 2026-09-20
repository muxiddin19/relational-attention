"""
New module: Composed Join Attention. Adds an optional, config-flagged
single-layer 2-hop relational composition: in addition to the primary
join-attention pair (j, l), each head also computes a secondary score
against the "next" slot in the cyclic chain (l+1) mod k, blended into the
primary score via a learned, sigmoid-gated scalar (initialized near the
primary term, so behavior starts close to the unmodified baseline and can
learn to lean on the composed term). This lets a single layer partially
compose a 2-hop FD chain (j -> l -> l+1) instead of requiring a second
RelAttn layer to reach the next hop, motivated by Edge Transformers'
triangular attention (Bergen et al., NeurIPS 2021) but scoped to avoid the
O(n^3) edge-state cost of a full triangular update.

use_composed_join defaults to False everywhere; existing behavior and all
prior verified results are unaffected unless explicitly enabled.

Run on the remote host: python3 patch_composed_join.py
"""

ATTN = "relational_attention/attention.py"

with open(ATTN, encoding="utf-8") as f:
    src = f.read()

# ---------------------------------------------------------------------------
# 1. JoinAttention.forward: accept an optional second key_attr + gate, blend
#    scores pre-softmax when provided.
# ---------------------------------------------------------------------------
old_join_fwd_sig = '''    def forward(
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
        # Clamp temperature away from zero to prevent fp16 overflow → NaN
        temp = self.temperature.clamp(min=1e-2)
        scores = scores * self.scale / temp
        # Clamp scores before softmax to prevent fp16 overflow
        scores = scores.clamp(min=-1e4, max=1e4)'''

new_join_fwd_sig = '''    def forward(
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
        scores = scores.clamp(min=-1e4, max=1e4)'''

assert old_join_fwd_sig in src, "JoinAttention.forward signature/score block not found verbatim"
src = src.replace(old_join_fwd_sig, new_join_fwd_sig, 1)

# ---------------------------------------------------------------------------
# 2. RelationalAttentionHead.__init__: accept use_composed_join, register
#    the per-head chain gate parameter.
# ---------------------------------------------------------------------------
old_head_init_sig = '''    def __init__(
        self,
        hidden_dim: int,
        num_attributes: int,
        query_attr_idx: int,
        key_attr_idx: int,
        dropout: float = 0.1,
        use_gating: bool = True,
        pairing_strategy: str = "cyclic"
    ):
        super().__init__()

        assert hidden_dim % num_attributes == 0, \\
            f"hidden_dim ({hidden_dim}) must be divisible by num_attributes ({num_attributes})"
        assert pairing_strategy in ("cyclic", "random_fixed", "learned"), \\
            f"unknown pairing_strategy: {pairing_strategy}"

        self.hidden_dim = hidden_dim
        self.num_attributes = num_attributes
        self.attr_dim = hidden_dim // num_attributes
        self.query_attr_idx = query_attr_idx
        self.key_attr_idx = key_attr_idx
        self.use_gating = use_gating
        self.pairing_strategy = pairing_strategy'''

new_head_init_sig = '''    def __init__(
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

        assert hidden_dim % num_attributes == 0, \\
            f"hidden_dim ({hidden_dim}) must be divisible by num_attributes ({num_attributes})"
        assert pairing_strategy in ("cyclic", "random_fixed", "learned"), \\
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
            self.chain_gate = nn.Parameter(torch.tensor(2.0))'''

assert old_head_init_sig in src, "RelationalAttentionHead.__init__ signature not found verbatim"
src = src.replace(old_head_init_sig, new_head_init_sig, 1)

# ---------------------------------------------------------------------------
# 3. Add a _get_composed_key_attribute method (next slot in the chain).
# ---------------------------------------------------------------------------
old_get_key_attr = '''        return self._get_attribute(x, self.key_attr_idx)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:'''

new_get_key_attr = '''        return self._get_attribute(x, self.key_attr_idx)

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:'''

assert old_get_key_attr in src, "insertion point before forward() not found verbatim"
src = src.replace(old_get_key_attr, new_get_key_attr, 1)

# ---------------------------------------------------------------------------
# 4. RelationalAttentionHead.forward: extract composed key attr and pass
#    through to JoinAttention when enabled.
# ---------------------------------------------------------------------------
old_fwd_body = '''        # Extract attributes for joining
        q_attr = self._get_attribute(Q, self.query_attr_idx)
        k_attr = self._get_key_attribute(K)

        # Apply Join Attention
        joined, attn_weights = self.join_attention(q_attr, k_attr, V, mask)'''

new_fwd_body = '''        # Extract attributes for joining
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
            joined, attn_weights = self.join_attention(q_attr, k_attr, V, mask)'''

assert old_fwd_body in src, "RelationalAttentionHead.forward body not found verbatim"
src = src.replace(old_fwd_body, new_fwd_body, 1)

with open(ATTN, "w", encoding="utf-8") as f:
    f.write(src)
print("Patched", ATTN, "(JoinAttention + RelationalAttentionHead) successfully.")

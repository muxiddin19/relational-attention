"""
Patch relational_attention/attention.py to support pairing_strategy in
{"cyclic", "random_fixed", "learned"}, isolating exactly the variable
reviewers asked about: the key-side attribute assignment. Query-side stays
cyclic (query_attr = i % k) in all three strategies, so the comparison is
clean and well-controlled.

Run on the remote host: python3 patch_pairing.py
"""
import re

PATH = "relational_attention/attention.py"

with open(PATH, encoding="utf-8") as f:
    src = f.read()

# ---------------------------------------------------------------------------
# 1. RelationalAttentionHead.__init__: accept pairing_strategy, add learned
#    key-attribute logits when pairing_strategy == "learned".
# ---------------------------------------------------------------------------
old_init_sig = '''    def __init__(
        self,
        hidden_dim: int,
        num_attributes: int,
        query_attr_idx: int,
        key_attr_idx: int,
        dropout: float = 0.1,
        use_gating: bool = True
    ):
        super().__init__()

        assert hidden_dim % num_attributes == 0, \\
            f"hidden_dim ({hidden_dim}) must be divisible by num_attributes ({num_attributes})"

        self.hidden_dim = hidden_dim
        self.num_attributes = num_attributes
        self.attr_dim = hidden_dim // num_attributes
        self.query_attr_idx = query_attr_idx
        self.key_attr_idx = key_attr_idx
        self.use_gating = use_gating'''

new_init_sig = '''    def __init__(
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
        self.pairing_strategy = pairing_strategy

        # Ablation: "learned" pairing replaces the fixed key_attr_idx with a
        # learned softmax mixture over all k attribute slots, initialized
        # uniformly. query_attr_idx stays fixed/cyclic in every strategy so
        # the comparison isolates the key-side assignment only.
        if pairing_strategy == "learned":
            self.key_attr_logits = nn.Parameter(torch.zeros(num_attributes))'''

assert old_init_sig in src, "RelationalAttentionHead.__init__ signature not found verbatim"
src = src.replace(old_init_sig, new_init_sig, 1)

# ---------------------------------------------------------------------------
# 2. Add a _get_key_attribute method that dispatches on pairing_strategy.
# ---------------------------------------------------------------------------
old_get_attr = '''    def _get_attribute(self, x: torch.Tensor, attr_idx: int) -> torch.Tensor:
        """Extract specific attribute from tensor."""
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_attributes, self.attr_dim)
        return x[:, :, attr_idx, :]'''

new_get_attr = '''    def _get_attribute(self, x: torch.Tensor, attr_idx: int) -> torch.Tensor:
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
            return torch.einsum("k,bslk->bsl", w, x.permute(0, 1, 3, 2)) \\
                if False else (x * w.view(1, 1, -1, 1)).sum(dim=2)
        return self._get_attribute(x, self.key_attr_idx)'''

assert old_get_attr in src, "_get_attribute method not found verbatim"
src = src.replace(old_get_attr, new_get_attr, 1)

# ---------------------------------------------------------------------------
# 3. forward(): use _get_key_attribute instead of hard _get_attribute for K.
# ---------------------------------------------------------------------------
old_fwd = '''        # Extract attributes for joining
        q_attr = self._get_attribute(Q, self.query_attr_idx)
        k_attr = self._get_attribute(K, self.key_attr_idx)'''

new_fwd = '''        # Extract attributes for joining
        q_attr = self._get_attribute(Q, self.query_attr_idx)
        k_attr = self._get_key_attribute(K)'''

assert old_fwd in src, "forward() attribute-extraction lines not found verbatim"
src = src.replace(old_fwd, new_fwd, 1)

# ---------------------------------------------------------------------------
# 4. MultiRelationAttention.__init__: accept pairing_strategy, assign
#    key_attr per head accordingly. query_attr stays i % k in all cases.
# ---------------------------------------------------------------------------
old_mra_sig = '''    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_attributes: int = 8,
        dropout: float = 0.1,
        use_gating: bool = True,
        use_mixing: bool = True
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, \\
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_attributes = num_attributes
        self.head_dim = hidden_dim // num_heads
        self.use_mixing = use_mixing

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
                    dropout=dropout,
                    use_gating=use_gating
                )
            )'''

new_mra_sig = '''    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_attributes: int = 8,
        dropout: float = 0.1,
        use_gating: bool = True,
        use_mixing: bool = True,
        pairing_strategy: str = "cyclic",
        pairing_seed: int = 1234
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, \\
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        assert pairing_strategy in ("cyclic", "random_fixed", "learned"), \\
            f"unknown pairing_strategy: {pairing_strategy}"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_attributes = num_attributes
        self.head_dim = hidden_dim // num_heads
        self.use_mixing = use_mixing
        self.pairing_strategy = pairing_strategy

        # Ablation (reviewer-requested, ICDE 2027 review D4/D7 & D2):
        # query_attr is i % k in every strategy, isolating the key-side
        # assignment as the sole varying factor across pairing strategies.
        #   cyclic:       key_attr = (i + 1) % k                 [original]
        #   random_fixed: key_attr ~ Uniform({0..k-1} \\ {query_attr}),
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
                    pairing_strategy=pairing_strategy
                )
            )'''

assert old_mra_sig in src, "MultiRelationAttention.__init__ not found verbatim"
src = src.replace(old_mra_sig, new_mra_sig, 1)

# ---------------------------------------------------------------------------
# 5. Ensure `random` is imported.
# ---------------------------------------------------------------------------
if re.search(r"^import random$", src, flags=re.M) is None:
    # Insert after the first import line.
    lines = src.split("\n")
    for idx, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            lines.insert(idx, "import random")
            break
    src = "\n".join(lines)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("Patched", PATH, "successfully.")

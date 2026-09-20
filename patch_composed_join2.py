"""
Thread use_composed_join through MultiRelationAttention, layers.py, and
model.py, mirroring the pairing_strategy threading pattern exactly.
Run on the remote host: python3 patch_composed_join2.py
"""

ATTN = "relational_attention/attention.py"
LAYERS = "relational_attention/layers.py"
MODEL = "relational_attention/model.py"

# ===========================================================================
# attention.py: MultiRelationAttention
# ===========================================================================
with open(ATTN, encoding="utf-8") as f:
    src = f.read()

old = '''    def __init__(
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
        self.pairing_strategy = pairing_strategy'''

new = '''    def __init__(
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
        self.use_composed_join = use_composed_join'''

assert old in src, "MultiRelationAttention.__init__ signature not found verbatim"
src = src.replace(old, new, 1)

old_head_ctor = '''            self.heads.append(
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
new_head_ctor = '''            self.heads.append(
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
            )'''
assert old_head_ctor in src, "RelationalAttentionHead(...) construction in MultiRelationAttention not found verbatim"
src = src.replace(old_head_ctor, new_head_ctor, 1)

with open(ATTN, "w", encoding="utf-8") as f:
    f.write(src)
print("Patched", ATTN, "(MultiRelationAttention threading)")

# ===========================================================================
# layers.py: RelationalTransformerBlock (encoder) and
# RelationalTransformerEncoderBlock (decoder, self+cross attention)
# ===========================================================================
with open(LAYERS, encoding="utf-8") as f:
    src = f.read()

old_enc = '''        use_gating: bool = True,
        use_mixing: bool = True,
        pairing_strategy: str = "cyclic",
        pairing_seed: int = 1234
    ):
        super().__init__()

        # Multi-Relation Self-Attention
        self.self_attention = MultiRelationAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_attributes=num_attributes,
            dropout=dropout,
            use_gating=use_gating,
            use_mixing=use_mixing,
            pairing_strategy=pairing_strategy,
            pairing_seed=pairing_seed
        )'''
new_enc = '''        use_gating: bool = True,
        use_mixing: bool = True,
        pairing_strategy: str = "cyclic",
        pairing_seed: int = 1234,
        use_composed_join: bool = False
    ):
        super().__init__()

        # Multi-Relation Self-Attention
        self.self_attention = MultiRelationAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_attributes=num_attributes,
            dropout=dropout,
            use_gating=use_gating,
            use_mixing=use_mixing,
            pairing_strategy=pairing_strategy,
            pairing_seed=pairing_seed,
            use_composed_join=use_composed_join
        )'''
assert old_enc in src, "RelationalTransformerBlock.__init__ pattern not found"
src = src.replace(old_enc, new_enc, 1)

old_dec = '''        dropout: float = 0.1,
        use_gating: bool = True,
        use_mixing: bool = True,
        pairing_strategy: str = "cyclic",
        pairing_seed: int = 1234
    ):
        super().__init__()

        # Self-attention
        self.self_attention = MultiRelationAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_attributes=num_attributes,
            dropout=dropout,
            use_gating=use_gating,
            use_mixing=use_mixing,
            pairing_strategy=pairing_strategy,
            pairing_seed=pairing_seed
        )

        # Cross-attention
        self.cross_attention = MultiRelationAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_attributes=num_attributes,
            dropout=dropout,
            use_gating=use_gating,
            use_mixing=use_mixing,
            pairing_strategy=pairing_strategy,
            pairing_seed=pairing_seed
        )'''
new_dec = '''        dropout: float = 0.1,
        use_gating: bool = True,
        use_mixing: bool = True,
        pairing_strategy: str = "cyclic",
        pairing_seed: int = 1234,
        use_composed_join: bool = False
    ):
        super().__init__()

        # Self-attention
        self.self_attention = MultiRelationAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_attributes=num_attributes,
            dropout=dropout,
            use_gating=use_gating,
            use_mixing=use_mixing,
            pairing_strategy=pairing_strategy,
            pairing_seed=pairing_seed,
            use_composed_join=use_composed_join
        )

        # Cross-attention
        self.cross_attention = MultiRelationAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_attributes=num_attributes,
            dropout=dropout,
            use_gating=use_gating,
            use_mixing=use_mixing,
            pairing_strategy=pairing_strategy,
            pairing_seed=pairing_seed,
            use_composed_join=use_composed_join
        )'''
assert old_dec in src, "RelationalTransformerEncoderBlock.__init__ pattern not found"
src = src.replace(old_dec, new_dec, 1)

with open(LAYERS, "w", encoding="utf-8") as f:
    f.write(src)
print("Patched", LAYERS)

# ===========================================================================
# model.py: RelationalTransformerConfig + both construction call sites
# ===========================================================================
with open(MODEL, encoding="utf-8") as f:
    src = f.read()

old_cfg = '''    pairing_strategy: str = "cyclic"  # {"cyclic", "random_fixed", "learned"}
    pairing_seed: int = 1234'''
new_cfg = '''    pairing_strategy: str = "cyclic"  # {"cyclic", "random_fixed", "learned"}
    pairing_seed: int = 1234
    use_composed_join: bool = False  # scoped 2-hop relational composition'''
assert old_cfg in src, "config field block not found"
src = src.replace(old_cfg, new_cfg, 1)

old_enc_call = '''                pairing_strategy=config.pairing_strategy,
                pairing_seed=config.pairing_seed
            )
            for _ in range(config.num_encoder_layers)
        ])'''
new_enc_call = '''                pairing_strategy=config.pairing_strategy,
                pairing_seed=config.pairing_seed,
                use_composed_join=config.use_composed_join
            )
            for _ in range(config.num_encoder_layers)
        ])'''
assert old_enc_call in src, "encoder layer construction call not found"
src = src.replace(old_enc_call, new_enc_call, 1)

old_dec_call = '''                pairing_strategy=config.pairing_strategy,
                pairing_seed=config.pairing_seed
            )
            for _ in range(config.num_decoder_layers)
        ])'''
new_dec_call = '''                pairing_strategy=config.pairing_strategy,
                pairing_seed=config.pairing_seed,
                use_composed_join=config.use_composed_join
            )
            for _ in range(config.num_decoder_layers)
        ])'''
assert old_dec_call in src, "decoder layer construction call not found"
src = src.replace(old_dec_call, new_dec_call, 1)

with open(MODEL, "w", encoding="utf-8") as f:
    f.write(src)
print("Patched", MODEL)

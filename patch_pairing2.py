"""
Thread pairing_strategy / pairing_seed through layers.py and model.py so it
is configurable end-to-end from RelationalTransformerConfig down to
MultiRelationAttention, for both encoder (RelationalTransformerBlock) and
decoder (RelationalTransformerEncoderBlock, self- and cross-attention).

Run on the remote host: python3 patch_pairing2.py
"""

LAYERS = "relational_attention/layers.py"
MODEL = "relational_attention/model.py"

# ===========================================================================
# layers.py
# ===========================================================================
with open(LAYERS, encoding="utf-8") as f:
    src = f.read()

# --- RelationalTransformerBlock (encoder) ---------------------------------
old = '''        use_gating: bool = True,
        use_mixing: bool = True
    ):
        super().__init__()

        # Multi-Relation Self-Attention
        self.self_attention = MultiRelationAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_attributes=num_attributes,
            dropout=dropout,
            use_gating=use_gating,
            use_mixing=use_mixing
        )'''
new = '''        use_gating: bool = True,
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
assert old in src, "RelationalTransformerBlock.__init__ pattern not found"
src = src.replace(old, new, 1)

# --- RelationalTransformerEncoderBlock (decoder: self + cross attention) --
old = '''    def __init__(
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
        )'''
new = '''    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_attributes: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
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
            pairing_strategy=pairing_strategy,
            pairing_seed=pairing_seed
        )

        # Cross-attention
        self.cross_attention = MultiRelationAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_attributes=num_attributes,
            dropout=dropout,
            pairing_strategy=pairing_strategy,
            pairing_seed=pairing_seed
        )'''
assert old in src, "RelationalTransformerEncoderBlock.__init__ pattern not found"
src = src.replace(old, new, 1)

with open(LAYERS, "w", encoding="utf-8") as f:
    f.write(src)
print("Patched", LAYERS)

# ===========================================================================
# model.py
# ===========================================================================
with open(MODEL, encoding="utf-8") as f:
    src = f.read()

old_cfg = '''    num_attributes: int = 8
    use_gating: bool = True
    use_mixing: bool = True'''
new_cfg = '''    num_attributes: int = 8
    use_gating: bool = True
    use_mixing: bool = True
    pairing_strategy: str = "cyclic"  # {"cyclic", "random_fixed", "learned"}
    pairing_seed: int = 1234'''
assert old_cfg in src, "RelationalTransformerConfig field block not found"
src = src.replace(old_cfg, new_cfg, 1)

old_enc = '''            RelationalTransformerBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_attributes=config.num_attributes,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
                use_gating=config.use_gating,
                use_mixing=config.use_mixing
            )'''
new_enc = '''            RelationalTransformerBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_attributes=config.num_attributes,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
                use_gating=config.use_gating,
                use_mixing=config.use_mixing,
                pairing_strategy=config.pairing_strategy,
                pairing_seed=config.pairing_seed
            )'''
if old_enc not in src:
    raise SystemExit(
        "RelationalTransformerBlock(...) call site in model.py did not match "
        "expected text verbatim -- inspect model.py:~110-120 manually and "
        "adjust this patch before re-running."
    )
src = src.replace(old_enc, new_enc, 1)

old_dec = '''            RelationalTransformerEncoderBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_attributes=config.num_attributes,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout
            )'''
new_dec = '''            RelationalTransformerEncoderBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_attributes=config.num_attributes,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
                pairing_strategy=config.pairing_strategy,
                pairing_seed=config.pairing_seed
            )'''
if old_dec not in src:
    raise SystemExit(
        "RelationalTransformerEncoderBlock(...) call site in model.py did "
        "not match expected text verbatim -- inspect model.py:~218-226 "
        "manually and adjust this patch before re-running."
    )
src = src.replace(old_dec, new_dec, 1)

with open(MODEL, "w", encoding="utf-8") as f:
    f.write(src)
print("Patched", MODEL)

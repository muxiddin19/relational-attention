"""
Fix a real architectural gap: RelationalTransformerEncoderBlock (the decoder
block, self+cross attention) never accepted use_gating/use_mixing, so the
decoder's attention always used gating and mixing regardless of the ablation
setting -- even after train.py correctly forwards them. This means the
no_gating/no_mixing ablations previously run only ever affected the encoder
half of the model and must be re-run once this is fixed.

Run on the remote host: python3 patch_decoder_gating.py
"""

LAYERS = "relational_attention/layers.py"
MODEL = "relational_attention/model.py"

# ===========================================================================
# layers.py: RelationalTransformerEncoderBlock (decoder block)
# ===========================================================================
with open(LAYERS, encoding="utf-8") as f:
    src = f.read()

old = '''    def __init__(
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
new = '''    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_attributes: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
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
assert old in src, "RelationalTransformerEncoderBlock.__init__ pattern not found"
src = src.replace(old, new, 1)

with open(LAYERS, "w", encoding="utf-8") as f:
    f.write(src)
print("Patched", LAYERS)

# ===========================================================================
# model.py: RelationalTransformerDecoder's construction of
# RelationalTransformerEncoderBlock must now pass use_gating/use_mixing too.
# ===========================================================================
with open(MODEL, encoding="utf-8") as f:
    src = f.read()

old_dec = '''            RelationalTransformerEncoderBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_attributes=config.num_attributes,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
                pairing_strategy=config.pairing_strategy,
                pairing_seed=config.pairing_seed
            )'''
new_dec = '''            RelationalTransformerEncoderBlock(
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
assert old_dec in src, "RelationalTransformerEncoderBlock(...) call site in model.py not found"
src = src.replace(old_dec, new_dec, 1)

with open(MODEL, "w", encoding="utf-8") as f:
    f.write(src)
print("Patched", MODEL)

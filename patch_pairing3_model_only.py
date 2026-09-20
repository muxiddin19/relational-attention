"""model.py-only version (layers.py already patched successfully)."""

MODEL = "relational_attention/model.py"

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
if old_cfg not in src:
    raise SystemExit("config field block not found (may already be patched)")
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
    raise SystemExit("encoder call site not found (may already be patched)")
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
    raise SystemExit("decoder call site not found (may already be patched)")
src = src.replace(old_dec, new_dec, 1)

with open(MODEL, "w", encoding="utf-8") as f:
    f.write(src)
print("Patched", MODEL, "successfully.")

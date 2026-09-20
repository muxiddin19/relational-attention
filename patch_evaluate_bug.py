"""
Fix the same class of bug in evaluate.py's load_model(): it reconstructs
RelationalTransformerConfig from the checkpoint's saved config.yaml but never
forwards use_gating/use_mixing/pairing_strategy/pairing_seed, even though
config.yaml (dumped by train.py) already contains them. This broke loading
any checkpoint trained with a non-default ablation flag once train.py was
fixed to actually respect those flags (previously this was silently masked
because the buggy train.py always produced a use_gating=True/use_mixing=True
model regardless of the YAML, matching load_model()'s hardcoded defaults).

Run on the remote host: python3 patch_evaluate_bug.py
"""

PATH = "scripts/evaluate.py"

with open(PATH, encoding="utf-8") as f:
    src = f.read()

old = '''    model_cfg = RelationalTransformerConfig(
        vocab_size=cfg_dict["vocab_size"],
        hidden_dim=cfg_dict["hidden_dim"],
        num_encoder_layers=cfg_dict["num_encoder_layers"],
        num_decoder_layers=cfg_dict["num_decoder_layers"],
        num_heads=cfg_dict["num_heads"],
        num_attributes=k,
        ffn_dim=cfg_dict.get("ffn_dim", cfg_dict["hidden_dim"] * 4),
        max_seq_len=cfg_dict.get("max_seq_len", 512),
        copy_mechanism=cfg_dict.get("copy_mechanism", False),
        num_copy_heads=cfg_dict.get("num_copy_heads", 1),
    )'''
new = '''    model_cfg = RelationalTransformerConfig(
        vocab_size=cfg_dict["vocab_size"],
        hidden_dim=cfg_dict["hidden_dim"],
        num_encoder_layers=cfg_dict["num_encoder_layers"],
        num_decoder_layers=cfg_dict["num_decoder_layers"],
        num_heads=cfg_dict["num_heads"],
        num_attributes=k,
        use_gating=cfg_dict.get("use_gating", True),
        use_mixing=cfg_dict.get("use_mixing", True),
        pairing_strategy=cfg_dict.get("pairing_strategy", "cyclic"),
        pairing_seed=cfg_dict.get("pairing_seed", 1234),
        ffn_dim=cfg_dict.get("ffn_dim", cfg_dict["hidden_dim"] * 4),
        max_seq_len=cfg_dict.get("max_seq_len", 512),
        copy_mechanism=cfg_dict.get("copy_mechanism", False),
        num_copy_heads=cfg_dict.get("num_copy_heads", 1),
    )'''
assert old in src, "load_model() RelationalTransformerConfig(...) call not found verbatim"
src = src.replace(old, new, 1)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)
print("Patched", PATH, "successfully.")

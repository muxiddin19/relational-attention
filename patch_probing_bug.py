"""
Fix the same class of bug in analysis/probing.py: model_cfg reconstruction
from a checkpoint's config.yaml never forwards use_gating/use_mixing/
pairing_strategy/pairing_seed. For a "learned" checkpoint this crashes on
load_state_dict (missing key_attr_logits params); for "random_fixed" it is
worse -- it silently succeeds but reconstructs the model with default
pairing_strategy="cyclic", so probing attributes representations to the
wrong head-to-slot mapping without raising any error.

Run on the remote host: python3 patch_probing_bug.py
"""

PATH = "analysis/probing.py"

with open(PATH, encoding="utf-8") as f:
    src = f.read()

old = '''    model_cfg = RelationalTransformerConfig(
        vocab_size=cfg["vocab_size"],
        hidden_dim=cfg["hidden_dim"],
        num_encoder_layers=cfg["num_encoder_layers"],
        num_decoder_layers=cfg["num_decoder_layers"],
        num_heads=cfg["num_heads"],
        num_attributes=k,
        ffn_dim=cfg.get("ffn_dim", cfg["hidden_dim"] * 4),
        max_seq_len=cfg.get("max_seq_len", 512),
    )'''
new = '''    model_cfg = RelationalTransformerConfig(
        vocab_size=cfg["vocab_size"],
        hidden_dim=cfg["hidden_dim"],
        num_encoder_layers=cfg["num_encoder_layers"],
        num_decoder_layers=cfg["num_decoder_layers"],
        num_heads=cfg["num_heads"],
        num_attributes=k,
        use_gating=cfg.get("use_gating", True),
        use_mixing=cfg.get("use_mixing", True),
        pairing_strategy=cfg.get("pairing_strategy", "cyclic"),
        pairing_seed=cfg.get("pairing_seed", 1234),
        ffn_dim=cfg.get("ffn_dim", cfg["hidden_dim"] * 4),
        max_seq_len=cfg.get("max_seq_len", 512),
    )'''
assert old in src, "probing.py model_cfg construction not found verbatim"
src = src.replace(old, new, 1)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)
print("Patched", PATH, "successfully.")

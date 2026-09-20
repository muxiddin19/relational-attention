"""
Fix a real bug in scripts/train.py: TrainingConfig reads use_gating/use_mixing
from the YAML config, but build_model() never forwards them into
RelationalTransformerConfig, so every gating/mixing ablation silently trained
with the defaults (use_gating=True, use_mixing=True) regardless of the YAML.
This explains the identical no_gating/no_mixing eval predictions.

Also adds pairing_strategy/pairing_seed plumbing so the new ablation is not
hit by the same class of bug.

Run on the remote host: python3 patch_train_bug.py
"""

PATH = "scripts/train.py"

with open(PATH, encoding="utf-8") as f:
    src = f.read()

old_cfg = '''    num_attributes: int = 8              # k; set to 1 for standard attention
    use_gating: bool = True             # ablation: set False to disable NeuralSelection
    use_mixing: bool = True             # ablation: set False to use averaging instead of W^O
    ffn_dim: int = 2048'''
new_cfg = '''    num_attributes: int = 8              # k; set to 1 for standard attention
    use_gating: bool = True             # ablation: set False to disable NeuralSelection
    use_mixing: bool = True             # ablation: set False to use averaging instead of W^O
    pairing_strategy: str = "cyclic"    # ablation: {"cyclic", "random_fixed", "learned"}
    pairing_seed: int = 1234
    ffn_dim: int = 2048'''
assert old_cfg in src, "TrainingConfig field block not found verbatim"
src = src.replace(old_cfg, new_cfg, 1)

old_build = '''    model_cfg = RelationalTransformerConfig(
        vocab_size=cfg.vocab_size,
        hidden_dim=cfg.hidden_dim,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        num_heads=cfg.num_heads,
        num_attributes=k,
        ffn_dim=cfg.ffn_dim,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
        copy_mechanism=cfg.copy_mechanism,
        num_copy_heads=cfg.num_copy_heads)'''
new_build = '''    model_cfg = RelationalTransformerConfig(
        vocab_size=cfg.vocab_size,
        hidden_dim=cfg.hidden_dim,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        num_heads=cfg.num_heads,
        num_attributes=k,
        use_gating=cfg.use_gating,
        use_mixing=cfg.use_mixing,
        pairing_strategy=cfg.pairing_strategy,
        pairing_seed=cfg.pairing_seed,
        ffn_dim=cfg.ffn_dim,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
        copy_mechanism=cfg.copy_mechanism,
        num_copy_heads=cfg.num_copy_heads)'''
assert old_build in src, "build_model() RelationalTransformerConfig(...) call not found verbatim"
src = src.replace(old_build, new_build, 1)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)
print("Patched", PATH, "successfully.")

"""
Thread use_composed_join through scripts/train.py (TrainingConfig +
build_model) and scripts/evaluate.py (load_model), matching the exact
pattern already fixed for pairing_strategy/pairing_seed -- to avoid
reintroducing the same class of bug found earlier in this codebase.
Run on the remote host: python3 patch_composed_join3.py
"""

TRAIN = "scripts/train.py"
EVAL = "scripts/evaluate.py"

with open(TRAIN, encoding="utf-8") as f:
    src = f.read()

old_cfg = '''    pairing_strategy: str = "cyclic"    # ablation: {"cyclic", "random_fixed", "learned"}
    pairing_seed: int = 1234
    ffn_dim: int = 2048'''
new_cfg = '''    pairing_strategy: str = "cyclic"    # ablation: {"cyclic", "random_fixed", "learned"}
    pairing_seed: int = 1234
    use_composed_join: bool = False     # scoped 2-hop relational composition
    ffn_dim: int = 2048'''
assert old_cfg in src, "TrainingConfig field block not found"
src = src.replace(old_cfg, new_cfg, 1)

old_build = '''        pairing_strategy=cfg.pairing_strategy,
        pairing_seed=cfg.pairing_seed,
        ffn_dim=cfg.ffn_dim,'''
new_build = '''        pairing_strategy=cfg.pairing_strategy,
        pairing_seed=cfg.pairing_seed,
        use_composed_join=cfg.use_composed_join,
        ffn_dim=cfg.ffn_dim,'''
assert old_build in src, "build_model() RelationalTransformerConfig(...) call not found"
src = src.replace(old_build, new_build, 1)

with open(TRAIN, "w", encoding="utf-8") as f:
    f.write(src)
print("Patched", TRAIN)

with open(EVAL, encoding="utf-8") as f:
    src = f.read()

old_eval = '''        pairing_strategy=cfg_dict.get("pairing_strategy", "cyclic"),
        pairing_seed=cfg_dict.get("pairing_seed", 1234),
        ffn_dim=cfg_dict.get("ffn_dim", cfg_dict["hidden_dim"] * 4),'''
new_eval = '''        pairing_strategy=cfg_dict.get("pairing_strategy", "cyclic"),
        pairing_seed=cfg_dict.get("pairing_seed", 1234),
        use_composed_join=cfg_dict.get("use_composed_join", False),
        ffn_dim=cfg_dict.get("ffn_dim", cfg_dict["hidden_dim"] * 4),'''
assert old_eval in src, "evaluate.py load_model() RelationalTransformerConfig(...) call not found"
src = src.replace(old_eval, new_eval, 1)

with open(EVAL, "w", encoding="utf-8") as f:
    f.write(src)
print("Patched", EVAL)

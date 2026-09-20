import sys
from pathlib import Path
sys.path.insert(0, "/home/muhiddin/relational-attention")
sys.path.insert(0, "/home/muhiddin/relational-attention/scripts")
sys.path.insert(0, "/home/muhiddin/relational-attention/analysis")

import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial
import yaml

from relational_attention import RelationalTransformer, RelationalTransformerConfig
from train import load_dataset_examples, Seq2SeqDataset, collate_fn, SPTokenizer
import probing as probing_mod

ckpt = Path("/nas/muhiddin/gsm8k_final_local/relational_125m_gsm8k_final_s42/best_model")
cfg = yaml.safe_load(open(ckpt / "config.yaml"))
k = cfg["num_attributes"]

model_cfg = RelationalTransformerConfig(
    vocab_size=cfg["vocab_size"], hidden_dim=cfg["hidden_dim"],
    num_encoder_layers=cfg["num_encoder_layers"], num_decoder_layers=cfg["num_decoder_layers"],
    num_heads=cfg["num_heads"], num_attributes=k,
    use_gating=cfg.get("use_gating", True), use_mixing=cfg.get("use_mixing", True),
    pairing_strategy=cfg.get("pairing_strategy", "cyclic"),
    pairing_seed=cfg.get("pairing_seed", 1234),
    ffn_dim=cfg.get("ffn_dim", cfg["hidden_dim"] * 4),
    max_seq_len=cfg.get("max_seq_len", 512),
)
model = RelationalTransformer(model_cfg)
sd = torch.load(ckpt / "model.pt", map_location="cuda")
model.load_state_dict(sd, strict=False)
model.eval().cuda()

tokenizer = SPTokenizer("/nas/Dataset/nlp/tokenizer/sp32k.model")
examples = load_dataset_examples("spider", "dev", "/nas/Dataset/nlp")
ds = Seq2SeqDataset(examples, tokenizer, cfg.get("max_src_len", 256), cfg.get("max_tgt_len", 128))
dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=partial(collate_fn, pad_id=tokenizer.pad_id))

slot_embs, labels = probing_mod.extract_representations(model, dl, "cuda", k)
print("labels unique:", np.unique(labels, return_counts=True))
for j in range(k):
    e = slot_embs[j]
    print(f"slot {j}: shape={e.shape} mean={e.mean():.6f} std={e.std():.6f} min={e.min():.4f} max={e.max():.4f}")

print("\nFull precision Fisher F (unrounded):")
roles = ["Identity", "FD", "Schema"]
for j in range(k):
    row = [probing_mod.compute_fisher_f(slot_embs[j], labels, r_idx + 1) for r_idx in range(len(roles))]
    print(f"slot {j}:", row)

# Also try with role_label offsets 0,1,2 instead of 1,2,3 (checking labeling convention bug)
print("\nFull precision Fisher F (role_label = 0,1,2 instead of 1,2,3):")
for j in range(k):
    row = [probing_mod.compute_fisher_f(slot_embs[j], labels, r_idx) for r_idx in range(len(roles))]
    print(f"slot {j}:", row)

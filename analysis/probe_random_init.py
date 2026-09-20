import sys, json
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

torch.manual_seed(1234)

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
model = RelationalTransformer(model_cfg)  # freshly initialized, NO checkpoint loaded
model.eval().cuda()
print(f"Random-init model ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params) -- NOT loading any checkpoint")

tokenizer = SPTokenizer("/nas/Dataset/nlp/tokenizer/sp32k.model")
examples = load_dataset_examples("spider", "dev", "/nas/Dataset/nlp")
ds = Seq2SeqDataset(examples, tokenizer, cfg.get("max_src_len", 256), cfg.get("max_tgt_len", 128))
dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=partial(collate_fn, pad_id=tokenizer.pad_id))

print(f"Probing {len(examples)} Spider dev examples on RANDOM-INIT model...")
slot_embs, labels = probing_mod.extract_representations(model, dl, "cuda", k)

roles = ["Identity", "FD", "Schema"]
fisher_f = []
for j in range(k):
    row = [round(probing_mod.compute_fisher_f(slot_embs[j], labels, r_idx + 1), 4)
           for r_idx in range(len(roles))]
    fisher_f.append(row)

probing_mod.print_table(fisher_f, roles)
result = {"source": "computed_random_init", "fisher_f": fisher_f, "roles": roles, "k": k, "n_examples": len(examples)}
json.dump(result, open("/home/muhiddin/rel_data/probing_results_random_init.json", "w"), indent=2)
print("Saved.")

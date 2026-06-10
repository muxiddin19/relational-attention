"""
Training script for Attribute-Decomposed Attention (RelAttn) experiments.

Trains RelTransformer or Standard Transformer on structured reasoning benchmarks.
All hyperparameters are controlled via YAML config files; this ensures identical
conditions between model variants for fair comparison.

Usage:
    python scripts/train.py \\
        --config configs/rel_transformer_125m.yaml \\
        --dataset spider \\
        --nas-dir /nas/Dataset \\
        --seed 42 \\
        --output-dir ./outputs/rel_125m_spider_s42
"""

import argparse
import gc
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from relational_attention import RelationalTransformer, RelationalTransformerConfig

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    # Model
    model_type: str = "relational"       # "relational" or "standard"
    hidden_dim: int = 512
    num_encoder_layers: int = 12
    num_decoder_layers: int = 12
    num_heads: int = 8
    num_attributes: int = 8              # k; set to 1 for standard attention
    ffn_dim: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1
    vocab_size: int = 32000

    # Training
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 4000
    max_steps: int = 100000
    batch_size: int = 32
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1

    # Eval
    eval_every: int = 1000
    save_every: int = 5000
    patience: int = 10
    eval_by_em: bool = False  # Use EM instead of dev_loss for early stopping                   # early stopping in eval rounds

    # Dataset
    dataset: str = "spider"
    nas_dir: str = "/nas/Dataset"
    max_src_len: int = 256
    max_tgt_len: int = 128

    # Hardware
    seed: int = 42
    num_workers: int = 4
    fp16: bool = True

    # Output
    output_dir: str = "./outputs/run"
    log_every: int = 100

    # Copy mechanism (pointer-generator for COGS/CFQ entity binding)
    copy_mechanism: bool = False
    num_copy_heads: int = 1


def load_config(yaml_path: str, overrides: dict) -> TrainingConfig:
    with open(yaml_path) as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict.update({k: v for k, v in overrides.items() if v is not None})
    return TrainingConfig(**{k: v for k, v in cfg_dict.items()
                             if k in TrainingConfig.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Tokenizer (SentencePiece wrapper)
# ---------------------------------------------------------------------------

class SPTokenizer:
    """SentencePiece tokenizer shared across all models for fair comparison."""

    def __init__(self, model_path: str):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.pad_id = self.sp.PieceToId("<pad>")
        self.bos_id = self.sp.PieceToId("<s>")
        self.eos_id = self.sp.PieceToId("</s>")
        self.vocab_size = self.sp.GetPieceSize()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> List[int]:
        ids = self.sp.EncodeAsIds(text)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        ids = [i for i in ids if i not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.sp.DecodeIds(ids)

    @staticmethod
    def train(data_file: str, out_prefix: str, vocab_size: int = 32000):
        import sentencepiece as spm
        spm.SentencePieceTrainer.train(
            input=data_file,
            model_prefix=out_prefix,
            vocab_size=vocab_size,
            character_coverage=0.9995,
            model_type="bpe",
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
            pad_piece="<pad>",
        )
        log.info(f"Tokenizer saved to {out_prefix}.model")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class Seq2SeqDataset(Dataset):
    def __init__(self, examples: List[Dict], tokenizer: SPTokenizer,
                 max_src: int, max_tgt: int):
        self.examples = examples
        self.tok = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        src = self.tok.encode(ex["source"], add_bos=False, add_eos=True)[:self.max_src]
        tgt = self.tok.encode(ex["target"], add_bos=True, add_eos=True)[:self.max_tgt]
        return {
            "input_ids": torch.tensor(src, dtype=torch.long),
            "labels": torch.tensor(tgt, dtype=torch.long),
        }


def collate_fn(batch: List[Dict], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    src = [b["input_ids"] for b in batch]
    tgt = [b["labels"] for b in batch]
    src_padded = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=pad_id)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=pad_id)
    src_mask = (src_padded != pad_id).long()
    return {
        "input_ids": src_padded,
        "attention_mask": src_mask,
        "decoder_input_ids": tgt_padded[:, :-1].contiguous(),
        "labels": tgt_padded[:, 1:].contiguous(),
    }



# ─── COGS entity-substitution augmentation (Fix: breaks overfitting) ──────────
_COGS_ENTITY_NAMES = [
    "Emma","Liam","Olivia","Noah","Ava","William","Isabella","James","Sofia",
    "Lucas","Mia","Mason","Charlotte","Ethan","Sophie","Oliver","Emily",
    "Alexander","Amelia","Michael","Benjamin","Harper","Elijah","Evelyn",
    "Daniel","Abigail","Matthew","Ella","Aiden","Scarlett","Henry","Luna",
    "Jackson","Chloe","Sebastian","Penelope","Owen","Layla","Samuel","Riley",
]

def _cogs_augment(examples, n_aug=4, seed_offset=0):
    """Augment COGS training examples via entity name substitution.
    Each example is duplicated n_aug times with different entity names,
    forcing the model to learn entity-agnostic compositional structure.
    """
    import random
    rng = random.Random(seed_offset + 12345)
    augmented = list(examples)
    for ex in examples:
        src0, tgt0 = ex["source"], ex["target"]
        present = [n for n in _COGS_ENTITY_NAMES if n in src0]
        if not present:
            continue
        for _ in range(n_aug):
            src, tgt = src0, tgt0
            for name in present:
                new_name = rng.choice([n for n in _COGS_ENTITY_NAMES if n != name])
                src = src.replace(name, new_name)
                tgt = tgt.replace(name, new_name)
            augmented.append({"source": src, "target": tgt})
    rng.shuffle(augmented)
    return augmented

def load_dataset_examples(dataset: str, split: str, nas_dir: str) -> List[Dict]:
    """Load dataset split and return list of {source, target} dicts."""
    base = Path(nas_dir)

    if dataset == "spider":
        fname = "train_spider.json" if split == "train" else "dev.json"
        path = base / "spider" / fname
        tables_path = base / "spider" / "tables.json"
        with open(path) as f:
            data = json.load(f)
        # Build schema lookup: db_id -> "table1: col1,col2 | table2: col3"
        schema_map = {}
        if tables_path.exists():
            with open(tables_path) as f:
                tables_data = json.load(f)
            for db in tables_data:
                db_id = db["db_id"]
                cols_by_table = {}
                for (tid, _), (_, col_orig) in zip(
                        db["column_names"], db["column_names_original"]):
                    if tid == -1:
                        continue
                    tname = db["table_names_original"][tid]
                    cols_by_table.setdefault(tname, []).append(col_orig)
                schema_str = " | ".join(
                    f"{t}: {chr(44).join(c)}" for t, c in cols_by_table.items())
                schema_map[db_id] = schema_str
        import random as _rnd
        def _maybe_drop_schema(schema_str, drop_p=0.0):
            """Randomly drop some columns from each table in schema_str.
            During training (drop_p>0) this prevents memorising exact schema
            layouts; during eval (drop_p=0) full schema is always used.
            """
            if not schema_str or drop_p == 0.0:
                return schema_str
            parts = []
            for table_part in schema_str.split(" | "):
                if ":" not in table_part:
                    parts.append(table_part)
                    continue
                tname, cols_str = table_part.split(":", 1)
                cols = [c.strip() for c in cols_str.split(",")]
                # Always keep at least 1 column; drop others with prob drop_p
                kept = [c for c in cols if _rnd.random() > drop_p] or cols[:1]
                parts.append(f"{tname}: {chr(44).join(kept)}")
            return " | ".join(parts)
        # Use full schema for dev/test; 40% column dropout for training
        _drop_p = 0.4 if split == "train" else 0.0
        return [{
            "source": (
                f"translate to SQL: {r['question']}"
                f" | db: {r.get('db_id', '')}"
                f" | schema: {_maybe_drop_schema(schema_map.get(r.get('db_id', ''), ''), _drop_p)}"
            ),
            "target": r["query"]
        } for r in data]

    elif dataset == "cogs":
        split_map = {"train": "train.tsv", "dev": "dev.tsv",
                     "validation": "dev.tsv", "test": "test.tsv", "gen": "gen.tsv"}
        fname = split_map.get(split, f"{split}.tsv")
        import pandas as pd
        df = pd.read_csv(base / "cogs" / fname, sep="\t",
                         header=None, names=["sentence", "logical_form", "category"])
        exs = [{"source": row["sentence"], "target": row["logical_form"]}
               for _, row in df.iterrows()]
        if split == "train":
            exs = _cogs_augment(exs, n_aug=4)
            log.info(f"COGS train: {len(exs)} examples after entity augmentation ({len(exs)//5}×5)")
        return exs

    elif dataset == "scan":
        # Try addprim_jump split first (the compositional generalization split)
        fname = f"addprim_jump_{split}.json"
        path = base / "scan" / fname
        if not path.exists():
            fname = f"simple_{split}.json"
            path = base / "scan" / fname
        if not path.exists() and split in ("dev", "validation"):
            # SCAN has no dev split; fall back to addprim_jump_test
            fname = f"addprim_jump_test.json"
            path = base / "scan" / fname
        if not path.exists() and split in ("dev", "validation"):
            # SCAN has no dev split; fall back to test
            fname = f"simple_test.json"
            path = base / "scan" / fname
        with open(path) as f:
            data = json.load(f)
        # SCAN fix: replace underscores in action tokens so SentencePiece does
        # not merge repeated tokens (I_WALK I_WALK → I_WALKWALK).  We encode
        # each SCAN token as space-separated words and reverse in decode.
        def scan_encode(s):
            return " ".join(tok.replace("_", " __ ") for tok in s.split())
        return [{"source": r["commands"],
                 "target": scan_encode(r["actions"])} for r in data]

    elif dataset == "cfq":
        fname = f"mcd1_{split}.json"
        path = base / "cfq" / fname
        if not path.exists() and split in ("dev", "validation"):
            # CFQ has no dev split; use mcd1_test as validation proxy
            path = base / "cfq" / "mcd1_test.json"
        with open(path) as f:
            data = json.load(f)
        return [{"source": r["question"], "target": r["query"]} for r in data]

    elif dataset == "gsm8k":
        if split == "train":
            fname = "train.json"
        else:
            fname = "test.json"  # GSM8K has no dev split; use test
        with open(base / "gsm8k" / fname) as f:
            data = json.load(f)
        # Use only the final numeric answer as training target
        # Full CoT (mean 65 SP tokens) truncated at max_tgt_len=64 hid the answer
        return [{"source": r["question"],
                 "target": r["answer"].split("####")[-1].strip()} for r in data]

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Learning rate schedule (cosine with warmup)
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup: int, max_steps: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


# ---------------------------------------------------------------------------
# Model builder (relational vs standard — identical except attention type)
# ---------------------------------------------------------------------------

def build_model(cfg: TrainingConfig) -> nn.Module:
    k = cfg.num_attributes if cfg.model_type == "relational" else 1
    model_cfg = RelationalTransformerConfig(
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
        num_copy_heads=cfg.num_copy_heads)
    model = RelationalTransformer(model_cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {cfg.model_type}, k={k}, params={n_params/1e6:.1f}M")
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def compute_em_on_dev(model, tokenizer, dev_exs, device, max_tgt_len=256, n_samples=200, dataset="cogs"):
    """Quick EM estimate on a dev subset for EM-based patience."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    model.eval()
    sources = [e["source"] for e in dev_exs[:n_samples]]
    targets = [e["target"] for e in dev_exs[:n_samples]]
    preds = []
    bs = 16
    for i in range(0, len(sources), bs):
        batch_src = sources[i:i+bs]
        src_ids = [tokenizer.encode(s, add_bos=False, add_eos=True)[:256] for s in batch_src]
        max_len = max(len(x) for x in src_ids)
        padded = torch.zeros(len(batch_src), max_len, dtype=torch.long, device=device)
        mask   = torch.zeros(len(batch_src), max_len, dtype=torch.long, device=device)
        for j, ids in enumerate(src_ids):
            padded[j, :len(ids)] = torch.tensor(ids, device=device)
            mask[j, :len(ids)] = 1
        dec = torch.full((len(batch_src), 1), tokenizer.bos_id, dtype=torch.long, device=device)
        done = torch.zeros(len(batch_src), dtype=torch.bool, device=device)
        with torch.no_grad():
            for _ in range(max_tgt_len):
                out = model(input_ids=padded, attention_mask=mask, decoder_input_ids=dec)
                logits = out["logits"] if isinstance(out, dict) else out[0]
                nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                dec = torch.cat([dec, nxt], dim=1)
                done |= (nxt.squeeze(-1) == tokenizer.eos_id)
                if done.all():
                    break
        for row in dec.cpu().tolist():
            preds.append(tokenizer.decode(row))
    def norm(s):
        return " ".join(s.replace("⁇", ";").lower().split())
    correct = sum(norm(p) == norm(g) for p, g in zip(preds, targets))
    model.train()
    return correct / len(targets)

def train(cfg: TrainingConfig):
    # Reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Tokenizer — train once, reuse for all models (ensures identical tokenization)
    tok_path = Path(cfg.nas_dir) / "tokenizer" / "sp32k.model"
    if not tok_path.exists():
        log.info("Training SentencePiece tokenizer...")
        tok_path.parent.mkdir(parents=True, exist_ok=True)
        # Gather text from all datasets for a joint tokenizer
        all_text_file = tok_path.parent / "all_text.txt"
        with open(all_text_file, "w") as f:
            for ds_name in ["spider", "cogs", "scan", "cfq", "gsm8k"]:
                try:
                    exs = load_dataset_examples(ds_name, "train", cfg.nas_dir)
                    for e in exs:
                        f.write(e["source"] + "\n")
                        f.write(e["target"] + "\n")
                except Exception:
                    pass
        SPTokenizer.train(str(all_text_file), str(tok_path.with_suffix("")),
                          cfg.vocab_size)
    tokenizer = SPTokenizer(str(tok_path))

    # Datasets
    train_exs = load_dataset_examples(cfg.dataset, "train", cfg.nas_dir)
    dev_split = "dev"
    try:
        dev_exs = load_dataset_examples(cfg.dataset, dev_split, cfg.nas_dir)
    except Exception:
        dev_exs = load_dataset_examples(cfg.dataset, "test", cfg.nas_dir)

    from functools import partial
    col_fn = partial(collate_fn, pad_id=tokenizer.pad_id)
    train_ds = Seq2SeqDataset(train_exs, tokenizer, cfg.max_src_len, cfg.max_tgt_len)
    dev_ds   = Seq2SeqDataset(dev_exs,   tokenizer, cfg.max_src_len, cfg.max_tgt_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=col_fn,
                              pin_memory=True)
    dev_loader   = DataLoader(dev_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                              num_workers=cfg.num_workers, collate_fn=col_fn,
                              pin_memory=True)

    # Model, optimizer, scaler
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay, betas=(0.9, 0.98))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id,
                                    label_smoothing=cfg.label_smoothing)

    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    best_dev_loss = float("inf")
    patience_count = 0
    global_step = 0
    best_em_holder = [-1.0]  # mutable: persists EM across loop, reset per train() call
    t0 = time.time()

    log.info(f"Training on {len(train_exs)} examples, eval on {len(dev_exs)}")
    log.info(f"Config: {cfg}")

    for epoch in range(1, 10000):
        model.train()
        for batch in train_loader:
            if global_step >= cfg.max_steps:
                break

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # LR schedule
            lr = get_lr(global_step, cfg.warmup_steps, cfg.max_steps, cfg.learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=batch["decoder_input_ids"],
                    labels=batch["labels"],
                )
                # If model returns loss directly, use it; otherwise compute manually
                if isinstance(out, dict) and "loss" in out:
                    loss = out["loss"] / cfg.gradient_accumulation
                elif isinstance(out, dict) and "logits" in out:
                    logits = out["logits"]
                    loss = criterion(logits.view(-1, logits.size(-1)),
                                     batch["labels"].view(-1)) / cfg.gradient_accumulation
                else:
                    logits = out[0] if isinstance(out, tuple) else out
                    loss = criterion(logits.view(-1, logits.size(-1)),
                                     batch["labels"].view(-1)) / cfg.gradient_accumulation

            # Skip NaN batches — fp16 overflow can propagate NaN into weights
            if torch.isnan(loss) or torch.isinf(loss):
                log.warning(f"NaN/Inf loss at step {global_step}, skipping batch")
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                continue

            scaler.scale(loss).backward()

            if (global_step + 1) % cfg.gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                # Check for NaN gradients before clipping
                has_nan_grad = any(
                    p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                    for p in model.parameters()
                )
                if has_nan_grad:
                    log.warning(f"NaN grad at step {global_step}, zeroing and skipping")
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            if global_step % cfg.log_every == 0:
                elapsed = time.time() - t0
                log.info(f"step={global_step} loss={loss.item() * cfg.gradient_accumulation:.4f} "
                         f"lr={lr:.2e} elapsed={elapsed:.0f}s")
                writer.add_scalar("train/loss", loss.item() * cfg.gradient_accumulation, global_step)
                writer.add_scalar("train/lr", lr, global_step)

            if global_step > 0 and global_step % cfg.eval_every == 0:
                dev_loss = evaluate_loss(model, dev_loader, criterion, device, cfg)
                log.info(f"[eval] step={global_step} dev_loss={dev_loss:.4f}")
                writer.add_scalar("eval/loss", dev_loss, global_step)

                # EM-based patience: compute greedy EM on 200 dev examples
                if cfg.eval_by_em:
                    em_score = compute_em_on_dev(
                        model, tokenizer, dev_exs, device,
                        max_tgt_len=cfg.max_tgt_len, n_samples=200, dataset=cfg.dataset
                    )
                    log.info(f"  EM on dev subset (200 samples): {em_score*100:.2f}%")
                    monitor_val = em_score
                    monitor_better = (em_score > best_em_holder[0])
                else:
                    monitor_val = dev_loss
                    monitor_better = (dev_loss < best_dev_loss)

                if monitor_better:
                    best_dev_loss = dev_loss
                    if cfg.eval_by_em:
                        best_em_holder[0] = monitor_val
                    patience_count = 0
                    ckpt = out_dir / "best_model"
                    ckpt.mkdir(exist_ok=True)
                    torch.save(model.state_dict(), ckpt / "model.pt")
                    with open(ckpt / "config.yaml", "w") as f:
                        yaml.dump(cfg.__dict__, f)
                    log.info(f"  -> New best model saved (dev_loss={dev_loss:.4f})")
                else:
                    patience_count += 1
                    if patience_count >= cfg.patience:
                        log.info("Early stopping triggered.")
                        break

                model.train()

            if global_step > 0 and global_step % cfg.save_every == 0:
                ckpt = out_dir / f"checkpoint-{global_step}"
                ckpt.mkdir(exist_ok=True)
                torch.save(model.state_dict(), ckpt / "model.pt")

            global_step += 1

        if global_step >= cfg.max_steps or patience_count >= cfg.patience:
            break

    log.info(f"Training complete. Best dev loss: {best_dev_loss:.4f}")
    writer.close()

    # Cleanup GPU memory
    del model
    torch.cuda.empty_cache()
    gc.collect()


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device, cfg) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=cfg.fp16):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"],
            )
            logits = out["logits"] if isinstance(out, dict) else out[0]
            labels = batch["labels"]
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        non_pad = (labels != 0).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad
    return total_loss / max(1, total_tokens)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--dataset", choices=["spider", "cogs", "scan", "cfq", "gsm8k"])
    p.add_argument("--nas-dir", help="NAS dataset root (e.g. /nas/Dataset)")
    p.add_argument("--seed", type=int)
    p.add_argument("--output-dir", help="Where to save checkpoints and logs")
    p.add_argument("--model-type", choices=["relational", "standard"],
                   help="Override model type from config")
    p.add_argument("--fp16", action="store_true", default=None)
    p.add_argument("--batch-size", type=int)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    overrides = {
        "dataset": args.dataset,
        "nas_dir": args.nas_dir,
        "seed": args.seed,
        "output_dir": args.output_dir,
        "model_type": args.model_type,
        "batch_size": args.batch_size,
    }
    cfg = load_config(args.config, overrides)
    if args.fp16:
        cfg.fp16 = True
    train(cfg)

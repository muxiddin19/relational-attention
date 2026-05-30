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
    patience: int = 10                   # early stopping in eval rounds

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


def load_dataset_examples(dataset: str, split: str, nas_dir: str) -> List[Dict]:
    """Load dataset split and return list of {source, target} dicts."""
    base = Path(nas_dir)

    if dataset == "spider":
        fname = "train_spider.json" if split == "train" else "dev.json"
        path = base / "spider" / fname
        with open(path) as f:
            data = json.load(f)
        return [{"source": f"translate to SQL: {r['question']} | {r.get('db_id', '')}",
                 "target": r["query"]} for r in data]

    elif dataset == "cogs":
        split_map = {"train": "train.tsv", "dev": "dev.tsv",
                     "validation": "dev.tsv", "test": "test.tsv", "gen": "gen.tsv"}
        fname = split_map.get(split, f"{split}.tsv")
        import pandas as pd
        df = pd.read_csv(base / "cogs" / fname, sep="\t",
                         header=None, names=["sentence", "logical_form", "category"])
        return [{"source": row["sentence"], "target": row["logical_form"]}
                for _, row in df.iterrows()]

    elif dataset == "scan":
        fname = f"simple_{split}.json"
        path = base / "scan" / fname
        if not path.exists():
            fname = f"addprim_jump_{split}.json"
            path = base / "scan" / fname
        with open(path) as f:
            data = json.load(f)
        return [{"source": r["commands"], "target": r["actions"]} for r in data]

    elif dataset == "cfq":
        fname = f"mcd1_{split}.json"
        path = base / "cfq" / fname
        with open(path) as f:
            data = json.load(f)
        return [{"source": r["question"], "target": r["query"]} for r in data]

    elif dataset == "gsm8k":
        fname = "train.json" if split == "train" else "test.json"
        with open(base / "gsm8k" / fname) as f:
            data = json.load(f)
        return [{"source": r["question"], "target": r["answer"]} for r in data]

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
    )
    model = RelationalTransformer(model_cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {cfg.model_type}, k={k}, params={n_params/1e6:.1f}M")
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

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

            scaler.scale(loss).backward()

            if (global_step + 1) % cfg.gradient_accumulation == 0:
                scaler.unscale_(optimizer)
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

                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
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

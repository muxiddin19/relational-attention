#!/usr/bin/env python3
"""T5 fine-tuning with epoch-level train/dev loss logging.
Runs: (1) T5-Base vanilla, (2) T5-Base + RelAttn replacement.
Output: epoch_curves.json for each, used in Supplemental Remark 2.
"""
import argparse, json, logging, math, os, sys, time
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                    level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

NAS_NLP = "/nas/Dataset/nlp"
NAS_OUT = "/nas/Dataset/experiments/relational-attention/t5_finetuning"
Path(NAS_OUT).mkdir(parents=True, exist_ok=True)


class SpiderDS(Dataset):
    def __init__(self, tok, split="train", max_src=256, max_tgt=128):
        import json as _j
        p = Path(NAS_NLP) / "spider"
        fname = "train_spider.json" if split == "train" else "dev.json"
        with open(p / fname) as f: data = _j.load(f)
        self.src = [f"translate to SQL: {r['question']} | {r.get('db_id','')}" for r in data]
        self.tgt = [r["query"] for r in data]
        self.tok = tok; self.ms = max_src; self.mt = max_tgt

    def __len__(self): return len(self.src)

    def __getitem__(self, i):
        s = self.tok(self.src[i], max_length=self.ms, truncation=True,
                     return_tensors="pt", padding="max_length")
        t = self.tok(self.tgt[i], max_length=self.mt, truncation=True,
                     return_tensors="pt", padding="max_length")
        lbl = t.input_ids.squeeze(0).clone()
        lbl[lbl == self.tok.pad_token_id] = -100
        return {"input_ids": s.input_ids.squeeze(0),
                "attention_mask": s.attention_mask.squeeze(0), "labels": lbl}


def inject_rel_attn(model, k=8):
    """Replace T5Attention with our MultiRelationAttention."""
    import sys; sys.path.insert(0, "/home/muhiddin/relational-attention")
    from relational_attention.attention import MultiRelationAttention
    from transformers.models.t5.modeling_t5 import T5Attention

    class RelAttnT5(nn.Module):
        def __init__(self, orig, d, h, k):
            super().__init__()
            self.mra = MultiRelationAttention(d, h, k, dropout=0.1)
        def forward(self, hidden_states, mask=None, key_value_states=None,
                    position_bias=None, past_key_value=None, layer_head_mask=None,
                    query_length=None, use_cache=False, output_attentions=False):
            ctx = key_value_states if key_value_states is not None else hidden_states
            attn_mask = mask[:, 0, :, :] if (mask is not None and
                                              mask.dim()==4) else mask
            out, _ = self.mra(hidden_states, ctx, ctx, attn_mask)
            # T5 expects: (hidden_states, present_key_value_state, position_bias)
            # Must be 3-element tuple: T5LayerSelfAttention accesses [0] and [2]
            return (out, None, None)

    d = model.config.d_model; h = model.config.num_heads
    for name, mod in list(model.named_modules()):
        if isinstance(mod, T5Attention):
            parent_name, child = name.rsplit(".", 1)
            parent = dict(model.named_modules())[parent_name]
            setattr(parent, child, RelAttnT5(mod, d, h, k))
    return model


def run(tag, use_rel_attn, device, epochs=25, lr=1e-4, batch=16, use_fp16=False):
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    out = Path(NAS_OUT) / tag
    out.mkdir(parents=True, exist_ok=True)
    done_path = out / "epoch_curves.json"
    if done_path.exists():
        log.info(f"SKIP {tag} (already done)")
        return

    tok = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    if use_rel_attn:
        model = inject_rel_attn(model, k=8)
        log.info("RelAttn injected into T5")
    model.to(device)
    if use_rel_attn:
        # Gradient checkpointing reduces memory at cost of ~33% more compute
        try:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
            log.info("Gradient checkpointing enabled for T5+RelAttn")
        except Exception as e:
            log.warning(f"Could not enable grad checkpointing: {e}")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"{tag}: {n/1e6:.1f}M params")

    train_dl = DataLoader(SpiderDS(tok,"train"), batch_size=batch, shuffle=True, num_workers=2)
    dev_dl   = DataLoader(SpiderDS(tok,"dev"),   batch_size=batch*2, shuffle=False, num_workers=2)
    if use_rel_attn:
        from transformers.optimization import Adafactor
        opt = Adafactor(model.parameters(), lr=lr, scale_parameter=False,
                        relative_step=False, warmup_init=False, weight_decay=0.01)
        log.info("Adafactor optimizer (memory-efficient, ~4x less than AdamW)")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    records = []

    for ep in range(1, epochs+1):
        model.train(); tl, tk = 0.0, 0
        for b in train_dl:
            b = {k:v.to(device) for k,v in b.items()}
            loss = model(**b).loss
            if torch.isnan(loss) or torch.isinf(loss): continue
            if use_fp16 and not use_rel_attn:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            opt.zero_grad()
            nt = (b["labels"]!=-100).sum().item()
            tl += loss.item()*nt; tk += nt
        tl /= max(1,tk)

        model.eval(); el, ek = 0.0, 0
        with torch.no_grad():
            for b in dev_dl:
                b = {k:v.to(device) for k,v in b.items()}
                nt = (b["labels"]!=-100).sum().item()
                el += model(**b).loss.item()*nt; ek += nt
        el /= max(1,ek)

        row = {"epoch":ep,"train_loss":round(tl,4),"dev_loss":round(el,4)}
        records.append(row)
        log.info(f"[{tag}] ep={ep} train={tl:.4f} dev={el:.4f}")

    with open(done_path,"w") as f:
        json.dump({"tag":tag,"use_rel_attn":use_rel_attn,"curves":records},f,indent=2)
    log.info(f"Saved {done_path}")


if __name__ == "__main__":
    import json
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["base","relattr","both"], default="both")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8 if args.mode == "relattr" else 16  # RelAttn larger model
    if args.mode in ("base","both"):   run("t5_base",    False, dev, args.epochs, batch=batch_size)
    if args.mode in ("relattr","both"): run("t5_relattr", True,  dev, args.epochs, batch=4, use_fp16=True)

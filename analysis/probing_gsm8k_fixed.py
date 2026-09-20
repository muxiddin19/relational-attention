#!/usr/bin/env python3
"""
Corrected Fisher Discriminability Probing for GSM8K (methodology fix).

This is a bug-fixed version of analysis/probing.py's real compute path.
Two bugs are fixed relative to the original:

  1. Dataset: original hardcoded `load_dataset_examples("spider", "dev", ...)`
     even though the paper's Table 4 / Sec. "Mechanistic Analysis" claims
     probing is done "on the GSM8K-trained checkpoint (GSM8K math tokens)".
     This script loads GSM8K (test split, which train.py treats as the
     GSM8K "dev" split since GSM8K ships no dedicated dev file) instead.

  2. Role labeling: original divided each example's token sequence into
     three equal positional thirds ("first third = Identity, middle third
     = FD, last third = Schema") regardless of actual token content. This
     is not a content-based labeling scheme at all. This script instead
     does real lexical/content classification of the GSM8K source text:

       Identity  = numeric-quantity tokens (regex \\d+(\\.\\d+)?, i.e. any
                   number appearing in the word problem, PLUS a fixed list
                   of spelled-out cardinal-number words -- "three", "a
                   dozen", etc., since GSM8K frequently spells out small
                   quantities) and proper-noun tokens (capitalized word,
                   not sentence-initial -- a crude but real named-entity/
                   quantity-referent heuristic, e.g. person names "Janet",
                   "Betty" that quantities attach to; sentence-initial
                   capitalized words are excluded from this rule since
                   capitalization there is a sentence-position artifact,
                   not evidence of being a proper noun -- one documented
                   side-effect is that the person name at the very start
                   of a problem, when it also opens the sentence, is
                   labeled Schema rather than Identity).
       FD        = a fixed, hand-curated list of relational/arithmetic
                   predicate words (see FD_WORDS below) -- words expressing
                   a relationship between quantities (comparison, transfer,
                   rate, aggregation).
       Schema    = everything else (articles, generic nouns, punctuation,
                   connectives, question words, sentence scaffolding) --
                   the default/residual category.

     Word-level labels are computed on the raw source string, then mapped
     onto SentencePiece subword tokens via a greedy character-offset
     alignment (SentencePiece's Python API does not return offsets
     directly, so each piece's de-escaped text, "_"->" ", is located in
     the source string starting from the previous piece's end). A token
     inherits the label of the source word span it falls inside; special
     tokens (EOS/BOS/PAD) get Schema by convention and are otherwise
     excluded from analysis by the attention_mask exactly as before.

Everything else (extract_representations' slot-splitting logic and the
Fisher F formula itself, compute_fisher_f) is reused unchanged from the
original probing.py, per instructions -- the audit found the bug is in
what data/labels are fed in, not in the Fisher computation.

Usage:
    python probing_gsm8k_fixed.py \
        --checkpoint /nas/.../relational_125m_gsm8k_s42_rerun/best_model \
        --data-dir /nas/Dataset/nlp \
        --n-examples 300 \
        --output probing_results_gsm8k_relational.json
"""
import argparse
import json
import re
import sys
from pathlib import Path
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/muhiddin/relational-attention")
sys.path.insert(0, "/home/muhiddin/relational-attention/scripts")

ROLES = ["Identity", "FD", "Schema"]
ROLE_ID = {"Identity": 0, "FD": 1, "Schema": 2}

# ── Curated relational/arithmetic-predicate word list (FD role) ─────────────
# Words expressing a relationship *between* quantities: comparison, ratio,
# transfer/possession-change, rate, and aggregation predicates. Kept small,
# fixed in advance, and not tuned against results.
FD_WORDS = {
    "more", "less", "fewer", "greater", "smaller", "twice", "half", "double",
    "triple", "times", "each", "every", "per", "total", "sum", "combined",
    "altogether", "difference", "gives", "gave", "give", "given", "receives",
    "received", "receive", "spent", "spend", "spends", "left", "remain",
    "remains", "remaining", "than", "increase", "increased", "decrease",
    "decreased", "divided", "divide", "multiply", "multiplied", "percent",
    "percentage", "ratio", "rate", "average", "cost", "costs", "price",
    "discount", "profit", "loss", "extra", "additional", "plus", "minus",
    "sold", "sells", "sell", "bought", "buys", "buy", "add", "added",
    "subtract", "subtracted", "shared", "share", "split", "equal", "equally",
}

# Spelled-out cardinal-number words (GSM8K frequently spells out small
# quantities, e.g. "three", "a dozen"). These are quantity-variable tokens
# just like digit-form numbers, so they belong in Identity, not Schema.
NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "dozen", "couple",
}

NUM_RE = re.compile(r"^\d+(\.\d+)?$")
WORD_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?")


def _sentence_initial_mask(text: str):
    """True at the index of the first word-char of each sentence."""
    mask = np.zeros(len(text), dtype=bool)
    at_start = True
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if at_start and (c.isalnum()):
            mask[i] = True
            at_start = False
        if c in ".?!":
            at_start = True
        elif not c.isspace() and c not in ".?!":
            pass
        i += 1
    return mask


def label_source_text_v2(text: str):
    """Corrected single-pass labeler: numbers -> Identity, FD-list -> FD,
    sentence-initial-aware proper nouns -> Identity, else Schema."""
    char_labels = np.full(len(text), ROLE_ID["Schema"], dtype=np.int64)
    sent_initial_idx = _sentence_initial_mask(text)
    for m in WORD_RE.finditer(text):
        w = m.group(0)
        start, end = m.start(), m.end()
        if NUM_RE.match(w) or w.lower() in NUMBER_WORDS:
            role = ROLE_ID["Identity"]
        elif w.lower() in FD_WORDS:
            role = ROLE_ID["FD"]
        elif w[0].isupper() and not sent_initial_idx[start]:
            role = ROLE_ID["Identity"]
        else:
            role = ROLE_ID["Schema"]
        char_labels[start:end] = role
    return char_labels


def align_pieces_to_labels(sp, text: str, piece_ids, char_labels):
    """Greedy char-offset alignment of SentencePiece pieces to `char_labels`.
    Returns an int array of per-token role labels, same length as piece_ids
    (which includes the trailing </s> if present, labeled Schema).
    """
    pieces = [sp.IdToPiece(i) for i in piece_ids]
    labels = np.full(len(pieces), ROLE_ID["Schema"], dtype=np.int64)
    cursor = 0
    n = len(text)
    for idx, piece in enumerate(pieces):
        if piece in ("<s>", "</s>", "<pad>", "<unk>"):
            continue
        piece_text = piece.replace("▁", " ")  # SentencePiece '▁'
        search = piece_text.strip(" ")
        if search == "":
            continue
        pos = text.find(search, cursor)
        if pos == -1:
            pos = text.find(search)  # fallback: search from start
            if pos == -1:
                continue  # unmappable (rare unicode/normalization edge case)
        start, end = pos, min(pos + len(search), n)
        if end > start:
            seg = char_labels[start:end]
            vals, counts = np.unique(seg, return_counts=True)
            labels[idx] = vals[np.argmax(counts)]
            cursor = end
        else:
            cursor = pos
    return labels


def compute_fisher_f(embeddings: np.ndarray, labels: np.ndarray, role_label: int) -> float:
    """Fisher F for one slot against one binary role label. Unchanged from
    the original probing.py -- reused as-is."""
    binary = (labels == role_label).astype(int)
    grand_mean = embeddings.mean(axis=0)
    n = len(embeddings)
    sigma_b, sigma_w = 0.0, 0.0
    for c in [0, 1]:
        mask = binary == c
        if not mask.any():
            continue
        cm = embeddings[mask].mean(axis=0)
        sigma_b += mask.sum() * np.sum((cm - grand_mean) ** 2)
        sigma_w += np.sum((embeddings[mask] - cm) ** 2)
    sigma_b /= n * embeddings.shape[1]
    sigma_w /= n * embeddings.shape[1]
    return float(sigma_b / max(sigma_w, 1e-8))


def extract_representations(model, data_loader, device, k: int, sp, raw_texts):
    """Extract per-slot embeddings and CONTENT-BASED role labels from GSM8K
    source text. Slot-splitting logic reused unchanged from original
    probing.py; only the label source changed (was: divide-into-thirds;
    now: label_source_text_v2 + align_pieces_to_labels)."""
    model.eval()
    slot_embs = [[] for _ in range(k)]
    all_labels = []
    text_iter = iter(raw_texts)

    with torch.no_grad():
        for batch in data_loader:
            batch_texts = [next(text_iter) for _ in range(batch["input_ids"].shape[0])]
            gpu_batch = {key: val.to(device)
                         for key, val in batch.items() if isinstance(val, torch.Tensor)}
            enc_out, _ = model.encoder(
                input_ids=gpu_batch["input_ids"],
                attention_mask=gpu_batch.get("attention_mask"),
                return_attention=False,
            )
            B, S, D = enc_out.shape
            attr_dim = D // k
            attrs = enc_out.view(B, S, k, attr_dim).cpu().numpy()

            for b in range(B):
                if "attention_mask" in batch:
                    valid = batch["attention_mask"][b].cpu().numpy().astype(bool)
                else:
                    valid = np.ones(S, dtype=bool)
                T = int(valid.sum())
                ids = batch["input_ids"][b].cpu().numpy()[:S]

                text = batch_texts[b]
                char_labels = label_source_text_v2(text)
                tok_labels_full = align_pieces_to_labels(sp, text, ids.tolist(), char_labels)
                roles = tok_labels_full[valid]

                for j in range(k):
                    slot_embs[j].append(attrs[b, valid, j, :])
                all_labels.append(roles)

    return [np.vstack(e) for e in slot_embs], np.concatenate(all_labels)


def print_table(fisher_f: list, roles: list):
    header = f"{'Slot':<8}" + "".join(f"{r:>12}" for r in roles)
    print(header)
    print("-" * len(header))
    for j, row in enumerate(fisher_f):
        cells = "".join(f"{v:>12.4f}" for v in row)
        print(f"Slot {j+1:<4}{cells}")


def permutation_control(slot_embs, labels, n_perm=20, seed=0):
    """Shuffle-label null control: recompute Fisher F with role labels
    randomly permuted, n_perm times, per slot per role. Returns mean/std."""
    rng = np.random.RandomState(seed)
    k = len(slot_embs)
    means = np.zeros((k, len(ROLES)))
    stds = np.zeros((k, len(ROLES)))
    for j in range(k):
        vals = np.zeros((n_perm, len(ROLES)))
        for p in range(n_perm):
            shuffled = labels.copy()
            rng.shuffle(shuffled)
            for r in range(len(ROLES)):
                vals[p, r] = compute_fisher_f(slot_embs[j], shuffled, r)
        means[j] = vals.mean(axis=0)
        stds[j] = vals.std(axis=0)
    return means, stds


def build_model_from_cfg(cfg, device, load_state=True, ckpt_path=None):
    from relational_attention import RelationalTransformer, RelationalTransformerConfig
    k = cfg["num_attributes"]
    model_cfg = RelationalTransformerConfig(
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
    )
    model = RelationalTransformer(model_cfg)
    if load_state:
        sd = torch.load(Path(ckpt_path) / "model.pt", map_location=device)
        model.load_state_dict(sd)
    model.eval().to(device)
    return model, k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="/nas/Dataset/nlp")
    p.add_argument("--output", default="probing_gsm8k_results.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-examples", type=int, default=300)
    p.add_argument("--random-init", action="store_true",
                   help="Ignore --checkpoint weights; use freshly-initialized "
                        "weights with the checkpoint's architecture config.")
    p.add_argument("--n-perm", type=int, default=20)
    args = p.parse_args()

    import yaml
    from train import load_dataset_examples, Seq2SeqDataset, collate_fn, SPTokenizer

    ckpt = Path(args.checkpoint)
    cfg = yaml.safe_load(open(ckpt / "config.yaml"))

    model, k = build_model_from_cfg(cfg, args.device, load_state=not args.random_init,
                                     ckpt_path=ckpt)
    tag = "RANDOM-INIT" if args.random_init else str(ckpt)
    print(f"Model ready ({tag}), k={k}, "
          f"{sum(pp.numel() for pp in model.parameters())/1e6:.1f}M params")

    tokenizer = SPTokenizer(str(Path(args.data_dir) / "tokenizer" / "sp32k.model"))
    examples = load_dataset_examples("gsm8k", "dev", args.data_dir)
    examples = examples[:args.n_examples]
    raw_texts = [e["source"] for e in examples]

    ds = Seq2SeqDataset(examples, tokenizer,
                         cfg.get("max_src_len", 256), cfg.get("max_tgt_len", 64))
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0,
                     collate_fn=partial(collate_fn, pad_id=tokenizer.pad_id))

    print(f"Probing {len(examples)} GSM8K dev(test-split) examples...")
    slot_embs, labels = extract_representations(model, dl, args.device, k,
                                                  tokenizer.sp, raw_texts)

    label_counts = {ROLES[r]: int((labels == r).sum()) for r in range(len(ROLES))}
    print(f"Token role counts: {label_counts} (total {len(labels)})")

    fisher_f = []
    for j in range(k):
        row = [round(compute_fisher_f(slot_embs[j], labels, r_idx), 4)
               for r_idx in range(len(ROLES))]
        fisher_f.append(row)

    print("\nFisher Discriminability (computed, content-based GSM8K labels):")
    print_table(fisher_f, ROLES)

    print(f"\nPermutation control ({args.n_perm} shuffles)...")
    perm_mean, perm_std = permutation_control(slot_embs, labels, n_perm=args.n_perm)
    print("Null (shuffled-label) Fisher F, mean ± std:")
    for j in range(k):
        cells = ", ".join(f"{ROLES[r]}={perm_mean[j,r]:.4f}±{perm_std[j,r]:.4f}"
                           for r in range(len(ROLES)))
        print(f"  Slot {j+1}: {cells}")

    result = {
        "source": "computed_gsm8k_fixed",
        "checkpoint": tag,
        "config": {kk: cfg.get(kk) for kk in
                   ["dataset", "model_type", "fp16", "num_attributes",
                    "hidden_dim", "num_encoder_layers", "num_heads"]},
        "n_examples": len(examples),
        "token_role_counts": label_counts,
        "fisher_f": fisher_f,
        "fisher_f_null_mean": perm_mean.tolist(),
        "fisher_f_null_std": perm_std.tolist(),
        "roles": ROLES,
        "k": k,
        "fd_word_list": sorted(FD_WORDS),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.output, "w"), indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

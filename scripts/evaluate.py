"""
Evaluation script for all benchmarks.

Spider: uses the official test-suite-sql-eval (EX + EM).
COGS/SCAN/CFQ/GSM8K: exact match accuracy.

Usage:
    python scripts/evaluate.py \\
        --checkpoint ./outputs/rel_125m_spider_s42/best_model \\
        --dataset spider \\
        --nas-dir /nas/Dataset \\
        --split dev
"""

import argparse
import gc
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Dict

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from relational_attention import RelationalTransformer, RelationalTransformerConfig

log = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                    level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_dir: str, device: torch.device) -> tuple:
    ckpt = Path(checkpoint_dir)
    cfg_path = ckpt / "config.yaml"
    if not cfg_path.exists():
        # Fallback: look in sibling best_model dir (for periodic checkpoints)
        fallback = ckpt.parent / "best_model" / "config.yaml"
        if fallback.exists():
            cfg_path = fallback
        else:
            raise FileNotFoundError(f"config.yaml not found in {ckpt} or {fallback}")
    with open(cfg_path) as f:
        cfg_dict = yaml.safe_load(f)

    k = cfg_dict.get("num_attributes", 8)
    model_type = cfg_dict.get("model_type", "relational")
    if model_type == "standard":
        k = 1

    model_cfg = RelationalTransformerConfig(
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
    )
    model = RelationalTransformer(model_cfg)
    state = torch.load(ckpt / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    nas_dir = cfg_dict.get("nas_dir", "/nas/Dataset")
    tok_path = Path(nas_dir) / "tokenizer" / "sp32k.model"
    sys.path.insert(0, str(Path(__file__).parent))
    from train import SPTokenizer
    tokenizer = SPTokenizer(str(tok_path))
    return model, tokenizer, cfg_dict


# ---------------------------------------------------------------------------
# Generation (greedy / beam search)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, tokenizer, sources: List[str], device: torch.device,
             max_tgt_len: int = 128, beam_size: int = 4,
             batch_size: int = 32) -> List[str]:
    import torch.nn.functional as F
    results = []
    for i in range(0, len(sources), batch_size):
        batch_src = sources[i: i + batch_size]
        src_ids = [tokenizer.encode(s, add_bos=False, add_eos=True) for s in batch_src]
        max_len = max(len(x) for x in src_ids)
        padded = torch.zeros(len(batch_src), max_len, dtype=torch.long)
        mask   = torch.zeros(len(batch_src), max_len, dtype=torch.long)
        for j, ids in enumerate(src_ids):
            padded[j, :len(ids)] = torch.tensor(ids)
            mask[j, :len(ids)] = 1
        padded, mask = padded.to(device), mask.to(device)

        # Greedy decoding
        dec_ids = torch.full((len(batch_src), 1), tokenizer.bos_id,
                             dtype=torch.long, device=device)
        done = torch.zeros(len(batch_src), dtype=torch.bool, device=device)
        for _ in range(max_tgt_len):
            out = model(input_ids=padded, attention_mask=mask,
                        decoder_input_ids=dec_ids)
            logits = out["logits"] if isinstance(out, dict) else out[0]
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            dec_ids = torch.cat([dec_ids, next_tok], dim=1)
            done |= (next_tok.squeeze(-1) == tokenizer.eos_id)
            if done.all():
                break

        for row in dec_ids.cpu().tolist():
            results.append(tokenizer.decode(row))

    return results


# ---------------------------------------------------------------------------
# Spider evaluation (official test-suite-sql-eval)
# ---------------------------------------------------------------------------

def eval_spider(predictions: List[str], examples: List[Dict],
                nas_dir: str, split: str) -> Dict:
    spider_dir = Path(nas_dir) / "spider"
    eval_script = spider_dir / "test-suite-sql-eval" / "evaluation.py"
    db_dir = spider_dir / "database"

    if not eval_script.exists() or not db_dir.exists():
        reason = "eval script" if not eval_script.exists() else "database dir"
        log.warning(f"Spider {reason} not found; falling back to exact-match.")
        preds = [normalize(p) for p in predictions]
        golds = [normalize(e["target"]) for e in examples]
        em = sum(p == g for p, g in zip(preds, golds)) / len(golds)
        # Token-level F1 as secondary metric
        def token_f1(pred, gold):
            p_toks, g_toks = set(pred.split()), set(gold.split())
            if not p_toks or not g_toks:
                return 0.0
            common = p_toks & g_toks
            if not common:
                return 0.0
            prec = len(common) / len(p_toks)
            rec  = len(common) / len(g_toks)
            return 2 * prec * rec / (prec + rec)
        f1 = sum(token_f1(p, g) for p, g in zip(preds, golds)) / len(golds)
        return {"exact_match": em, "token_f1": f1, "execution_accuracy": None}

    gold_file = spider_dir / ("train_spider.json" if split == "train" else "dev.json")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False) as pred_f:
        for p in predictions:
            pred_f.write(p.strip() + "\n")
        pred_path = pred_f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False) as gold_f:
        with open(gold_file) as f:
            data = json.load(f)
        for row in data:
            gold_f.write(row["query"].strip() + "\t" + row.get("db_id", "") + "\n")
        gold_path = gold_f.name

    cmd = [
        sys.executable, str(eval_script),
        "--gold", gold_path,
        "--pred", pred_path,
        "--db",  str(spider_dir / "database"),
        "--table", str(spider_dir / "tables.json"),
        "--etype", "exec",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        output = result.stdout + result.stderr
        log.info("Spider eval output:\n" + output)
        # Parse EX from output
        # test-suite eval prints: "execution  0.750  0.680  ..." (decimal, not %)
        ex = None
        for line in output.splitlines():
            line_low = line.lower()
            if "execution" in line_low and "accuracy" not in line_low:
                parts = line.split()
                for p in reversed(parts):
                    try:
                        val = float(p.replace("%", ""))
                        if 0.0 <= val <= 1.0:
                            ex = val
                        elif 0.0 < val <= 100.0:
                            ex = val / 100.0
                        break
                    except ValueError:
                        continue
        return {"execution_accuracy": ex, "eval_output": output}
    except Exception as e:
        log.error(f"Spider eval failed: {e}")
        return {"execution_accuracy": None}


# ---------------------------------------------------------------------------
# Generic exact-match evaluation
# ---------------------------------------------------------------------------

def eval_exact_match(predictions: List[str], examples: List[Dict], dataset: str = "") -> Dict:
    preds = [p.strip() for p in predictions]
    golds = [e["target"].strip() for e in examples]
    # SCAN: gold targets in examples may contain the encoded form (I __ WALK);
    # normalise back to canonical I_WALK form for fair comparison
    if dataset == "scan":
        def _undo_scan_enc(s):
            toks = s.split(); result = []; i = 0
            while i < len(toks):
                if i + 2 < len(toks) and toks[i+1] == "__":
                    result.append(toks[i] + "_" + toks[i+2]); i += 3
                else:
                    result.append(toks[i]); i += 1
            return " ".join(result)
        golds = [_undo_scan_enc(g) for g in golds]
    correct = sum(normalize(p) == normalize(g) for p, g in zip(preds, golds))
    return {"exact_match": correct / len(golds), "n": len(golds)}


def normalize(s: str) -> str:
    s = s.replace("⁇", ";")  # SentencePiece sp32k maps ASCII ; to U+2047
    return " ".join(s.lower().split())


# ---------------------------------------------------------------------------
# GSM8K: extract final numeric answer
# ---------------------------------------------------------------------------

def eval_gsm8k(predictions: List[str], examples: List[Dict]) -> Dict:
    import re
    def extract_answer(text: str) -> Optional[str]:
        # GSM8K final answer follows "#### <number>"
        m = re.search(r"####\s*([\d,\-\.]+)", text)
        return m.group(1).replace(",", "") if m else None

    correct = 0
    for pred, ex in zip(predictions, examples):
        # Handle both "#### N" (CoT format) and "N" (direct answer format)
        gold_ans = extract_answer(ex["target"]) or ex["target"].strip().split()[-1]
        pred_ans = extract_answer(pred) or pred.strip().split()[-1]
        if gold_ans and pred_ans:
            try:
                correct += abs(float(pred_ans) - float(gold_ans)) < 1e-3
            except ValueError:
                correct += normalize(pred_ans) == normalize(gold_ans)
    return {"answer_accuracy": correct / len(examples), "n": len(examples)}


from typing import Optional


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True,
                   choices=["spider", "cogs", "scan", "cfq", "gsm8k"])
    p.add_argument("--nas-dir", default="/nas/Dataset")
    p.add_argument("--split", default="dev")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-tgt-len", type=int, default=128)
    p.add_argument("--output-file", help="Save predictions to JSON")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, cfg = load_model(args.checkpoint, device)

    # Load examples
    sys.path.insert(0, str(Path(__file__).parent))
    from train import load_dataset_examples
    examples = load_dataset_examples(args.dataset, args.split, args.nas_dir)
    sources = [e["source"] for e in examples]
    log.info(f"Evaluating {len(examples)} examples from {args.dataset}/{args.split}")

    # Generate predictions
    predictions = generate(model, tokenizer, sources, device,
                           max_tgt_len=args.max_tgt_len,
                           batch_size=args.batch_size)

    # SCAN post-process: reverse the space-separated encoding applied in train.py
    # "I __ WALK I __ RUN" → "I_WALK I_RUN"
    if args.dataset == "scan":
        def scan_decode(s):
            return " ".join(tok.replace(" __ ", "_").replace("__", "_")
                           for tok in s.replace(" __ ", "__").split())
        def scan_decode_v2(s):
            # Handle "I __ WALK" → "I_WALK" pattern
            import re
            s = re.sub(r"I\s+__\s+(\w+)", r"I_", s)
            return s
        # Also fix the gold targets for eval_exact_match
        from train import load_dataset_examples as _lde
        # Simpler: just un-encode the space-split pattern
        def _scan_postproc(pred):
            # Replace "I __ WALK" → "I_WALK", "TURN __ LEFT" → "TURN_LEFT" etc.
            toks = pred.split()
            result = []
            i = 0
            while i < len(toks):
                if i + 2 < len(toks) and toks[i+1] == "__":
                    result.append(toks[i] + "_" + toks[i+2])
                    i += 3
                elif i + 1 < len(toks) and toks[i] == "__":
                    # orphaned __, skip
                    i += 1
                else:
                    result.append(toks[i])
                    i += 1
            return " ".join(result)
        predictions = [_scan_postproc(p) for p in predictions]

    # Evaluate
    if args.dataset == "spider":
        metrics = eval_spider(predictions, examples, args.nas_dir, args.split)
    elif args.dataset == "gsm8k":
        metrics = eval_gsm8k(predictions, examples)
    else:
        metrics = eval_exact_match(predictions, examples, dataset=args.dataset)

    log.info(f"=== Results ({args.dataset}/{args.split}) ===")
    for k, v in metrics.items():
        if v is not None and k != "eval_output":
            log.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.output_file:
        out = {"metrics": metrics, "predictions": predictions[:100]}
        with open(args.output_file, "w") as f:
            json.dump(out, f, indent=2)
        log.info(f"Saved to {args.output_file}")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return metrics


if __name__ == "__main__":
    main()

"""
Fix a crash in evaluate.py's eval_gsm8k(): pred.strip().split()[-1] raises
IndexError when the model generates an empty (or whitespace-only) string,
which happens for some seeds/checkpoints. Fall back to "" instead of
crashing -- an empty prediction is correctly scored as wrong, not a fatal
error.

Run on the remote host: python3 patch_eval_empty_pred.py
"""

PATH = "scripts/evaluate.py"

with open(PATH, encoding="utf-8") as f:
    src = f.read()

old = '''        # Handle both "#### N" (CoT format) and "N" (direct answer format)
        gold_ans = extract_answer(ex["target"]) or ex["target"].strip().split()[-1]
        pred_ans = extract_answer(pred) or pred.strip().split()[-1]'''
new = '''        # Handle both "#### N" (CoT format) and "N" (direct answer format).
        # Fall back to "" (not IndexError) when the string is empty --
        # an empty generation is scored as wrong, not a crash.
        gold_toks = ex["target"].strip().split()
        pred_toks = pred.strip().split()
        gold_ans = extract_answer(ex["target"]) or (gold_toks[-1] if gold_toks else "")
        pred_ans = extract_answer(pred) or (pred_toks[-1] if pred_toks else "")'''
assert old in src, "eval_gsm8k fallback lines not found verbatim"
src = src.replace(old, new, 1)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)
print("Patched", PATH, "successfully.")

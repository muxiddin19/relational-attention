#!/usr/bin/env bash
# Master experiment runner.
# Trains all model/dataset/seed combinations used in the paper, then evaluates.
#
# Usage:
#   NAS_DIR=/nas/Dataset bash scripts/run_experiments.sh
#   NAS_DIR=/nas/Dataset GPU=0,1 bash scripts/run_experiments.sh
#
# Outputs go to ./outputs/<model>_<dataset>_s<seed>/
# Summary table is written to ./outputs/results_summary.csv

set -e

NAS_DIR="${NAS_DIR:-/nas/Dataset/nlp}"
GPU="${GPU:-0}"
SEEDS=(42 43 44)
DATASETS=(spider cogs scan cfq gsm8k)
MODELS=(relational standard)   # standard = baseline with identical architecture
SIZES=(125m 350m)
OUT_ROOT="./outputs"

# Use conda env python if available
PYTHON="${PYTHON:-python3}"
for p in \
    "$HOME/anaconda3/envs/relattn/bin/python" \
    "$HOME/miniconda3/envs/relattn/bin/python" \
    "$(which python3)"; do
    [ -x "$p" ] && { PYTHON="$p"; break; }
done
echo "Using Python: $PYTHON"
SUMMARY="$OUT_ROOT/results_summary.csv"

export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$OUT_ROOT"
echo "model,size,dataset,seed,metric,value" > "$SUMMARY"

run_one() {
    local MODEL="$1"   # relational | standard
    local SIZE="$2"    # 125m | 350m
    local DATASET="$3"
    local SEED="$4"

    local CONFIG="configs/${MODEL}_transformer_${SIZE}.yaml"
    local OUT_DIR="$OUT_ROOT/${MODEL}_${SIZE}_${DATASET}_s${SEED}"

    if [ -f "$OUT_DIR/best_model/model.pt" ]; then
        echo "=== SKIP (already trained): $OUT_DIR ==="
        return 0
    fi

    echo ""
    echo "=========================================="
    echo " Training: model=$MODEL size=$SIZE dataset=$DATASET seed=$SEED"
    echo "=========================================="

    $PYTHON scripts/train.py \
        --config "$CONFIG" \
        --dataset "$DATASET" \
        --nas-dir "$NAS_DIR" \
        --seed "$SEED" \
        --output-dir "$OUT_DIR" \
        --fp16

    echo "--- Evaluating: $OUT_DIR ---"
    $PYTHON scripts/evaluate.py \
        --checkpoint "$OUT_DIR/best_model" \
        --dataset "$DATASET" \
        --nas-dir "$NAS_DIR" \
        --output-file "$OUT_DIR/eval_results.json" \
        2>&1 | tee "$OUT_DIR/eval.log"

    # Append to summary CSV
    METRIC=$($PYTHON -c "
import json, sys
d = json.load(open('$OUT_DIR/eval_results.json'))
m = d.get('metrics', {})
for k, v in m.items():
    if v is not None and isinstance(v, float):
        print(k, v)
        break
" 2>/dev/null | head -1)
    if [ -n "$METRIC" ]; then
        echo "$MODEL,$SIZE,$DATASET,$SEED,$METRIC" >> "$SUMMARY"
    fi
}

# ---- Main loop ----
for SIZE in "${SIZES[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            for MODEL in "${MODELS[@]}"; do
                run_one "$MODEL" "$SIZE" "$DATASET" "$SEED"
            done
        done
    done
done

echo ""
echo "=========================================="
echo " All experiments done."
echo " Summary: $SUMMARY"
echo "=========================================="
cat "$SUMMARY"

# ---- Aggregate results (mean ± std over seeds) ----
$PYTHON - <<'EOF'
import csv, collections, statistics, sys

rows = list(csv.DictReader(open("outputs/results_summary.csv")))
groups = collections.defaultdict(list)
for r in rows:
    key = (r["model"], r["size"], r["dataset"], r["metric"])
    try:
        groups[key].append(float(r["value"]))
    except ValueError:
        pass

print("\n=== Aggregated Results (mean ± std, 3 seeds) ===")
print(f"{'model':<12} {'size':<6} {'dataset':<8} {'metric':<25} {'mean':>8} {'std':>8}")
print("-" * 75)
for (model, size, dataset, metric), vals in sorted(groups.items()):
    mean = statistics.mean(vals)
    std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"{model:<12} {size:<6} {dataset:<8} {metric:<25} {mean:>8.4f} {std:>8.4f}")
EOF

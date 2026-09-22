#!/bin/bash
# Genuine standard multi-head attention baseline for Spider, copy-mechanism
# protocol, same width/depth as RelTransformer and identical hyperparameters
# to the invalid std_transformer_spider_copy.yaml (k=1) config it replaces.
GPU=${1:?usage: run_true_std_spider.sh <gpu_id> <seed1> [seed2 ...]}
shift
PYTHON=/home/muhiddin/miniconda3/envs/har/bin/python
TRAIN=/home/muhiddin/relational-attention/scripts/train.py
EVAL=/home/muhiddin/relational-attention/scripts/evaluate.py
OUTBASE=/nas/Dataset/experiments/relational-attention/muhiddin_outputs
CFGS=/home/muhiddin/relational-attention/configs
LOGDIR=/home/muhiddin/relational-attention/results/verified_2026_revision/logs
EVALDIR=/home/muhiddin/relational-attention/results/verified_2026_revision
CHAINLOG=$LOGDIR/gpu${GPU}_true_std_spider_chain.log

mkdir -p "$LOGDIR"
echo "[GPU${GPU} TRUE-STD-SPIDER CHAIN START] $(date)" >> "$CHAINLOG"

for seed in "$@"; do
  OUTDIR="$OUTBASE/true_std_spider_copy_s${seed}"
  echo "[TRAIN START true_std_spider_copy s${seed}] $(date)" >> "$CHAINLOG"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON $TRAIN \
    --config "$CFGS/true_std_512_spider_copy.yaml" --seed $seed \
    --dataset spider --nas-dir /nas/Dataset/nlp --output-dir "$OUTDIR" \
    --batch-size 4 \
    >> "$LOGDIR/gpu${GPU}_true_std_spider_copy_s${seed}.log" 2>&1
  rc=$?
  echo "[TRAIN DONE true_std_spider_copy s${seed} rc=${rc}] $(date)" >> "$CHAINLOG"
  if [ $rc -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON $EVAL \
      --checkpoint "$OUTDIR/best_model" --dataset spider \
      --nas-dir /nas/Dataset/nlp --split dev --batch-size 4 \
      --output-file "$EVALDIR/eval_true_std_spider_copy_s${seed}.json" \
      >> "$LOGDIR/gpu${GPU}_eval_true_std_spider_copy_s${seed}.log" 2>&1
    echo "[EVAL DONE true_std_spider_copy s${seed} rc=$?] $(date)" >> "$CHAINLOG"
  fi
done

echo "[GPU${GPU} TRUE-STD-SPIDER CHAIN DONE] $(date)" >> "$CHAINLOG"

#!/bin/bash
# Genuine standard multi-head attention baseline chain: GSM8K (8 seeds) then
# COGS-with-copy (5 seeds). TAG identifies which variant (512 = same-width
# 121.8M; matched = FFN-widened 464.5M parameter-matched to RelTransformer).
GPU=${1:?usage: run_true_std_chain.sh <gpu_id> <tag: 512|matched>}
TAG=${2:?usage: run_true_std_chain.sh <gpu_id> <tag: 512|matched>}
PYTHON=/home/muhiddin/miniconda3/envs/har/bin/python
TRAIN=/home/muhiddin/relational-attention/scripts/train.py
EVAL=/home/muhiddin/relational-attention/scripts/evaluate.py
OUTBASE=/nas/Dataset/experiments/relational-attention/muhiddin_outputs
CFGS=/home/muhiddin/relational-attention/configs
LOGDIR=/home/muhiddin/relational-attention/results/verified_2026_revision/logs
EVALDIR=/home/muhiddin/relational-attention/results/verified_2026_revision
CHAINLOG=$LOGDIR/gpu${GPU}_true_std_${TAG}_chain.log

mkdir -p "$LOGDIR"
echo "[GPU${GPU} TRUE-STD-${TAG} CHAIN START] $(date)" >> "$CHAINLOG"

echo "[PHASE: GSM8K, 8 seeds] $(date)" >> "$CHAINLOG"
for seed in 42 43 44 45 46 47 48 49; do
  OUTDIR="$OUTBASE/true_std_${TAG}_gsm8k_s${seed}"
  echo "[TRAIN START true_std_${TAG}_gsm8k s${seed}] $(date)" >> "$CHAINLOG"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON $TRAIN \
    --config "$CFGS/true_std_${TAG}_gsm8k.yaml" --seed $seed \
    --dataset gsm8k --nas-dir /nas/Dataset/nlp --output-dir "$OUTDIR" \
    --batch-size 16 \
    >> "$LOGDIR/gpu${GPU}_true_std_${TAG}_gsm8k_s${seed}.log" 2>&1
  rc=$?
  echo "[TRAIN DONE true_std_${TAG}_gsm8k s${seed} rc=${rc}] $(date)" >> "$CHAINLOG"
  if [ $rc -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON $EVAL \
      --checkpoint "$OUTDIR/best_model" --dataset gsm8k \
      --nas-dir /nas/Dataset/nlp --split dev \
      --output-file "$EVALDIR/eval_true_std_${TAG}_gsm8k_s${seed}.json" \
      >> "$LOGDIR/gpu${GPU}_eval_true_std_${TAG}_gsm8k_s${seed}.log" 2>&1
    echo "[EVAL DONE true_std_${TAG}_gsm8k s${seed} rc=$?] $(date)" >> "$CHAINLOG"
  fi
done

echo "[PHASE: COGS, 5 seeds] $(date)" >> "$CHAINLOG"
for seed in 42 43 44 45 46; do
  OUTDIR="$OUTBASE/true_std_${TAG}_cogs_s${seed}"
  echo "[TRAIN START true_std_${TAG}_cogs s${seed}] $(date)" >> "$CHAINLOG"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON $TRAIN \
    --config "$CFGS/true_std_${TAG}_cogs.yaml" --seed $seed \
    --dataset cogs --nas-dir /nas/Dataset/nlp --output-dir "$OUTDIR" \
    --batch-size 4 \
    >> "$LOGDIR/gpu${GPU}_true_std_${TAG}_cogs_s${seed}.log" 2>&1
  rc=$?
  echo "[TRAIN DONE true_std_${TAG}_cogs s${seed} rc=${rc}] $(date)" >> "$CHAINLOG"
  if [ $rc -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON $EVAL \
      --checkpoint "$OUTDIR/best_model" --dataset cogs \
      --nas-dir /nas/Dataset/nlp --split dev --batch-size 4 \
      --output-file "$EVALDIR/eval_true_std_${TAG}_cogs_s${seed}_dev.json" \
      >> "$LOGDIR/gpu${GPU}_eval_true_std_${TAG}_cogs_s${seed}_dev.log" 2>&1
    echo "[EVAL DONE true_std_${TAG}_cogs s${seed} dev rc=$?] $(date)" >> "$CHAINLOG"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON $EVAL \
      --checkpoint "$OUTDIR/best_model" --dataset cogs \
      --nas-dir /nas/Dataset/nlp --split gen --batch-size 4 \
      --output-file "$EVALDIR/eval_true_std_${TAG}_cogs_s${seed}_gen.json" \
      >> "$LOGDIR/gpu${GPU}_eval_true_std_${TAG}_cogs_s${seed}_gen.log" 2>&1
    echo "[EVAL DONE true_std_${TAG}_cogs s${seed} gen rc=$?] $(date)" >> "$CHAINLOG"
  fi
done

echo "[GPU${GPU} TRUE-STD-${TAG} CHAIN DONE] $(date)" >> "$CHAINLOG"

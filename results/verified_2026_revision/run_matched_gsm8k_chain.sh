#!/bin/bash
# Parameter-matched Standard Transformer (hidden_dim=448, ffn_dim=1370, 464.5M
# params, exact match to RelTransformer) GSM8K chain, 8 seeds.
GPU=${1:?usage: run_matched_gsm8k_chain.sh <gpu_id>}
PYTHON=/home/muhiddin/miniconda3/envs/har/bin/python
TRAIN=/home/muhiddin/relational-attention/scripts/train.py
EVAL=/home/muhiddin/relational-attention/scripts/evaluate.py
OUTBASE=/nas/Dataset/experiments/relational-attention/muhiddin_outputs
CFGS=/home/muhiddin/relational-attention/configs
LOGDIR=/home/muhiddin/relational-attention/results/verified_2026_revision/logs
EVALDIR=/home/muhiddin/relational-attention/results/verified_2026_revision
CHAINLOG=$LOGDIR/gpu${GPU}_matched448_gsm8k_chain.log

mkdir -p "$LOGDIR"
echo "[GPU${GPU} MATCHED-448 GSM8K CHAIN START] $(date)" >> "$CHAINLOG"

for seed in 42 43 44 45 46 47 48 49; do
  OUTDIR="$OUTBASE/std_matched448_gsm8k_s${seed}"
  echo "[TRAIN START matched448_gsm8k s${seed}] $(date)" >> "$CHAINLOG"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON $TRAIN \
    --config "$CFGS/std_transformer_448_matched_gsm8k.yaml" --seed $seed \
    --dataset gsm8k --nas-dir /nas/Dataset/nlp --output-dir "$OUTDIR" \
    --batch-size 16 \
    >> "$LOGDIR/gpu${GPU}_matched448_gsm8k_s${seed}.log" 2>&1
  rc=$?
  echo "[TRAIN DONE matched448_gsm8k s${seed} rc=${rc}] $(date)" >> "$CHAINLOG"

  if [ $rc -eq 0 ]; then
    echo "[EVAL START matched448_gsm8k s${seed}] $(date)" >> "$CHAINLOG"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON $EVAL \
      --checkpoint "$OUTDIR/best_model" --dataset gsm8k \
      --nas-dir /nas/Dataset/nlp --split dev \
      --output-file "$EVALDIR/eval_matched448_gsm8k_s${seed}.json" \
      >> "$LOGDIR/gpu${GPU}_eval_matched448_gsm8k_s${seed}.log" 2>&1
    echo "[EVAL DONE matched448_gsm8k s${seed} rc=$?] $(date)" >> "$CHAINLOG"
  fi
done

echo "[GPU${GPU} MATCHED-448 GSM8K CHAIN DONE] $(date)" >> "$CHAINLOG"

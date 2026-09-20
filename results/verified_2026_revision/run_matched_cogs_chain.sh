#!/bin/bash
# Parameter-matched Standard Transformer (hidden_dim=448, ffn_dim=1370, 464.5M
# params, exact match to RelTransformer) COGS-with-copy-mechanism chain, 5 seeds
# (reviewer requested >=5 seeds for the COGS comparison generally; we apply the
# same seed count here for the matched control).
GPU=${1:?usage: run_matched_cogs_chain.sh <gpu_id>}
PYTHON=/home/muhiddin/miniconda3/envs/har/bin/python
TRAIN=/home/muhiddin/relational-attention/scripts/train.py
EVAL=/home/muhiddin/relational-attention/scripts/evaluate.py
OUTBASE=/nas/Dataset/experiments/relational-attention/muhiddin_outputs
CFGS=/home/muhiddin/relational-attention/configs
LOGDIR=/home/muhiddin/relational-attention/results/verified_2026_revision/logs
EVALDIR=/home/muhiddin/relational-attention/results/verified_2026_revision
CHAINLOG=$LOGDIR/gpu${GPU}_matched448_cogs_chain.log

mkdir -p "$LOGDIR"
echo "[GPU${GPU} MATCHED-448 COGS CHAIN START] $(date)" >> "$CHAINLOG"

for seed in 42 43 44 45 46; do
  OUTDIR="$OUTBASE/std_matched448_cogs_s${seed}"
  echo "[TRAIN START matched448_cogs s${seed}] $(date)" >> "$CHAINLOG"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON $TRAIN \
    --config "$CFGS/std_transformer_448_matched_cogs.yaml" --seed $seed \
    --dataset cogs --nas-dir /nas/Dataset/nlp --output-dir "$OUTDIR" \
    --batch-size 4 \
    >> "$LOGDIR/gpu${GPU}_matched448_cogs_s${seed}.log" 2>&1
  rc=$?
  echo "[TRAIN DONE matched448_cogs s${seed} rc=${rc}] $(date)" >> "$CHAINLOG"

  if [ $rc -eq 0 ]; then
    echo "[EVAL START matched448_cogs s${seed} dev] $(date)" >> "$CHAINLOG"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON $EVAL \
      --checkpoint "$OUTDIR/best_model" --dataset cogs \
      --nas-dir /nas/Dataset/nlp --split dev --batch-size 4 \
      --output-file "$EVALDIR/eval_matched448_cogs_s${seed}_dev.json" \
      >> "$LOGDIR/gpu${GPU}_eval_matched448_cogs_s${seed}_dev.log" 2>&1
    echo "[EVAL DONE matched448_cogs s${seed} dev rc=$?] $(date)" >> "$CHAINLOG"

    echo "[EVAL START matched448_cogs s${seed} gen] $(date)" >> "$CHAINLOG"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON $EVAL \
      --checkpoint "$OUTDIR/best_model" --dataset cogs \
      --nas-dir /nas/Dataset/nlp --split gen --batch-size 4 \
      --output-file "$EVALDIR/eval_matched448_cogs_s${seed}_gen.json" \
      >> "$LOGDIR/gpu${GPU}_eval_matched448_cogs_s${seed}_gen.log" 2>&1
    echo "[EVAL DONE matched448_cogs s${seed} gen rc=$?] $(date)" >> "$CHAINLOG"
  fi
done

echo "[GPU${GPU} MATCHED-448 COGS CHAIN DONE] $(date)" >> "$CHAINLOG"

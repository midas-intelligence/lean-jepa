#!/bin/bash
# Sequential full-validation evaluation of v3: ONE Lean worker at a time.
# model on all 2000 (batches of 200), then fixed-recipe, then random.
# Idempotent: batches whose output JSON already exists are skipped.
cd "$(dirname "$0")"

# wait for any already-running search worker (no parallelism)
while pgrep -f "lean.search" >/dev/null; do sleep 60; done

run_batch() {  # method offset ckpt_flag...
  local method=$1 off=$2
  local out="logs/search_v3_${method}_${off}.json"
  [ -f "$out" ] && return 0
  echo "[chain $(date '+%F %T')] $method offset $off" >> logs/val_chain.log
  python3 -u -m lean.search --method "$method" \
    ${method:+$( [ "$method" = model ] && echo "--ckpt checkpoints/v3_pilot_seed42.pt" )} \
    --num-theorems 200 --offset "$off" --timeout 240 --split val \
    --out "$out" >> "logs/search_v3_${method}_${off}.txt" 2>&1
}

# Migrate pre-chain outputs into the chain naming (search_v3_<method>_<offset>.json).
# search_v3_model_200.json was the FIRST run = OFFSET 0 (named by count, not offset).
if [ -f logs/search_v3_model_200.json ] && [ ! -f logs/search_v3_model_0.json ]; then
  mv logs/search_v3_model_200.json logs/search_v3_model_0.json
fi
# the second batch (offset 200) wrote *_200b.json
if [ -f logs/search_v3_model_200b.json ] && [ ! -f logs/search_v3_model_200.json ]; then
  mv logs/search_v3_model_200b.json logs/search_v3_model_200.json
fi

for off in 0 200 400 600 800 1000 1200 1400 1600 1800; do
  run_batch model "$off"
done
for off in 0 200 400 600 800 1000 1200 1400 1600 1800; do
  run_batch fixed "$off"
done
for off in 0 200 400 600 800 1000 1200 1400 1600 1800; do
  run_batch random "$off"
done
echo "[chain $(date '+%F %T')] ALL DONE" >> logs/val_chain.log
touch VAL_CHAIN_DONE

#!/bin/bash
# SEALED-TEST evaluation — the final, pre-registered paper number.
# ONE Lean worker at a time: model → fixed → random over the 2000 test
# theorems (batches of 200). Idempotent (skips completed batch JSONs).
# Run ONCE; do not tune anything after seeing these results.
cd "$(dirname "$0")"
while pgrep -f "Python.*lean.search" >/dev/null; do sleep 60; done

run_batch() {  # method offset
  local method=$1 off=$2
  local out="logs/search_test_${method}_${off}.json"
  [ -f "$out" ] && return 0
  echo "[test-chain $(date '+%F %T')] $method offset $off" >> logs/test_chain.log
  local ckpt=""
  [ "$method" = model ] && ckpt="--ckpt checkpoints/v3_pilot_seed42.pt"
  python3 -u -m lean.search --method "$method" $ckpt \
    --num-theorems 200 --offset "$off" --timeout 240 --split test \
    --out "$out" >> "logs/search_test_${method}_${off}.txt" 2>&1
}

echo "[test-chain $(date '+%F %T')] STARTED" >> logs/test_chain.log
for m in model fixed random; do
  for off in 0 200 400 600 800 1000 1200 1400 1600 1800; do
    run_batch "$m" "$off"
  done
done
echo "[test-chain $(date '+%F %T')] ALL DONE" >> logs/test_chain.log
touch TEST_CHAIN_DONE

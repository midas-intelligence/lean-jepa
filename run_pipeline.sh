#!/bin/bash
# Autonomous 3-seed training pipeline for Lean-JEPA v2.
# Runs independently of any API/agent availability.
# Chains: seed 42 (already running) -> seed 137 -> seed 256.
# Relaunches a seed once if it dies without finishing.
# Writes PIPELINE_DONE marker when all seeds complete.

cd "$(dirname "$0")"
mkdir -p logs

finished() {  # seed -> 0 if FINAL line present
  grep -q "^FINAL" "logs/seed$1.txt" 2>/dev/null
}

running() {  # seed -> 0 if a train process for that seed is alive
  pgrep -f "lean.train --seed $1" >/dev/null 2>&1
}

run_seed() {  # launch + wait + one retry
  local seed=$1
  for attempt in 1 2; do
    if finished "$seed"; then return 0; fi
    if ! running "$seed"; then
      echo "[pipeline $(date '+%F %T')] launching seed $seed (attempt $attempt)" >> logs/pipeline.log
      nohup python3 -u -m lean.train --seed "$seed" --max-epochs 12 > "logs/seed$seed.txt" 2>&1 &
    fi
    # wait until process exits
    while running "$seed"; do sleep 120; done
    if finished "$seed"; then
      echo "[pipeline $(date '+%F %T')] seed $seed FINISHED" >> logs/pipeline.log
      return 0
    fi
    echo "[pipeline $(date '+%F %T')] seed $seed died unfinished" >> logs/pipeline.log
  done
  echo "[pipeline $(date '+%F %T')] seed $seed FAILED twice — aborting" >> logs/pipeline.log
  return 1
}

echo "[pipeline $(date '+%F %T')] pipeline started" >> logs/pipeline.log
run_seed 42  || exit 1
run_seed 137 || exit 1
run_seed 256 || exit 1
echo "[pipeline $(date '+%F %T')] ALL SEEDS COMPLETE" >> logs/pipeline.log
touch PIPELINE_DONE

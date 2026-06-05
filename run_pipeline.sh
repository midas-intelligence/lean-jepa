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

stale() {  # seed -> 0 if log untouched for >3h while process "runs" (MPS hang)
  local f="logs/seed$1.txt"
  [ -f "$f" ] || return 1
  local age=$(( $(date +%s) - $(stat -f %m "$f") ))
  [ "$age" -gt 10800 ]
}

run_seed() {  # launch + wait + retries; resumes from checkpoint when available
  local seed=$1
  for attempt in 1 2 3; do
    if finished "$seed"; then return 0; fi
    if ! running "$seed"; then
      local resume_flag=""
      if [ -f "checkpoints/lean_jepa_seed${seed}_best.pt" ]; then
        resume_flag="--resume checkpoints/lean_jepa_seed${seed}_best.pt"
      fi
      echo "[pipeline $(date '+%F %T')] launching seed $seed (attempt $attempt) $resume_flag" >> logs/pipeline.log
      # Cap MPS allocator (default high=1.7 is unbounded growth -> wired-memory
      # pressure -> GPU command stalls, the hang we hit at epoch 6).
      # Low watermark must be <= high watermark (default low=1.4 -> crash at 0.7 high).
      PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 \
      PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8 \
      nohup python3 -u -m lean.train --seed "$seed" --max-epochs 12 $resume_flag >> "logs/seed$seed.txt" 2>&1 &
    fi
    # wait until process exits; kill if hung (log stale >3h)
    while running "$seed"; do
      sleep 120
      if stale "$seed"; then
        echo "[pipeline $(date '+%F %T')] seed $seed HUNG (log stale >3h) — killing" >> logs/pipeline.log
        pkill -f "lean.train --seed $seed"
        sleep 10
        break
      fi
    done
    if finished "$seed"; then
      echo "[pipeline $(date '+%F %T')] seed $seed FINISHED" >> logs/pipeline.log
      return 0
    fi
    echo "[pipeline $(date '+%F %T')] seed $seed died unfinished" >> logs/pipeline.log
  done
  echo "[pipeline $(date '+%F %T')] seed $seed FAILED 3 attempts — aborting" >> logs/pipeline.log
  return 1
}

echo "[pipeline $(date '+%F %T')] pipeline started" >> logs/pipeline.log
run_seed 42  || exit 1
run_seed 137 || exit 1
run_seed 256 || exit 1
echo "[pipeline $(date '+%F %T')] ALL SEEDS COMPLETE" >> logs/pipeline.log
touch PIPELINE_DONE

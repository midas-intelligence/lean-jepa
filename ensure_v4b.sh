#!/bin/bash
# Idempotent v4b launcher — prevents duplicate trainers.
# Safe to call repeatedly (e.g. from a monitoring loop): starts v4b ONLY if
# none is running and it hasn't already finished. A suspended process (laptop
# sleep) still shows in pgrep, so this won't spawn a duplicate on wake.
cd "$(dirname "$0")"
if grep -q "^FINAL" logs/v4b_seed42.txt 2>/dev/null; then
  echo "v4b already finished"; exit 0
fi
if pgrep -f "Python.*lean.train.*42v4b" >/dev/null; then
  echo "v4b already running (PID $(pgrep -f 'Python.*lean.train.*42v4b' | tr '\n' ' '))"; exit 0
fi
PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8 \
nohup python3 -u -m lean.train --seed 42 --max-epochs 12 --batch-size 12 --accum-steps 4 \
  --aux-weight 2.0 --value-log --value-into-encoder \
  --config large512 --data-dir data/processed_v4 --run-tag 42v4b >> logs/v4b_seed42.txt 2>&1 &
echo "v4b launched PID $!"

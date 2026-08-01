#!/usr/bin/env bash
set -euo pipefail
cd /nobackup/enzez/Representation-Anchored-Pseudo-Labeling
queue='artifacts/benchmark_queues/koniq10k_ratio_first_8gpu/queue_state.json'
while [[ ! -f /nobackup/enzez/data/rsna_bone_age/2017/protocol/audit_summary.json ]] || \
      [[ "$(jq -r '.benchmark_status // ""' "$queue")" != exploratory_complete_at_5pct ]] || \
      pgrep -f 'training.koniq10k_benchmark' >/dev/null || \
      [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; do
  sleep 30
done
exec /nobackup/enzez/.venv/bin/python scripts/run_rsna_bone_age_ratio_first_8gpu.py

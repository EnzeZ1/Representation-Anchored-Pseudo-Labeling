#!/usr/bin/env bash
set -o pipefail
set -u

project_root=/nobackup/enzez/LatentMemo/pseudo-labels
physical_gpu=${1:?physical GPU index required}
output="$project_root/artifacts/benchmarks/utkface/dinov2/smoke/ratio_0.05/rapl/seed_0"
queue_root="$project_root/artifacts/benchmark_queues/utkface_dinov2_ratios"
mkdir -p "$output" "$queue_root"

export CUDA_VISIBLE_DEVICES="$physical_gpu"
export PYTHONUNBUFFERED=1
export UTKFACE_BENCHMARK_ROOT="$project_root"
export PYTHONPATH="$project_root/artifacts/utkface_5pct/python_deps:$project_root${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$queue_root/matplotlib"
export NUMBA_CACHE_DIR="$queue_root/numba_cache"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

command=(/nobackup/enzez/.venv/bin/python "$project_root/train.py"
  -dataset utkface --method probe --labeled_ratio 0.05
  --backbone dinov2 --probe_backbone dinov2 --dino s
  --save "$output/best.pt"
  --benchmark_manifest "$project_root/data_processing/splits/utkface_ratio_0.05_seed_0.json"
  --benchmark_output_dir "$output"
  --data_dir "$project_root/data/utkface_all" --seed 0 --epochs 1)
printf '%q ' "${command[@]}" > "$output/command.txt"
printf '\n' >> "$output/command.txt"

start_timestamp=$(date --iso-8601=seconds)
gpu_uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F, -v gpu="$physical_gpu" '$1+0==gpu {gsub(/^ +| +$/, "", $2); print $2}')
printf 'timestamp,python_pid,parent_pid,physical_gpu_index,gpu_uuid,process_gpu_memory_mib,pytorch_allocated_bytes,pytorch_reserved_bytes,host_rss_bytes,host_available_bytes\n' > "$output/resource_trace.csv"

"${command[@]}" > >(tee -a "$output/run.log") 2>&1 &
python_pid=$!
parent_pid=$$
printf '%s\n' "$python_pid" > "$output/python.pid"
printf '%s\n' "$parent_pid" > "$output/parent.pid"

monitor() {
  while kill -0 "$python_pid" 2>/dev/null; do
    timestamp=$(date +%s.%N)
    gpu_mem=$(nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | awk -F, -v pid="$python_pid" '$1+0==pid {gsub(/ /,"",$2); print $2}' | tail -1)
    rss=$(awk '/VmRSS:/ {print $2*1024}' "/proc/$python_pid/status" 2>/dev/null || true)
    available=$(awk '/MemAvailable:/ {print $2*1024}' /proc/meminfo)
    allocated=0; reserved=0
    if [[ -s "$output/torch_resource_trace.csv" ]]; then
      latest=$(tail -1 "$output/torch_resource_trace.csv")
      allocated=$(printf '%s' "$latest" | awk -F, '{print $2}')
      reserved=$(printf '%s' "$latest" | awk -F, '{print $3}')
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$timestamp" "$python_pid" "$parent_pid" "$physical_gpu" "$gpu_uuid" "${gpu_mem:-0}" "${allocated:-0}" "${reserved:-0}" "${rss:-0}" "${available:-0}" >> "$output/resource_trace.csv"
    sleep 2
  done
}
monitor & monitor_pid=$!

set +e
wait "$python_pid"
exit_code=$?
set -e
wait "$monitor_pid" 2>/dev/null || true
end_timestamp=$(date --iso-8601=seconds)
inferred_signal=null
if (( exit_code > 128 )); then inferred_signal=$((exit_code-128)); fi

/nobackup/enzez/.venv/bin/python - "$output" "$python_pid" "$parent_pid" "$physical_gpu" "$gpu_uuid" "$start_timestamp" "$end_timestamp" "$exit_code" "$inferred_signal" <<'PY'
import csv,json,sys
from pathlib import Path
out=Path(sys.argv[1]); rows=list(csv.DictReader((out/'resource_trace.csv').open()))
def peak(name, fn=max):
 values=[float(row[name] or 0) for row in rows]; return fn(values) if values else None
markers=[]
if (out/'progress.jsonl').exists():
 markers=[json.loads(line) for line in (out/'progress.jsonl').read_text().splitlines() if line.strip()]
payload={
 'command':(out/'command.txt').read_text().strip(),'python_pid':int(sys.argv[2]),'parent_pid':int(sys.argv[3]),
 'physical_gpu_index':int(sys.argv[4]),'gpu_uuid':sys.argv[5],'process_local_cuda_device':'cuda:0',
 'start_timestamp':sys.argv[6],'end_timestamp':sys.argv[7],'exit_code':int(sys.argv[8]),
 'inferred_signal':None if sys.argv[9]=='null' else int(sys.argv[9]),
 'last_completed_progress_marker':markers[-1]['marker'] if markers else None,
 'peak_process_gpu_memory_mib':peak('process_gpu_memory_mib'),
 'peak_pytorch_allocated_memory_bytes':peak('pytorch_allocated_bytes'),
 'peak_pytorch_reserved_memory_bytes':peak('pytorch_reserved_bytes'),
 'peak_host_rss_bytes':peak('host_rss_bytes'),'minimum_observed_host_available_ram_bytes':peak('host_available_bytes',min),
 'python_traceback_found':'Traceback (most recent call last)' in (out/'run.log').read_text(),
}
(out/'smoke_exit.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY

if (( exit_code == 0 )); then
  if /nobackup/enzez/.venv/bin/python -c "from pathlib import Path; from scripts.run_utkface_dinov2_sweep import valid_smoke; assert valid_smoke(Path('$output'))"; then
    printf '[%s] diagnostic smoke valid; starting corrected formal pipeline\n' "$end_timestamp" | tee -a "$queue_root/launcher.log"
    exec /nobackup/enzez/.venv/bin/python "$project_root/scripts/run_utkface_dinov2_pipeline.py"
  fi
fi

printf '[%s] diagnostic smoke failed or did not validate; formal queue not started\n' "$end_timestamp" | tee -a "$queue_root/launcher.log"
while true; do sleep 3600; done

#!/usr/bin/env python3
"""Persistent, resumable 36-run formal CheXchoNet LVIDd Temporal Ensembling queue."""
from __future__ import annotations
import scripts.run_utkface_temporal_ensembling_8gpu as queue

queue.QUEUE = queue.ROOT / 'artifacts/benchmark_queues/chexchonet_lvidd_temporal_ensembling'
queue.RESULTS = queue.ROOT / 'artifacts/benchmarks/chexchonet_lvidd/temporal_ensembling'
queue.STATE = queue.QUEUE / 'queue_state.json'; queue.LOG = queue.QUEUE / 'launcher.log'
queue.DIGEST = 'b82d75b491170e123ca60e114f909913e09ecac15bef1153038d8845dc1cf911'
queue.DATASET_ARG = 'chexchonet_lvidd'; queue.DATASET_LABEL = 'CheXchoNet LVIDd'
queue.DATA_ROOT = queue.Path('/nobackup/enzez/data/chexchonet/1.0.0')
queue.MANIFEST_ROOT = queue.ROOT / 'artifacts/protected_protocol/chexchonet_lvidd'
queue.MANIFEST_STEM = 'chexchonet_lvidd'; queue.TEST_SIZE = 3708
queue.PROTOCOL_VERSION = 'chexchonet-lvidd-benchmark-v1'; queue.BATCH_SIZE = 32; queue.NUM_WORKERS = 4

if __name__ == '__main__':
 try: queue.main()
 except Exception:
  queue.QUEUE.mkdir(parents=True,exist_ok=True); queue.log_exception('fatal queue-controller exception'); raise

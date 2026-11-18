#!/usr/bin/env python3
"""Persistent, resumable 36-run formal RSNA bone-age Temporal Ensembling queue."""
from __future__ import annotations
import scripts.run_utkface_temporal_ensembling_8gpu as queue

queue.QUEUE = queue.ROOT / 'artifacts/benchmark_queues/rsna_bone_age_temporal_ensembling'
queue.RESULTS = queue.ROOT / 'artifacts/benchmarks/rsna_bone_age/temporal_ensembling'
queue.STATE = queue.QUEUE / 'queue_state.json'; queue.LOG = queue.QUEUE / 'launcher.log'
queue.DIGEST = '43ac007dc7469b8b6d85dc9ec6af6424753399b8a6171e168bc8d14dac0cd93f'
queue.DATASET_ARG = 'rsna_bone_age'; queue.DATASET_LABEL = 'RSNA Pediatric Bone Age Challenge 2017'
queue.DATA_ROOT = queue.Path('/nobackup/enzez/data/rsna_bone_age/2017/extracted')
queue.MANIFEST_ROOT = queue.Path('/nobackup/enzez/data/rsna_bone_age/2017/protocol')
queue.MANIFEST_STEM = ''; queue.MANIFEST_TEMPLATE = 'ratio_0{ratio}_seed_{seed}.json'; queue.TEST_SIZE = 1425
queue.PROTOCOL_VERSION = 'rsna-bone-age-benchmark-v1'; queue.BATCH_SIZE = 16; queue.NUM_WORKERS = 2

if __name__ == '__main__':
 # The canonical RSNA filenames have no dataset-name prefix.
 original = queue.MANIFEST_STEM
 try: queue.main()
 except Exception:
  queue.QUEUE.mkdir(parents=True,exist_ok=True); queue.log_exception('fatal queue-controller exception'); raise

#!/usr/bin/env python3
"""Persistent, resumable 36-run formal IMDB-WIKI Temporal Ensembling queue."""
from __future__ import annotations

import scripts.run_utkface_temporal_ensembling_8gpu as queue

queue.QUEUE = queue.ROOT / 'artifacts/benchmark_queues/imdb_wiki_temporal_ensembling'
queue.RESULTS = queue.ROOT / 'artifacts/benchmarks/imdb_wiki/temporal_ensembling'
queue.STATE = queue.QUEUE / 'queue_state.json'
queue.LOG = queue.QUEUE / 'launcher.log'
queue.DIGEST = '919fe3e1b959e1fe75e08e83310a84c1c3a9d53a16812a1bb5f1e0117ba97f43'
queue.DATASET_ARG = 'imdb_wiki'
queue.DATASET_LABEL = 'IMDB-WIKI-DIR'
queue.DATA_ROOT = queue.ROOT / 'data/imdb_wiki'
queue.MANIFEST_STEM = 'imdb_wiki'
queue.TEST_SIZE = 11022
queue.PROTOCOL_VERSION = 'imdb-wiki-dir-benchmark-v1'

if __name__ == '__main__':
    try:
        queue.main()
    except Exception:
        queue.QUEUE.mkdir(parents=True, exist_ok=True)
        queue.log_exception('fatal queue-controller exception')
        raise

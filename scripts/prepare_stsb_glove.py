#!/usr/bin/env python3
"""Prepare the HPL-compatible 30k STS-B vocabulary and GloVe matrix."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from data_processing.stsb import load_cohort, simple_tokenize

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, default=ROOT / "data_processing/splits/stsb_dir_cohort_v1.json")
    parser.add_argument("--glove", type=Path, default=ROOT / "data/glove/glove.840B.300d.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "data/glove/stsb_glove_840b_300d.pt")
    parser.add_argument("--max-vocab", type=int, default=30_000)
    args = parser.parse_args()
    cohort = load_cohort(args.cohort)
    counts = Counter()
    for record in cohort["records"]:
        counts.update(simple_tokenize(record["sentence1"]))
        counts.update(simple_tokenize(record["sentence2"]))
    vocabulary = ["<pad>", "<unk>"] + [
        token for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ][: args.max_vocab - 2]
    token_to_index = {token: index for index, token in enumerate(vocabulary)}
    generator = np.random.default_rng(0)
    embeddings = generator.standard_normal((len(vocabulary), 300), dtype=np.float32)
    embeddings[0] = 0
    found = 0
    with args.glove.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            token, separator, values = line.rstrip().partition(" ")
            index = token_to_index.get(token)
            if separator and index is not None:
                vector = np.fromstring(values, sep=" ", dtype=np.float32)
                if vector.shape == (300,):
                    embeddings[index] = vector
                    found += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "vocabulary": vocabulary,
        "embeddings": torch.from_numpy(embeddings),
        "padding_index": 0,
        "unknown_index": 1,
        "glove_identifier": "glove.840B.300d",
        "glove_source": "https://nlp.stanford.edu/data/glove.840B.300d.zip",
        "cohort_sha256": cohort["cohort_sha256"],
        "found_vectors": found,
    }, args.output)
    print(json.dumps({
        "output": str(args.output),
        "vocabulary_size": len(vocabulary),
        "found_vectors": found,
        "cohort_sha256": cohort["cohort_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()

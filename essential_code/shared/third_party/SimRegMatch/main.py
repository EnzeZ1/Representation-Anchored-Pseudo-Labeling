import json
import os
from pathlib import Path

import torch
from itertools import product

from utils.args import SimRegMatch_parser
from tasks.SimRegMatchTrainer import SimRegMatchTrainer

_MIB = 1024 ** 2


def _memory_record(device, label, epoch=None):
    """Read-only CUDA allocator telemetry; does not allocate tensors."""
    record = {
        "label": label,
        "epoch": epoch,
        "allocated_mib": torch.cuda.memory_allocated(device) / _MIB,
        "reserved_mib": torch.cuda.memory_reserved(device) / _MIB,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / _MIB,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / _MIB,
        "gpu_total_mib": torch.cuda.get_device_properties(device).total_memory / _MIB,
    }
    return record


def _write_memory_record(output_dir, record):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "vram_peak.jsonl").open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    tmp = output_dir / f"vram_peak.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, output_dir / "vram_peak.json")


def main():
    parser = SimRegMatch_parser()
    args = parser.parse_args()
    args.cuda = torch.device("cuda:0")
    # Operational VRAM-logger initialization; no RNG or training semantics change.
    torch.cuda.set_device(args.cuda)

    if args.benchmark_epochs is not None:
        args.epochs = args.benchmark_epochs

    # Future-run instrumentation only: reset before model construction so the
    # recorded peak includes model, optimizer, training, validation, and test.
    torch.cuda.reset_peak_memory_stats(args.cuda)
    trainer = SimRegMatchTrainer(args)
    _write_memory_record(
        args.benchmark_output_dir,
        _memory_record(args.cuda, "after_trainer_initialization"),
    )

    for epoch in range(1, args.epochs + 1, 1):
        trainer.train(epoch)
        trainer.validation(epoch)
        trainer.collect_reliability_audit(epoch)
        if trainer.scheduler is not None:
            trainer.scheduler.step()
        _write_memory_record(
            args.benchmark_output_dir,
            _memory_record(args.cuda, "epoch_complete", epoch),
        )

    trainer.inference(epoch)
    _write_memory_record(
        args.benchmark_output_dir,
        _memory_record(args.cuda, "inference_complete", epoch),
    )


if __name__ == "__main__":
    main()

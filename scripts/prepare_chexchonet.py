#!/usr/bin/env python3
"""Aggregate-only authorized-release readiness audit; never downloads data."""
import argparse, json, sys
from dataclasses import asdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from data_processing.chexchonet import audit_release

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--decode", action="store_true"); args = parser.parse_args()
    audit = audit_release(args.data_root, decode=args.decode)
    print(json.dumps({**asdict(audit), "ready": audit.ready}, indent=2, sort_keys=True))
    raise SystemExit(0 if audit.ready else 2)
if __name__ == "__main__": main()

#!/usr/bin/env python3
"""
Converts factory.py JSONL triplets to Parquet for distill.py.
Filters to only verified (all-passed) samples — the student learns from correct code.
"""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import DATASET_DIR

JSONL_FILE = DATASET_DIR / "reasoning_triplets.jsonl"
PARQUET_FILE = DATASET_DIR / "reasoning_triplets.parquet"


def convert(jsonl_path: Path = JSONL_FILE, parquet_path: Path = PARQUET_FILE,
            filter_passed: bool = True):
    if not jsonl_path.exists():
        print(f"Not found: {jsonl_path}. Run factory.py first.")
        return

    print(f"Loading {jsonl_path}...")
    df = pd.read_json(str(jsonl_path), lines=True)

    total = len(df)
    passed = int((df["verification_stage"] == "all-passed").sum())
    failed = total - passed

    print(f"Total samples:     {total}")
    print(f"Verified (passed): {passed}")
    print(f"Failed refactors:  {failed}")

    if filter_passed:
        df = df[df["verification_stage"] == "all-passed"]
        if len(df) == 0:
            print("No verified samples to convert. Run factory.py with more/better prompts.")
            return
        print(f"Writing {len(df)} verified samples to {parquet_path}...")

    table = pa.Table.from_pandas(df)
    pq.write_table(table, str(parquet_path), compression="snappy")

    size_mb = parquet_path.stat().st_size / (1024 * 1024)
    print(f"Done: {parquet_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert JSONL triplets to Parquet")
    p.add_argument("--jsonl", type=Path, default=JSONL_FILE)
    p.add_argument("--parquet", type=Path, default=PARQUET_FILE)
    p.add_argument("--keep-failed", action="store_true",
                   help="Include failed samples too (not recommended)")
    args = p.parse_args()
    convert(args.jsonl, args.parquet, filter_passed=not args.keep_failed)

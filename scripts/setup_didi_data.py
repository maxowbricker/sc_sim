#!/usr/bin/env python3
"""
Prepare the DiDi GAIA dataset for sc_sim.

The raw dataset is distributed as one .tar.gz per day, each containing two
flat files (no extension, no subdirectory):
    order_YYYYMMDD   — trip orders (workers / tasks)
    gps_YYYYMMDD     — GPS traces

The simulator expects them renamed and placed into:
    data/didi/full_didi_gaia/496528674@qq.com_YYYYMMDD/order.txt
    data/didi/full_didi_gaia/496528674@qq.com_YYYYMMDD/gps.txt

This script does that extraction and renaming for every .tar.gz found in
the source directory.

Usage:
    python scripts/setup_didi_data.py --source "/path/to/滴滴 gaiya"
    python scripts/setup_didi_data.py --source "/path/to/滴滴 gaiya" --dest data/didi/full_didi_gaia
    python scripts/setup_didi_data.py --source "/path/to/滴滴 gaiya" --day 20161128   # single day only

Example (from repo root):
    python scripts/setup_didi_data.py \\
        --source "/Users/maxapple/Documents/All/Career/Fair Task Allocation in Spatial Crowdsourcing/honours files/Implementation/Didi-Gaia/滴滴 gaiya"
"""

import argparse
import os
import sys
import tarfile
import tempfile
from pathlib import Path


def extract_day(tar_path: Path, dest_root: Path, day_name: str) -> bool:
    """
    Extract one .tar.gz, rename the flat files, place into dest_root/<day_name>/.
    Returns True on success, False on skip/error.
    """
    out_dir = dest_root / day_name
    order_out = out_dir / "order.txt"
    gps_out = out_dir / "gps.txt"

    if order_out.exists() and gps_out.exists():
        print(f"  [skip] {day_name} — already extracted")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(tmp_path, filter="data")

        extracted = list(tmp_path.iterdir())
        date_suffix = day_name.split("_")[-1]  # e.g. "20161128"

        order_src = tmp_path / f"order_{date_suffix}"
        gps_src = tmp_path / f"gps_{date_suffix}"

        if not order_src.exists() or not gps_src.exists():
            # Fallback: look for any file containing "order" or "gps"
            found = {f.name: f for f in extracted}
            order_candidates = [f for n, f in found.items() if "order" in n]
            gps_candidates = [f for n, f in found.items() if "gps" in n]
            if not order_candidates or not gps_candidates:
                print(f"  [error] {day_name} — could not find order/gps files in archive. Contents: {[f.name for f in extracted]}")
                return False
            order_src = order_candidates[0]
            gps_src = gps_candidates[0]

        import shutil
        shutil.copy2(order_src, order_out)
        shutil.copy2(gps_src, gps_out)

    print(f"  [done]  {day_name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Extract DiDi GAIA .tar.gz files into sc_sim format")
    parser.add_argument(
        "--source", required=True,
        help='Path to folder containing 496528674@qq.com_YYYYMMDD.tar.gz files (e.g. "滴滴 gaiya")'
    )
    parser.add_argument(
        "--dest", default="data/didi/full_didi_gaia",
        help="Destination root directory (default: data/didi/full_didi_gaia)"
    )
    parser.add_argument(
        "--day", default=None,
        help="Extract a single day only, e.g. 20161128 (default: all days)"
    )
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    dest = Path(args.dest)

    if not source.exists():
        print(f"Error: source folder not found: {source}")
        sys.exit(1)

    # Resolve dest relative to repo root if not absolute
    if not dest.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        dest = repo_root / dest

    dest.mkdir(parents=True, exist_ok=True)

    archives = sorted(source.glob("496528674@qq.com_*.tar.gz"))
    if not archives:
        print(f"No .tar.gz files found in: {source}")
        sys.exit(1)

    if args.day:
        archives = [a for a in archives if args.day in a.name]
        if not archives:
            print(f"No archive found for day: {args.day}")
            sys.exit(1)

    print(f"Source : {source}")
    print(f"Dest   : {dest}")
    print(f"Days   : {len(archives)}")
    print()

    done, skipped, errors = 0, 0, 0
    for tar_path in archives:
        day_name = tar_path.stem.replace(".tar", "")  # e.g. 496528674@qq.com_20161128
        result = extract_day(tar_path, dest, day_name)
        if result is True:
            done += 1
        elif result is False:
            skipped += 1

    print()
    print(f"Complete: {done} extracted, {skipped} skipped (already present)")
    print(f"Data ready at: {dest}")


if __name__ == "__main__":
    main()

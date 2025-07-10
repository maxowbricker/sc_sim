#!/usr/bin/env python3
"""
Extract a small, self-consistent subset of the Didi Gaia dataset.

Usage
-----
(Place this script in the same folder as *gps.txt* and *order.txt*.)
It will write *small_order.txt* and *small_gps.txt* in the same folder.

Optional flags:
    --start "2016-11-02 00:00:00"    # ISO or epoch seconds; default = earliest in order.txt
    --span "15min"                   # pandas offset alias (default)
    --max_drivers 50                # cap number of drivers
"""
import argparse
import pandas as pd
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", help="Window start time (ISO or epoch seconds). If omitted uses first order timestamp.")
    p.add_argument("--span", default="15min", help="Window length as pandas offset alias (default: 15min)")
    p.add_argument("--max_drivers", type=int, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    # Root is directory where this script resides (alongside gps.txt / order.txt)
    root = Path(__file__).resolve().parent

    # Paths
    orders_path = root / "order.txt"
    gps_path    = root / "gps.txt"

    if not (orders_path.exists() and gps_path.exists()):
        raise FileNotFoundError("order.txt or gps.txt not found in script directory.")

    # Load orders first to determine default start if not supplied
    orders = pd.read_csv(orders_path, header=None)
    orders.columns = [
        "order_id",
        "start_billing",
        "end_billing",
        "pickup_lon",
        "pickup_lat",
        "dropoff_lon",
        "dropoff_lat",
    ]

    # Convert epoch to pandas Timestamp for filtering
    orders["start_billing_dt"] = pd.to_datetime(orders["start_billing"], unit="s", utc=True)

    if args.start is None:
        # Earliest timestamp in orders
        min_epoch = orders["start_billing"].min()
        start = pd.to_datetime(min_epoch, unit="s", utc=True)
    else:
        # Try epoch int first, else parse as datetime string
        try:
            start = pd.to_datetime(int(args.start), unit="s", utc=True)
        except ValueError:
            start = pd.to_datetime(args.start, utc=True)

    span = pd.Timedelta(args.span)
    end   = start + span

    # 1. Orders ----------------------------------------------------------
    orders_sub = orders[(orders["start_billing_dt"] >= start) &
                         (orders["start_billing_dt"] < end)]

    if args.max_drivers:
        # Keep only orders from the first <max_drivers> unique drivers
        gps_for_driver = pd.read_csv(gps_path, usecols=[0,2], header=None,
                                     names=["driver_id","timestamp"],
                                     nrows=10_000_000)  # read just a chunk to find early drivers
        early_drivers = gps_for_driver[gps_for_driver["timestamp"] < span.total_seconds() ] \
                           .driver_id.unique()[:args.max_drivers]
        orders_sub = orders_sub[orders_sub["order_id"].isin(
                        orders_sub[orders_sub["order_id"].isin(
                            gps_for_driver[gps_for_driver.driver_id.isin(early_drivers)].order_id.unique()
                        )].order_id)]

    orders_sub.to_csv(root / "small_order.txt", header=False, index=False)

    # 2. GPS -------------------------------------------------------------
    order_ids = set(orders_sub["order_id"])
    chunks = pd.read_csv(gps_path, header=None,
                         names=["driver_id","order_id","timestamp","lon","lat"],
                         chunksize=1_000_000)

    sel = []
    for chunk in chunks:
        chunk["timestamp_dt"] = pd.to_datetime(chunk["timestamp"], unit="s", utc=True)
        mask = (chunk["timestamp_dt"]>=start) & (chunk["timestamp_dt"]<end) & \
               (chunk["order_id"].isin(order_ids))
        sel.append(chunk.loc[mask, ["driver_id","order_id","timestamp","lon","lat"]])
    gps_sub = pd.concat(sel, ignore_index=True)
    gps_sub.to_csv(root / "small_gps.txt", header=False, index=False)

    print(
        f"Subset written to {root}. small_order.txt ({len(orders_sub)} rows), "
        f"small_gps.txt ({len(gps_sub)} rows)."
    )

if __name__ == "__main__":
    main()

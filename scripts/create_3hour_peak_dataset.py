#!/usr/bin/env python3
"""
Extract the optimal 3-hour peak window into dedicated dataset files.
This ensures guaranteed temporal overlap for all experiments.

Output:
- data/didi/gps_3hour_peak.txt
- data/didi/order_3hour_peak.txt
"""

import pandas as pd
from pathlib import Path

# Configuration
data_dir = Path("data/didi")
window_start = pd.Timestamp("2016-11-13 05:18:17", tz='UTC')
window_end = pd.Timestamp("2016-11-13 08:18:17", tz='UTC')

print("=" * 80)
print("3-HOUR PEAK WINDOW DATASET EXTRACTION")
print("=" * 80)
print(f"\n🎯 Target window: {window_start} to {window_end}")
print(f"   Duration: {(window_end - window_start).total_seconds() / 3600:.1f} hours")

# ============================================================================
# 1. Load and filter GPS data
# ============================================================================
print(f"\n📂 Loading GPS data...")
gps_path = data_dir / "gps_quarter_fixed.txt"
gps_df = pd.read_csv(gps_path, header=None)
gps_df.columns = ["driver_id", "order_id", "timestamp", "lon", "lat"]
gps_df["timestamp"] = pd.to_datetime(gps_df["timestamp"], unit="s", utc=True)

print(f"   Original: {len(gps_df):,} GPS pings, {gps_df['driver_id'].nunique():,} drivers")

# Filter GPS pings within window
gps_filtered = gps_df[
    (gps_df['timestamp'] >= window_start) & 
    (gps_df['timestamp'] <= window_end)
].copy()

print(f"   Filtered: {len(gps_filtered):,} GPS pings, {gps_filtered['driver_id'].nunique():,} drivers")
print(f"   Retention: {len(gps_filtered) / len(gps_df) * 100:.1f}% of pings")

# ============================================================================
# 2. Load and filter Orders data
# ============================================================================
print(f"\n📂 Loading Orders data...")
orders_path = data_dir / "order_quarter_fixed.txt"
orders_df = pd.read_csv(orders_path, header=None)
if orders_df.shape[1] == 8:
    orders_df = orders_df.iloc[:, :7]
orders_df.columns = [
    "order_id", "start_billing", "end_billing",
    "pickup_lon", "pickup_lat", "dropoff_lon", "dropoff_lat"
]
orders_df["start_billing"] = pd.to_datetime(orders_df["start_billing"], unit="s", utc=True)
orders_df["end_billing"] = pd.to_datetime(orders_df["end_billing"], unit="s", utc=True)

print(f"   Original: {len(orders_df):,} orders")

# Filter orders that START within window
orders_filtered = orders_df[
    (orders_df['start_billing'] >= window_start) & 
    (orders_df['start_billing'] <= window_end)
].copy()

print(f"   Filtered: {len(orders_filtered):,} orders")
print(f"   Retention: {len(orders_filtered) / len(orders_df) * 100:.1f}% of orders")

# ============================================================================
# 3. Convert back to Unix timestamps and save
# ============================================================================
print(f"\n💾 Saving filtered datasets...")

# Convert timestamps back to Unix epoch (seconds) for compatibility
gps_filtered['timestamp'] = gps_filtered['timestamp'].astype('int64') // 10**9
orders_filtered['start_billing'] = orders_filtered['start_billing'].astype('int64') // 10**9
orders_filtered['end_billing'] = orders_filtered['end_billing'].astype('int64') // 10**9

# Save GPS
gps_output = data_dir / "gps_3hour_peak.txt"
gps_filtered.to_csv(gps_output, index=False, header=False)
print(f"   ✅ Saved: {gps_output}")
print(f"      Size: {gps_output.stat().st_size / 1024 / 1024:.1f} MB")

# Save Orders
orders_output = data_dir / "order_3hour_peak.txt"
orders_filtered.to_csv(orders_output, index=False, header=False)
print(f"   ✅ Saved: {orders_output}")
print(f"      Size: {orders_output.stat().st_size / 1024 / 1024:.1f} MB")

# ============================================================================
# 4. Verify and summarize
# ============================================================================
print(f"\n🔍 VERIFICATION:")
print("=" * 80)

# Verify temporal overlap
gps_times = pd.to_datetime(gps_filtered['timestamp'], unit='s', utc=True)
orders_times = pd.to_datetime(orders_filtered['start_billing'], unit='s', utc=True)

print(f"GPS data:")
print(f"   Time range: {gps_times.min()} to {gps_times.max()}")
print(f"   Duration: {(gps_times.max() - gps_times.min()).total_seconds() / 3600:.2f} hours")
print(f"   Unique drivers: {gps_filtered['driver_id'].nunique():,}")

print(f"\nOrders data:")
print(f"   Time range: {orders_times.min()} to {orders_times.max()}")
print(f"   Duration: {(orders_times.max() - orders_times.min()).total_seconds() / 3600:.2f} hours")
print(f"   Total orders: {len(orders_filtered):,}")

print(f"\n✅ OVERLAP GUARANTEED:")
overlap_start = max(gps_times.min(), orders_times.min())
overlap_end = min(gps_times.max(), orders_times.max())
overlap_hours = (overlap_end - overlap_start).total_seconds() / 3600

print(f"   Overlap: {overlap_start} to {overlap_end}")
print(f"   Duration: {overlap_hours:.2f} hours")

print(f"\n📊 DATASET CAPACITY:")
print("=" * 80)
print(f"Available for sampling:")
print(f"   Workers: {gps_filtered['driver_id'].nunique():,}")
print(f"   Tasks: {len(orders_filtered):,}")
print(f"   Ratio: {len(orders_filtered) / gps_filtered['driver_id'].nunique():.2f} tasks/worker")

print(f"\nExperiment support (20K tasks fixed):")
for worker_count in [2000, 4000, 6000, 8000, 10000, 12000, 15000]:
    workers_ok = "✅" if worker_count <= gps_filtered['driver_id'].nunique() else "❌"
    tasks_ok = "✅" if 20000 <= len(orders_filtered) else "❌"
    print(f"   {workers_ok} {worker_count:>6,} workers, {tasks_ok} 20,000 tasks")

print("\n" + "=" * 80)
print("USAGE IN EXPERIMENTS:")
print("=" * 80)
print("""
Update data/didi/didi.py adapter to prefer these files:

def _load_gps(self) -> pd.DataFrame:
    peak_path = self.root / "gps_3hour_peak.txt"
    quarter_path = self.root / "gps_quarter_fixed.txt"
    path = peak_path if peak_path.exists() else quarter_path
    # ... rest of loading logic

def _load_orders(self) -> pd.DataFrame:
    peak_path = self.root / "order_3hour_peak.txt"
    quarter_path = self.root / "order_quarter_fixed.txt"
    path = peak_path if peak_path.exists() else quarter_path
    # ... rest of loading logic
""")
print("=" * 80)


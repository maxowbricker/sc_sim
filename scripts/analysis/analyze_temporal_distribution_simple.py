#!/usr/bin/env python3
"""
Analyze temporal distribution of DiDi GPS and Order data.
Shows when workers are active vs when tasks arrive.
"""

import pandas as pd
from pathlib import Path

# Load the data
data_dir = Path("data/didi")

print("=" * 80)
print("TEMPORAL DISTRIBUTION ANALYSIS")
print("=" * 80)

# Load GPS data (workers)
gps_path = data_dir / "gps_quarter_fixed.txt"
print(f"\n📂 Loading GPS data from: {gps_path}")
print(f"   Size: {gps_path.stat().st_size / 1024 / 1024:.1f} MB")

gps_df = pd.read_csv(gps_path, header=None)
gps_df.columns = ["driver_id", "order_id", "timestamp", "lon", "lat"]
gps_df["timestamp"] = pd.to_datetime(gps_df["timestamp"], unit="s", utc=True)

print(f"✅ Loaded {len(gps_df):,} GPS pings")
print(f"   Unique drivers: {gps_df['driver_id'].nunique():,}")
print(f"   Time range: {gps_df['timestamp'].min()} to {gps_df['timestamp'].max()}")
print(f"   Duration: {(gps_df['timestamp'].max() - gps_df['timestamp'].min()).total_seconds() / 3600:.1f} hours")

# Load Orders data (tasks)
orders_path = data_dir / "order_quarter_fixed.txt"
print(f"\n📂 Loading Orders data from: {orders_path}")
print(f"   Size: {orders_path.stat().st_size / 1024 / 1024:.1f} MB")

orders_df = pd.read_csv(orders_path, header=None)
if orders_df.shape[1] == 8:
    orders_df = orders_df.iloc[:, :7]
orders_df.columns = [
    "order_id", "start_billing", "end_billing",
    "pickup_lon", "pickup_lat", "dropoff_lon", "dropoff_lat"
]
orders_df["start_billing"] = pd.to_datetime(orders_df["start_billing"], unit="s", utc=True)

print(f"✅ Loaded {len(orders_df):,} orders")
print(f"   Time range: {orders_df['start_billing'].min()} to {orders_df['start_billing'].max()}")
print(f"   Duration: {(orders_df['start_billing'].max() - orders_df['start_billing'].min()).total_seconds() / 3600:.1f} hours")

# Check for temporal overlap
print(f"\n🔍 TEMPORAL OVERLAP CHECK:")
print("=" * 80)
gps_start = gps_df['timestamp'].min()
gps_end = gps_df['timestamp'].max()
orders_start = orders_df['start_billing'].min()
orders_end = orders_df['start_billing'].max()

overlap_start = max(gps_start, orders_start)
overlap_end = min(gps_end, orders_end)

if overlap_start < overlap_end:
    overlap_hours = (overlap_end - overlap_start).total_seconds() / 3600
    print(f"✅ OVERLAP EXISTS!")
    print(f"   Overlap period: {overlap_start} to {overlap_end}")
    print(f"   Overlap duration: {overlap_hours:.1f} hours")
else:
    print(f"❌ NO OVERLAP! GPS and Orders are in different time periods!")
    print(f"   GPS ends at:    {gps_end}")
    print(f"   Orders start at: {orders_start}")
    print(f"   Gap: {(orders_start - gps_end).total_seconds() / 3600:.1f} hours")

# Analyze worker availability (from GPS data)
print(f"\n📊 WORKER AVAILABILITY ANALYSIS:")
print("=" * 80)
worker_periods = gps_df.groupby('driver_id')['timestamp'].agg(['min', 'max']).reset_index()
worker_periods.columns = ['driver_id', 'first_seen', 'last_seen']
worker_periods['active_duration_hours'] = (worker_periods['last_seen'] - worker_periods['first_seen']).dt.total_seconds() / 3600

print(f"Total unique workers: {len(worker_periods):,}")
print(f"Average worker shift: {worker_periods['active_duration_hours'].mean():.2f} hours")
print(f"Median worker shift:  {worker_periods['active_duration_hours'].median():.2f} hours")
print(f"Min shift:            {worker_periods['active_duration_hours'].min():.2f} hours")
print(f"Max shift:            {worker_periods['active_duration_hours'].max():.2f} hours")
print(f"\nShift duration breakdown:")
print(f"  < 1 hour:    {(worker_periods['active_duration_hours'] < 1).sum():>6,} ({(worker_periods['active_duration_hours'] < 1).sum() / len(worker_periods) * 100:>5.1f}%)")
print(f"  1-3 hours:   {((worker_periods['active_duration_hours'] >= 1) & (worker_periods['active_duration_hours'] < 3)).sum():>6,} ({((worker_periods['active_duration_hours'] >= 1) & (worker_periods['active_duration_hours'] < 3)).sum() / len(worker_periods) * 100:>5.1f}%)")
print(f"  3-6 hours:   {((worker_periods['active_duration_hours'] >= 3) & (worker_periods['active_duration_hours'] < 6)).sum():>6,} ({((worker_periods['active_duration_hours'] >= 3) & (worker_periods['active_duration_hours'] < 6)).sum() / len(worker_periods) * 100:>5.1f}%)")
print(f"  6+ hours:    {(worker_periods['active_duration_hours'] >= 6).sum():>6,} ({(worker_periods['active_duration_hours'] >= 6).sum() / len(worker_periods) * 100:>5.1f}%)")

# Hourly distribution
print(f"\n📈 HOURLY DISTRIBUTION:")
print("=" * 80)
gps_hourly = gps_df.set_index('timestamp').resample('1H')['driver_id'].agg(['count', 'nunique'])
gps_hourly.columns = ['gps_pings', 'active_workers']

orders_hourly = orders_df.set_index('start_billing').resample('1H').size()
orders_hourly.name = 'orders'

combined = pd.concat([gps_hourly, orders_hourly], axis=1).fillna(0)
combined['ratio'] = combined['orders'] / combined['active_workers'].replace(0, 1)

print(f"\nHourly statistics:")
print(f"{'Hour':<20} {'Active Workers':<15} {'Orders':<12} {'Orders/Worker'}")
print("-" * 80)
for idx in combined.index[::4]:  # Show every 4th hour to keep output manageable
    row = combined.loc[idx]
    print(f"{str(idx):<20} {row['active_workers']:>14,.0f} {row['orders']:>11,.0f} {row['ratio']:>13.2f}")

# Find best 3-hour windows
print(f"\n🔍 FINDING BEST 3-HOUR WINDOWS FOR SAMPLING:")
print("=" * 80)

window_size = pd.Timedelta('3h')
min_time = min(gps_df['timestamp'].min(), orders_df['start_billing'].min())
max_time = max(gps_df['timestamp'].max(), orders_df['start_billing'].max())

best_windows = []
current = min_time
while current + window_size <= max_time:
    window_end = current + window_size
    
    # Count workers with GPS pings in this window
    workers_in_window = gps_df[
        (gps_df['timestamp'] >= current) & 
        (gps_df['timestamp'] <= window_end)
    ]['driver_id'].nunique()
    
    # Count orders in this window  
    orders_in_window = orders_df[
        (orders_df['start_billing'] >= current) & 
        (orders_df['start_billing'] <= window_end)
    ].shape[0]
    
    # Score: balance of both (prefer windows with enough of each)
    score = min(workers_in_window, 20000) + min(orders_in_window, 25000)
    
    best_windows.append({
        'start': current,
        'end': window_end,
        'workers': workers_in_window,
        'orders': orders_in_window,
        'score': score,
        'ratio': orders_in_window / workers_in_window if workers_in_window > 0 else 0
    })
    
    current += pd.Timedelta('30min')

# Sort by score
best_windows_sorted = sorted(best_windows, key=lambda x: x['score'], reverse=True)

print(f"\nTop 10 Best 3-Hour Windows:")
print("-" * 90)
print(f"{'Rank':<5} {'Start Time':<20} {'Workers':<12} {'Orders':<12} {'Orders/Worker':<15} {'Score'}")
print("-" * 90)
for i, window in enumerate(best_windows_sorted[:10], 1):
    print(f"{i:<5} {str(window['start']):<20} {window['workers']:<12,} {window['orders']:<12,} {window['ratio']:<15.2f} {window['score']:,}")

print("\n" + "=" * 80)
print("RECOMMENDATIONS:")
print("=" * 80)
best = best_windows_sorted[0]
print(f"✅ BEST WINDOW: {best['start']} to {best['end']}")
print(f"   Available: {best['workers']:,} workers, {best['orders']:,} orders")
print(f"   Ratio: {best['ratio']:.2f} orders per worker")
print(f"\n   Can support experiments:")
for worker_count in [2000, 4000, 6000, 8000, 10000, 12000, 15000]:
    if worker_count <= best['workers']:
        status = "✅" 
    else:
        status = "⚠️"
    print(f"   {status} {worker_count:>6,} workers: {'Sufficient' if worker_count <= best['workers'] else 'INSUFFICIENT (only ' + str(best['workers']) + ' available)'}")

print(f"\n   For 20,000 tasks: {'✅ Sufficient' if 20000 <= best['orders'] else '⚠️ INSUFFICIENT (only ' + str(best['orders']) + ' available)'}")
print("=" * 80)


#!/usr/bin/env python3
"""
Analyze temporal distribution of DiDi GPS and Order data.
Shows when workers are active vs when tasks arrive.
"""

import pandas as pd
import matplotlib.pyplot as plt
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

# Analyze worker availability (from GPS data)
# Group by driver to get their active periods
print(f"\n📊 Analyzing worker availability...")
worker_periods = gps_df.groupby('driver_id')['timestamp'].agg(['min', 'max']).reset_index()
worker_periods.columns = ['driver_id', 'first_seen', 'last_seen']
worker_periods['active_duration_hours'] = (worker_periods['last_seen'] - worker_periods['first_seen']).dt.total_seconds() / 3600

print(f"   Average worker shift: {worker_periods['active_duration_hours'].mean():.1f} hours")
print(f"   Median worker shift: {worker_periods['active_duration_hours'].median():.1f} hours")
print(f"   Workers active < 1 hour: {(worker_periods['active_duration_hours'] < 1).sum():,} ({(worker_periods['active_duration_hours'] < 1).sum() / len(worker_periods) * 100:.1f}%)")

# Create hourly distribution
print(f"\n📈 Creating temporal distribution plots...")

# Resample to hourly bins
gps_hourly = gps_df.set_index('timestamp').resample('1H')['driver_id'].agg(['count', 'nunique'])
gps_hourly.columns = ['gps_pings', 'active_workers']

orders_hourly = orders_df.set_index('start_billing').resample('1H').size()
orders_hourly.name = 'orders'

# Combine
combined = pd.concat([gps_hourly, orders_hourly], axis=1).fillna(0)

# Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle('Temporal Distribution: GPS Pings vs Orders', fontsize=16, fontweight='bold')

# Plot 1: Active workers over time
ax = axes[0]
ax.plot(combined.index, combined['active_workers'], linewidth=2, color='#3498db')
ax.set_ylabel('Unique Active Workers', fontsize=11, fontweight='bold')
ax.set_title('A. Worker Availability Over Time', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=2000, color='red', linestyle='--', alpha=0.5, label='2K workers')
ax.axhline(y=15000, color='green', linestyle='--', alpha=0.5, label='15K workers')
ax.legend()

# Plot 2: Order arrival rate
ax = axes[1]
ax.plot(combined.index, combined['orders'], linewidth=2, color='#e74c3c')
ax.set_ylabel('Orders per Hour', fontsize=11, fontweight='bold')
ax.set_title('B. Task Arrival Rate Over Time', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 3: Overlap ratio (orders per active worker)
ax = axes[2]
combined['ratio'] = combined['orders'] / combined['active_workers'].replace(0, 1)
ax.plot(combined.index, combined['ratio'], linewidth=2, color='#9b59b6')
ax.set_xlabel('Time', fontsize=11, fontweight='bold')
ax.set_ylabel('Orders per Active Worker', fontsize=11, fontweight='bold')
ax.set_title('C. Supply-Demand Ratio (Higher = More Tasks per Worker)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='10 tasks/worker (high scarcity)')
ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='1 task/worker (balanced)')
ax.legend()

plt.tight_layout()
plt.savefig('temporal_distribution_analysis.png', dpi=150, bbox_inches='tight')
print(f"✅ Plot saved to: temporal_distribution_analysis.png")
plt.show()

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
    
    # Score: balance of both
    score = min(workers_in_window, 20000) + min(orders_in_window, 25000)
    
    best_windows.append({
        'start': current,
        'end': window_end,
        'workers': workers_in_window,
        'orders': orders_in_window,
        'score': score
    })
    
    current += pd.Timedelta('30min')

# Sort by score
best_windows_df = pd.DataFrame(best_windows).sort_values('score', ascending=False).head(10)

print(f"\nTop 10 Best 3-Hour Windows:")
print("-" * 80)
print(f"{'Rank':<5} {'Start Time':<20} {'Workers':<10} {'Orders':<10} {'Orders/Worker':<15}")
print("-" * 80)
for i, (_, row) in enumerate(best_windows_df.iterrows(), 1):
    ratio = row['orders'] / row['workers'] if row['workers'] > 0 else 0
    print(f"{i:<5} {str(row['start']):<20} {row['workers']:<10,} {row['orders']:<10,} {ratio:<15.2f}")

print("\n" + "=" * 80)
print("RECOMMENDATIONS:")
print("=" * 80)
best = best_windows_df.iloc[0]
print(f"✅ Use window: {best['start']} to {best['end']}")
print(f"   Available: {best['workers']:,} workers, {best['orders']:,} orders")
print(f"   This ensures temporal overlap for all experiments!")
print("=" * 80)


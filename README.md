# sc_sim – Spatial-Crowdsourcing Simulator

A timestep-based simulator for my honours research: https://drive.google.com/file/d/15yfEliaieEv4Ulx9Z_bCm_OOsYGUIZOo/view?usp=sharing

---
## 1  Quick start
```bash

# (1) drop your dataset in ./data/<dataset>/
#     ├── didi/      → gps.txt  order.txt
#     ├── checkin/   → tasks.txt workers.txt - Look at /scripts/CheckinSynthesiser.py to syntehsise data from Gowalla and Weeplace Check-in Data
#     └── synthetic/ → workers.txt tasks.txt (unfinished)

# (2) configure simulator behaviour
vim config.py     # or use CLI flags later

# (3) run
python main.py              # uses settings from config.py
# override on-the-fly
python main.py --dataset checkin --root ./data/sample_checkins
```
Console output shows assignments and completions each tick; a `metrics_snapshot.csv` file appears after the run.

---

## 2  Dataset formats
### Didi Gaia (`dataset = "didi"`)
Place both files in the same directory:
```
order.txt : orderID,startBillingTime,endBillableTime,pickupLon,pickupLat,dropoffLon,dropoffLat[,human-readable-ts]
 gps.txt  : driverID,orderID,timestamp,lon,lat
```
Timestamps are Unix seconds.  A helper script is provided to create small subsets:
```
cd data/didi && python ../../scripts/sample_didi.py --span 30min
```
This will write `small_order.txt / small_gps.txt` alongside the originals.

### Check-ins (`dataset = "checkin"`)
Drop one or more CSV files with columns:
```
user_id,datetime,lat,lon,point_id
```
Header row optional.

### Synthetic
Two CSVs expected:
```
workers.txt : worker_id,start_lat,start_lon,release_time,deadline
tasks.txt   : task_id,pickup_lat,pickup_lon,dropoff_lat,dropoff_lon,release_time,expire_time
```
---

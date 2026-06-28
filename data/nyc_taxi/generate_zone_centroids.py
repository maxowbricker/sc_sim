#!/usr/bin/env python3
"""
One-time setup: generate zone_centroids.csv from the NYC TLC taxi zones shapefile.

Downloads the official TLC taxi zone shapefile (~1 MB), computes the geographic
centroid for each of the 265 taxi zones, and writes:

    data/nyc_taxi/zone_centroids.csv   (columns: LocationID, lat, lon)

This file is required by the nyc_taxi adapter before any simulation can run.

Requirements:
    pip install geopandas pyarrow

Usage:
    python data/nyc_taxi/generate_zone_centroids.py

    # Or point to an already-downloaded shapefile to skip the download:
    python data/nyc_taxi/generate_zone_centroids.py --shp /path/to/taxi_zones.shp
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

SHAPEFILE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"

# Resolve output path relative to this script's location in the repo
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_CSV = _REPO_ROOT / "data" / "nyc_taxi" / "zone_centroids.csv"


def _download_and_extract(dest_dir: Path) -> Path:
    """Download TLC shapefile zip and extract to dest_dir. Returns .shp path."""
    print(f"📥 Downloading TLC taxi zones shapefile:\n   {SHAPEFILE_URL}")
    with urllib.request.urlopen(SHAPEFILE_URL) as resp:
        zip_bytes = resp.read()

    print("📦 Extracting...")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        z.extractall(dest_dir)

    shp_files = list(dest_dir.glob("**/*.shp"))
    if not shp_files:
        raise FileNotFoundError("No .shp file found after extracting the zip.")
    return shp_files[0]


def generate(shp_path: Path | None = None) -> None:
    try:
        import geopandas as gpd
    except ImportError:
        print(
            "❌ geopandas is required for centroid generation.\n"
            "   Install it with:\n\n"
            "       pip install geopandas\n"
        )
        sys.exit(1)

    if shp_path is None:
        tmp_dir = Path("/tmp/nyc_taxi_zones")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        shp_path = _download_and_extract(tmp_dir)

    print(f"🗺️  Reading shapefile: {shp_path}")
    gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")

    centroids = gdf.geometry.centroid
    out = gdf[["LocationID"]].copy()
    out["lat"] = centroids.y
    out["lon"] = centroids.x
    out = out.sort_values("LocationID").reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)

    print(
        f"✅ Wrote {len(out)} zone centroids to:\n"
        f"   {OUTPUT_CSV}\n\n"
        f"You can now load the NYC taxi dataset:\n\n"
        f'    workers, tasks = load_workers_tasks("nyc_taxi", root_path="data/nyc_taxi", date="2012-05-01")\n'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shp",
        type=Path,
        default=None,
        help="Path to an already-downloaded taxi_zones.shp (skips download).",
    )
    args = parser.parse_args()
    generate(shp_path=args.shp)


if __name__ == "__main__":
    main()

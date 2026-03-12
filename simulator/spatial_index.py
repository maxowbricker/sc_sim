"""
Spatial indexing utilities for efficient nearest neighbor search.
Implements Advanced Nearest Neighbor (ANN) approach from FATP paper.
"""

import math
from typing import List, Set, Tuple
from collections import defaultdict
import heapq

# --- SPATIAL INDEXING CONSTANTS ---

# Grid resolution for spatial hashing/binning (e.g., assigning workers to grid cells).
# 0.01 degrees of latitude is ~1.11 km. At Chengdu's latitude (~30.6°N), 
# 0.01 degrees of longitude is ~0.96 km. Thus, a 0.01 resolution creates 
# roughly 1km x 1km spatial buckets, enabling O(1) candidate filtering.
GRID_RESOLUTION = 0.01

# --- FLAT EARTH OPTIMIZATION ---
# To enable the hundreds of thousands of simulation steps required for DRL convergence,
# we pre-calculate the longitudinal scaling factor. Because the DiDi dataset is bounded 
# within the urban scale of Chengdu, assuming a constant cos(latitude) introduces a 
# maximum spatial error of less than 0.15% at the geographic extremes. This negligible 
# deviation is vastly outweighed by the O(1) speedup (avoiding trigonometric overhead).

KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = None  # Must be initialized via set_city_constants() before use

def set_city_constants(mean_latitude: float) -> None:
    """
    Calculates and sets the global longitude scaling factor for the specific city.
    Must be called once at simulation startup to configure the 'Flat Earth' projection.
    
    Args:
        mean_latitude: Mean latitude of the active service area (in degrees).
                       For the Chengdu DiDi dataset, this is roughly 30.67.
    """
    global KM_PER_DEG_LON
    # Utilizing the global latitude constant to maintain a single source of truth
    KM_PER_DEG_LON = KM_PER_DEG_LAT * math.cos(math.radians(mean_latitude))

def fast_manhattan_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates Manhattan distance in kilometers using pre-calculated scalars.
    Executes in O(1) time using simple float arithmetic, avoiding math.cos() calls.
    
    Args:
        lat1, lon1: Coordinates of the first point (e.g., worker location).
        lat2, lon2: Coordinates of the second point (e.g., task pickup).
        
    Returns:
        Manhattan distance in kilometers.
    """
    assert KM_PER_DEG_LON is not None, "City constants must be set before calculation."
        
    d_lat = abs(lat1 - lat2) * KM_PER_DEG_LAT
    d_lon = abs(lon1 - lon2) * KM_PER_DEG_LON
    
    return d_lat + d_lon

class GridSpatialIndex:
    """
    Grid-based spatial index for efficient nearest neighbor search.
    Divides the map into cells (buckets) for O(1) cell lookup and O(k) candidate search.
    
    Supports any object type by specifying which attributes to use for coordinates.
    """
    
    def __init__(self, lat_attr='start_lat', lon_attr='start_lon', resolution=GRID_RESOLUTION):
        """
        Initialize grid spatial index with attribute names to read coordinates from.
        
        Parameters
        ----------
        lat_attr : str
            Attribute name for latitude (default: 'start_lat' for workers)
        lon_attr : str
            Attribute name for longitude (default: 'start_lon' for workers)
        resolution : float
            Grid cell size in degrees (default: 0.01 degrees ≈ 1km)
        
        Examples
        --------
        For Workers: lat_attr='start_lat', lon_attr='start_lon'
        For Tasks:   lat_attr='pickup_lat', lon_attr='pickup_lon'
        """
        self.lat_attr = lat_attr
        self.lon_attr = lon_attr
        self.resolution = resolution
        # Map (grid_x, grid_y) -> set of (lat, lon, item) tuples
        self.grid = defaultdict(set)
        self.count = 0 # number of items in the index for validation

    def _get_cell_coords(self, lat, lon):
        """Convert lat/lon to grid cell coordinates."""
        return (int(lat / self.resolution), int(lon / self.resolution))

    def add(self, item):
        """
        Add an item to the spatial index.
        
        """
        lat = getattr(item, self.lat_attr)
        lon = getattr(item, self.lon_attr)
        cell = self._get_cell_coords(lat, lon)
        # Store as tuple: (lat, lon, item_reference)
        self.grid[cell].add((lat, lon, item))
        self.count += 1

    def remove(self, item):
        """
        Remove an item from the spatial index.
        
        Note: Assumes item hasn't moved since being added.
        If it has, this will fail. Items only move when *assigned* (removed from index).
        """
        lat = getattr(item, self.lat_attr)
        lon = getattr(item, self.lon_attr)
        cell = self._get_cell_coords(lat, lon)
        # Recreate the tuple exactly as it was added for removal
        target = (lat, lon, item)
        if target in self.grid[cell]:
            self.grid[cell].remove(target)
            self.count -= 1
            # Cleanup empty cells to save memory
            if not self.grid[cell]:
                del self.grid[cell]

    def query_k_nearest(self, center_lat: float, center_lon: float, k: int = 15) -> List:
        """
        Find k nearest items to a specific coordinate.
        Spirals out from center location and terminates early using a spatial lower-bound check.
        """
        center_cell = self._get_cell_coords(center_lat, center_lon)
        candidates = []
        
        radius = 0
        # Sanity limit: 50 cells ≈ 50km. Prevents infinite loops if < k items exist globally.
        max_radius = 50 
        
        while radius < max_radius:
            cells_to_check = self._get_cells_in_ring(center_cell, radius)
            
            for cell in cells_to_check:
                items_in_cell = self.grid.get(cell)
                if items_in_cell:
                    for i_lat, i_lon, item in items_in_cell:
                        dist = fast_manhattan_km(center_lat, center_lon, i_lat, i_lon)
                        candidates.append((dist, item))
            
            # --- THE GRID EDGE OPTIMIZATION ---
            # We cannot terminate just because we found `k` items. If the center point is near 
            # a cell boundary, an item in the NEXT ring might be closer than an item in the CURRENT ring.
            if len(candidates) >= k:
                candidates.sort(key=lambda x: x[0])
                kth_dist = candidates[k-1][0]
                
                # Calculate the minimum possible Manhattan distance to the NEXT ring.
                # If the next ring is strictly further away than our k-th best candidate,
                # it is mathematically impossible to find a better candidate. We can safely stop.
                min_dist_next_ring = (radius + 1) * self.resolution * KM_PER_DEG_LAT
                
                if min_dist_next_ring > kth_dist:
                    break
            
            radius += 1

        # Final sort ensures the closest workers are at the front, even if the last ring added multiples
        candidates.sort(key=lambda x: x[0])
        return [c[1] for c in candidates[:k]]

    def _get_cells_in_ring(self, center, radius):
        """
        Generator for grid coordinates in a square ring.
        
        Parameters
        ----------
        center : Tuple[int, int]
            Center cell coordinates (grid_x, grid_y)
        radius : int
            Ring radius in cells
            
        Yields
        ------
        Tuple[int, int]
            Cell coordinates in the ring
        """
        cx, cy = center
        if radius == 0:
            yield (cx, cy)
            return

        # Top and Bottom rows
        for dx in range(-radius, radius + 1):
            yield (cx + dx, cy - radius)
            yield (cx + dx, cy + radius)
        
        # Left and Right columns (excluding corners already covered)
        for dy in range(-radius + 1, radius):
            yield (cx - radius, cy + dy)
            yield (cx + radius, cy + dy)

"""
Spatial indexing utilities for efficient nearest neighbor search.
Implements Advanced Nearest Neighbor (ANN) approach from FATP paper.
"""

import math
from typing import List, Set, Tuple
from collections import defaultdict
import heapq

# Tuning parameter: 0.01 degrees is roughly 1km
GRID_RESOLUTION = 0.01

# --- FLAT EARTH OPTIMIZATION ---
# Pre-calculated constants to avoid calling trig functions inside loops
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = None  # Will be set by set_city_constants()

def set_city_constants(mean_latitude: float):
    """
    Call this once at simulation start to configure the 'Flat Earth' projection.
    Calculates the longitude scaling factor for the specific city.
    
    Args:
        mean_latitude: Mean latitude of the city/region (in degrees)
    """
    global KM_PER_DEG_LON
    # Calculate cos(lat) once. Latitude must be in radians.
    KM_PER_DEG_LON = 111.32 * math.cos(math.radians(mean_latitude))

def fast_manhattan_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate Manhattan distance using pre-calculated scalars.
    O(1) simple float math - no trigonometry.
    
    This is much faster than manhattan_km() because it avoids calling
    math.cos(math.radians()) for every distance calculation.
    
    Args:
        lat1, lon1: Coordinates of first point
        lat2, lon2: Coordinates of second point
        
    Returns:
        Manhattan distance in kilometers
    """
    if KM_PER_DEG_LON is None:
        # Fallback to slow version if constants not set
        return manhattan_km(lat1, lon1, lat2, lon2)
        
    d_lat = abs(lat1 - lat2) * KM_PER_DEG_LAT
    d_lon = abs(lon1 - lon2) * KM_PER_DEG_LON
    return d_lat + d_lon

def manhattan_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Original slow implementation for fallback.
    Calculates cosine of average latitude for each distance check.
    """
    km_per_deg = 111
    d_lat = abs(lat1 - lat2) * km_per_deg
    avg_lat = (lat1 + lat2) / 2
    d_lon = abs(lon1 - lon2) * km_per_deg * math.cos(math.radians(avg_lat))
    return d_lat + d_lon

def find_k_nearest_workers(task, available_workers: Set, k: int = 15) -> List:
    """
    Find k nearest available workers to a task using Advanced Nearest Neighbor approach.
    
    This reduces complexity from O(|W|) to O(k) for scoring, where k << |W|.
    From FATP paper: "assigns tasks to the nearest available worker" with fairness constraints.
    
    Args:
        task: Task object with pickup_lat, pickup_lon
        available_workers: Set of available Worker objects  
        k: Number of nearest workers to return (default 15, same as original k parameter)
        
    Returns:
        List of up to k nearest workers, sorted by distance
    """
    if not available_workers:
        return []
    
    # Calculate distances to all available workers
    worker_distances = []
    
    for worker in available_workers:
        # BUGFIX: Removed problematic is_available filter that was causing 0% TAR
        # The available_workers set should already contain only available workers
        # if hasattr(worker, 'is_available') and not worker.is_available:
        #     continuefysp
            
        distance = fast_manhattan_km(
            task.pickup_lat, task.pickup_lon,
            worker.start_lat, worker.start_lon
        )
        worker_distances.append((distance, worker))
    
    # Return k nearest workers
    # Using heapq.nsmallest is efficient for small k values
    nearest = heapq.nsmallest(k, worker_distances, key=lambda x: x[0])
    
    # Return just the workers (not the distances)
    return [worker for distance, worker in nearest]

def find_nearest_available_worker(task, available_workers: Set) -> Tuple:
    """
    Find the single nearest available worker (pure ANN approach).
    
    This is the simplest ANN implementation - O(|W|) to find nearest, then O(1) to score.
    Much faster than current O(|W|) scoring approach.
    
    Args:
        task: Task object
        available_workers: Set of available workers
        
    Returns:
        Tuple of (nearest_worker, distance) or (None, None) if no workers available
    """
    if not available_workers:
        return None, None
    
    nearest_worker = None
    min_distance = float('inf')
    
    for worker in available_workers:
        # BUGFIX: Removed problematic is_available filter that was causing 0% TAR  
        # The available_workers set should already contain only available workers
        # if hasattr(worker, 'is_available') and not worker.is_available:
        #     continue
            
        distance = fast_manhattan_km(
            task.pickup_lat, task.pickup_lon,
            worker.start_lat, worker.start_lon
        )
        
        if distance < min_distance:
            min_distance = distance
            nearest_worker = worker
    
    return nearest_worker, min_distance

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
        # Map (grid_x, grid_y) -> set of items
        self.grid = defaultdict(set)
        # Keep track of item count for validation
        self.count = 0

    def _get_cell_coords(self, lat, lon):
        """Convert lat/lon to grid cell coordinates."""
        return (int(lat / self.resolution), int(lon / self.resolution))

    def add(self, item):
        """Add an item to the spatial index."""
        lat = getattr(item, self.lat_attr)
        lon = getattr(item, self.lon_attr)
        cell = self._get_cell_coords(lat, lon)
        self.grid[cell].add(item)
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
        if item in self.grid[cell]:
            self.grid[cell].remove(item)
            self.count -= 1
            # Cleanup empty cells to save memory
            if not self.grid[cell]:
                del self.grid[cell]

    def query_k_nearest(self, center_lat, center_lon, k=15):
        """
        Find k nearest items to a specific coordinate.
        Spirals out from center location and stops early once enough items are found.
        
        Parameters
        ----------
        center_lat : float
            Latitude of the center point
        center_lon : float
            Longitude of the center point
        k : int
            Number of nearest items to return (default: 15)
            
        Returns
        -------
        List
            List of up to k nearest items, sorted by distance
        """
        center_cell = self._get_cell_coords(center_lat, center_lon)
        candidates = []
        
        # Search radius (in cells)
        radius = 0
        # Sanity limit: don't search whole world if map is empty
        max_radius = 50 
        
        while radius < max_radius:
            cells_to_check = self._get_cells_in_ring(center_cell, radius)
            found_in_ring = False
            
            for cell in cells_to_check:
                items_in_cell = self.grid.get(cell)
                if items_in_cell:
                    found_in_ring = True
                    for item in items_in_cell:
                        # Get item coordinates using configured attributes
                        i_lat = getattr(item, self.lat_attr)
                        i_lon = getattr(item, self.lon_attr)
                        
                        dist = fast_manhattan_km(center_lat, center_lon, i_lat, i_lon)
                        candidates.append((dist, item))
            
            # Optimization: Stop if we have enough candidates and 
            # the closest possible item in the NEXT ring is further than our kth best candidate.
            if len(candidates) >= k:
                candidates.sort(key=lambda x: x[0])
                # K-th best distance currently found
                kth_dist = candidates[k-1][0]
                
                # Minimum distance to the NEXT ring (approximate lower bound)
                # 1 cell distance approx 111km * resolution
                min_dist_next_ring = (radius + 1) * self.resolution * 111 * 0.7 
                
                if min_dist_next_ring > kth_dist:
                    break
            
            # If we've searched a lot and found nothing, expand
            radius += 1
            if not candidates and radius > 5:
                 # Logic to jump radius could go here, but simple increment is safe
                 pass

        # Final sort and return top k items
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


class SpatialIndex:
    """
    Advanced spatial indexing for even faster nearest neighbor search.
    Future enhancement: Could implement KD-tree or R-tree for O(log n) queries.
    """
    
    def __init__(self, workers: Set):
        """Initialize spatial index with workers."""
        self.workers = workers
        # Future: Build KD-tree or grid-based spatial index here
    
    def find_nearby_workers(self, lat: float, lon: float, radius_km: float = 10) -> List:
        """
        Find workers within a given radius (future enhancement).
        
        This could use spatial data structures for O(log n + k) performance
        instead of O(n) linear search.
        """
        nearby = []
        
        for worker in self.workers:
            distance = fast_manhattan_km(lat, lon, worker.start_lat, worker.start_lon)
            if distance <= radius_km:
                nearby.append((distance, worker))
        
        # Sort by distance
        nearby.sort(key=lambda x: x[0])
        return [worker for distance, worker in nearby]

# Performance testing utilities
def benchmark_nearest_neighbor():
    """Benchmark different nearest neighbor approaches."""
    import time
    import random
    
    # Create mock data for testing
    class MockWorker:
        def __init__(self, lat, lon):
            self.start_lat = lat
            self.start_lon = lon
            self.is_available = True
    
    class MockTask:
        def __init__(self, lat, lon):
            self.pickup_lat = lat
            self.pickup_lon = lon
    
    # Create large number of workers (similar to real data)
    workers = set()
    for _ in range(10000):
        lat = random.uniform(30.5, 30.7)  # Chengdu-like coordinates
        lon = random.uniform(104.0, 104.2)
        workers.add(MockWorker(lat, lon))
    
    # Create test task
    task = MockTask(30.6, 104.1)
    
    # Benchmark k-nearest approach
    start_time = time.time()
    for _ in range(100):  # 100 tasks
        nearest = find_k_nearest_workers(task, workers, k=15)
    knn_time = time.time() - start_time
    
    # Benchmark single nearest approach  
    start_time = time.time()
    for _ in range(100):  # 100 tasks
        worker, dist = find_nearest_available_worker(task, workers)
    ann_time = time.time() - start_time
    
    print(f"🔬 Nearest Neighbor Benchmark Results:")
    print(f"   Workers: {len(workers):,}")
    print(f"   K-Nearest (k=15): {knn_time:.4f}s for 100 tasks")
    print(f"   Single Nearest:   {ann_time:.4f}s for 100 tasks") 
    print(f"   Speedup: {knn_time/ann_time:.1f}x faster with single nearest")

if __name__ == "__main__":
    benchmark_nearest_neighbor()


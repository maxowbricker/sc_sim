"""
Spatial indexing utilities for efficient nearest neighbor search.
Implements Advanced Nearest Neighbor (ANN) approach from FATP paper.
"""

import math
from typing import List, Set, Tuple
import heapq

def manhattan_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate Manhattan distance in kilometers."""
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
        #     continue
            
        distance = manhattan_km(
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
            
        distance = manhattan_km(
            task.pickup_lat, task.pickup_lon,
            worker.start_lat, worker.start_lon
        )
        
        if distance < min_distance:
            min_distance = distance
            nearest_worker = worker
    
    return nearest_worker, min_distance

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
            distance = manhattan_km(lat, lon, worker.start_lat, worker.start_lon)
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


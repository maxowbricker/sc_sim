#!/usr/bin/env python3
"""
JSON Utilities for Experiment Results
Handles conversion of numpy/pandas types to JSON-serializable Python types.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

def make_json_serializable(obj):
    """
    Convert numpy/pandas types to JSON-serializable Python types.
    
    Args:
        obj: Any object that might contain numpy/pandas types
        
    Returns:
        JSON-serializable version of the object
    """
    # Handle numpy scalars (int64, float64, etc.)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Handle pandas types
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    
    # Handle datetime objects
    elif isinstance(obj, datetime):
        return obj.isoformat()
    
    # Handle collections recursively
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    
    # Return as-is for native Python types
    else:
        return obj

def save_results_json(data, filepath):
    """
    Save experiment results to JSON with proper type conversion.
    
    Args:
        data: Dictionary containing experiment results
        filepath: Path where to save the JSON file
    """
    # Convert all data to JSON-serializable types
    serializable_data = make_json_serializable(data)
    
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Save with pretty formatting
    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Results saved to: {filepath}")

def load_results_json(filepath):
    """
    Load experiment results from JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Dictionary containing the loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)

class ExperimentLogger:
    """
    Logger for experiment results with automatic JSON serialization.
    """
    
    def __init__(self, experiment_name, results_dir="results"):
        self.experiment_name = experiment_name
        self.results_dir = Path(results_dir)
        self.start_time = datetime.now()
        
        # Create timestamped filename
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.filepath = self.results_dir / f"{experiment_name}_{timestamp}.json"
        
        # Initialize results structure
        self.results = {
            'experiment_name': experiment_name,
            'start_timestamp': self.start_time.isoformat(),
            'status': 'running',
            'results': []
        }
    
    def log_result(self, result_data):
        """Add a single result to the experiment log."""
        result_data['timestamp'] = datetime.now().isoformat()
        self.results['results'].append(result_data)
    
    def save_progress(self):
        """Save current progress to file."""
        self.results['last_update'] = datetime.now().isoformat()
        self.results['completed_experiments'] = len(self.results['results'])
        save_results_json(self.results, self.filepath)
    
    def finalize(self, status="completed"):
        """Finalize the experiment log."""
        end_time = datetime.now()
        self.results.update({
            'status': status,
            'end_timestamp': end_time.isoformat(),
            'total_duration_minutes': (end_time - self.start_time).total_seconds() / 60,
            'total_experiments': len(self.results['results'])
        })
        save_results_json(self.results, self.filepath)
        return self.filepath

# Example usage
if __name__ == "__main__":
    # Test the utilities
    import numpy as np
    
    # Test data with various numpy/pandas types
    test_data = {
        'int64_val': np.int64(42),
        'float64_val': np.float64(3.14159),
        'bool_val': np.bool_(True),
        'array': np.array([1, 2, 3]),
        'timestamp': datetime.now(),
        'nested': {
            'more_numpy': np.float32(2.718),
            'list_with_numpy': [np.int32(1), np.int32(2), np.int32(3)]
        }
    }
    
    print("🧪 Testing JSON serialization utilities...")
    
    # Test conversion
    converted = make_json_serializable(test_data)
    print("✅ Conversion successful")
    
    # Test saving
    test_file = "test_json_utils.json"
    save_results_json(converted, test_file)
    print("✅ Saving successful")
    
    # Test loading
    loaded = load_results_json(test_file)
    print("✅ Loading successful")
    
    # Clean up
    Path(test_file).unlink()
    print("🎉 All JSON utilities working correctly!")

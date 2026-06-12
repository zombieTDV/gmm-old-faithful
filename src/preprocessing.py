"""
preprocessing.py — Manual data cleaning and standardization pipeline.

Implements z-score normalization from scratch: x' = (x - mean) / std
No pandas, no sklearn preprocessing — pure Python + NumPy arrays.
"""
import numpy as np
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.data_loader import load_csv


def clean_data(raw_data):
    """
    Remove invalid rows (NaN, negative, or zero values).
    
    Old Faithful data should have:
    - eruptions > 0 (duration in minutes)
    - waiting > 0 (wait time in minutes)
    
    Args:
        raw_data (list): List of [eruptions, waiting] pairs.
        
    Returns:
        numpy.ndarray: Cleaned data array of shape (N, 2).
    """
    cleaned = []
    removed_count = 0
    
    for row in raw_data:
        eruptions, waiting = row[0], row[1]
        
        # Check for valid numeric values
        if eruptions > 0 and waiting > 0:
            cleaned.append([eruptions, waiting])
        else:
            removed_count += 1
    
    if removed_count > 0:
        print(f"  Removed {removed_count} invalid rows")
    
    print(f"  Clean data: {len(cleaned)} rows")
    return np.array(cleaned)


def compute_mean(data):
    """
    Compute column-wise mean manually.
    
    Formula: mean = (1/N) * sum(x_i)
    
    Args:
        data (numpy.ndarray): Data array of shape (N, D).
        
    Returns:
        numpy.ndarray: Mean vector of shape (D,).
    """
    n_samples = data.shape[0]
    mean = np.zeros(data.shape[1])
    for i in range(n_samples):
        mean += data[i]
    return mean / n_samples


def compute_std(data, mean):
    """
    Compute column-wise standard deviation manually.
    
    Formula: std = sqrt((1/N) * sum((x_i - mean)^2))
    
    Args:
        data (numpy.ndarray): Data array of shape (N, D).
        mean (numpy.ndarray): Mean vector of shape (D,).
        
    Returns:
        numpy.ndarray: Standard deviation vector of shape (D,).
    """
    n_samples = data.shape[0]
    variance = np.zeros(data.shape[1])
    for i in range(n_samples):
        diff = data[i] - mean
        variance += diff * diff
    variance = variance / n_samples
    return np.sqrt(variance)


def standardize(data):
    """
    Apply z-score standardization: x' = (x - mean) / std
    
    This transforms each feature to have mean=0 and std=1.
    Essential for GMM because:
    - Features on different scales would dominate the Euclidean distance
    - Covariance matrix becomes ill-conditioned with mixed scales
    - EM convergence is faster with normalized features
    
    Args:
        data (numpy.ndarray): Raw data of shape (N, D).
        
    Returns:
        tuple: (standardized_data, mean, std) for inverse transform.
    """
    mean = compute_mean(data)
    std = compute_std(data, mean)
    
    print(f"  Feature means: eruptions={mean[0]:.4f}, waiting={mean[1]:.4f}")
    print(f"  Feature stds:  eruptions={std[0]:.4f}, waiting={std[1]:.4f}")
    
    standardized = (data - mean) / std
    
    return standardized, mean, std


def save_csv(data, filepath, header="eruptions,waiting"):
    """
    Save numpy array to CSV file using manual file I/O.
    
    Args:
        data (numpy.ndarray): Data array of shape (N, 2).
        filepath (str): Output file path.
        header (str): CSV header line.
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write(header + '\n')
        for row in data:
            f.write(f"{row[0]:.6f},{row[1]:.6f}\n")
    
    print(f"  Saved {len(data)} rows to {filepath}")


def run_pipeline():
    """
    Execute the full preprocessing pipeline:
    1. Load raw CSV
    2. Clean invalid rows
    3. Standardize features (z-score)
    4. Save processed data
    
    Returns:
        tuple: (standardized_data, mean, std)
    """
    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE")
    print("="*60)
    
    # Step 1: Load
    print("\n[Step 1] Loading raw data...")
    raw_data = load_csv(RAW_DATA_PATH)
    
    # Step 2: Clean
    print("\n[Step 2] Cleaning data...")
    clean = clean_data(raw_data)
    
    # Step 3: Standardize
    print("\n[Step 3] Standardizing features (z-score)...")
    standardized, mean, std = standardize(clean)
    
    # Step 4: Save
    print("\n[Step 4] Saving processed data...")
    save_csv(standardized, PROCESSED_DATA_PATH)
    
    print("\n[DONE] Preprocessing complete.")
    return standardized, mean, std


if __name__ == "__main__":
    run_pipeline()

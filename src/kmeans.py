"""
kmeans.py — K-Means clustering algorithm from scratch.

K-Means partitions data into K clusters using hard assignment:
1. Initialize K centroids (random data points)
2. Assign each point to nearest centroid (Euclidean distance)
3. Recompute centroids as cluster means
4. Repeat until convergence

Compared to GMM: K-Means is a special case where clusters are spherical
and assignments are hard (0 or 1) instead of probabilistic.
"""
import numpy as np


def euclidean_distance(a, b):
    """
    Compute Euclidean distance between two points.
    
    Formula: d(a,b) = sqrt(sum_i (a_i - b_i)^2)
    
    Args:
        a (numpy.ndarray): First point of shape (D,).
        b (numpy.ndarray): Second point of shape (D,).
        
    Returns:
        float: Euclidean distance.
    """
    diff = a - b
    return np.sqrt(np.sum(diff * diff))


def assign_clusters(X, centroids):
    """
    Assign each data point to the nearest centroid.
    
    Hard assignment: each point belongs to exactly one cluster.
    This is the key difference from GMM's soft (probabilistic) assignment.
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        centroids (numpy.ndarray): Centroids of shape (K, D).
        
    Returns:
        numpy.ndarray: Cluster labels of shape (N,).
    """
    n_samples = X.shape[0]
    K = centroids.shape[0]
    labels = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        min_dist = np.inf
        for k in range(K):
            dist = euclidean_distance(X[i], centroids[k])
            if dist < min_dist:
                min_dist = dist
                labels[i] = k
    
    return labels


def update_centroids(X, labels, K):
    """
    Recompute centroids as the mean of assigned points.
    
    Formula: c_k = (1/|C_k|) * sum_{x in C_k} x
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        labels (numpy.ndarray): Cluster assignments of shape (N,).
        K (int): Number of clusters.
        
    Returns:
        numpy.ndarray: Updated centroids of shape (K, D).
    """
    n_features = X.shape[1]
    centroids = np.zeros((K, n_features))
    
    for k in range(K):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            centroids[k] = np.mean(cluster_points, axis=0)
    
    return centroids


def fit_kmeans(X, K, max_iters=100, seed=42):
    """
    Run the K-Means algorithm.
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        K (int): Number of clusters.
        max_iters (int): Maximum iterations.
        seed (int): Random seed.
        
    Returns:
        tuple: (labels, centroids)
    """
    rng = np.random.RandomState(seed)
    n_samples = X.shape[0]
    
    # Initialize: pick K random data points as centroids
    indices = rng.choice(n_samples, size=K, replace=False)
    centroids = X[indices].copy()
    
    for iteration in range(1, max_iters + 1):
        # Assign points to nearest centroid
        labels = assign_clusters(X, centroids)
        
        # Recompute centroids
        new_centroids = update_centroids(X, labels, K)
        
        # Check convergence: centroids stopped moving
        shift = np.sum([euclidean_distance(centroids[k], new_centroids[k]) 
                       for k in range(K)])
        
        centroids = new_centroids
        
        if shift < 1e-6:
            print(f"    K-Means converged at iteration {iteration}")
            break
    
    return labels, centroids

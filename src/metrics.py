"""
metrics.py — Evaluation metrics implemented from scratch.

All metrics are computed manually without scikit-learn or scipy.

Metrics:
1. Log-Likelihood: How well the GMM fits the data (higher = better)
2. Silhouette Score: Cluster cohesion vs separation (-1 to 1, higher = better)
3. Cluster Separation: Mean distance between cluster centroids (higher = more separated)
"""
import numpy as np


def euclidean_distance(a, b):
    """
    Compute Euclidean distance between two points.
    
    Args:
        a (numpy.ndarray): Point of shape (D,).
        b (numpy.ndarray): Point of shape (D,).
        
    Returns:
        float: Euclidean distance.
    """
    diff = a - b
    return np.sqrt(np.sum(diff * diff))


def compute_silhouette_sample(X, labels, point_idx):
    """
    Compute silhouette score for a single data point.
    
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    
    Where:
    - a(i) = mean distance to other points in the SAME cluster (cohesion)
    - b(i) = min mean distance to points in ANOTHER cluster (separation)
    
    Interpretation:
    - s ≈ 1: point is well-matched to its cluster
    - s ≈ 0: point is on the boundary between clusters
    - s ≈ -1: point is likely misassigned
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        labels (numpy.ndarray): Cluster labels of shape (N,).
        point_idx (int): Index of the point to evaluate.
        
    Returns:
        float: Silhouette score for this point.
    """
    point_label = labels[point_idx]
    unique_labels = np.unique(labels)
    
    # a(i): mean distance to same-cluster points
    same_cluster = X[labels == point_label]
    if len(same_cluster) <= 1:
        return 0.0  # Single-point cluster
    
    a_i = 0.0
    count_same = 0
    for j in range(len(X)):
        if j != point_idx and labels[j] == point_label:
            a_i += euclidean_distance(X[point_idx], X[j])
            count_same += 1
    a_i /= count_same
    
    # b(i): minimum mean distance to any OTHER cluster
    b_i = np.inf
    for label in unique_labels:
        if label == point_label:
            continue
        
        other_points = X[labels == label]
        mean_dist = 0.0
        for j in range(len(X)):
            if labels[j] == label:
                mean_dist += euclidean_distance(X[point_idx], X[j])
        mean_dist /= len(other_points)
        
        if mean_dist < b_i:
            b_i = mean_dist
    
    # Silhouette score
    max_ab = max(a_i, b_i)
    if max_ab == 0:
        return 0.0
    
    return (b_i - a_i) / max_ab


def silhouette_score(X, labels):
    """
    Compute mean silhouette score across all data points.
    
    Range: [-1, 1]
    - Near 1: Dense, well-separated clusters
    - Near 0: Overlapping clusters  
    - Negative: Possible misassignment
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        labels (numpy.ndarray): Cluster labels of shape (N,).
        
    Returns:
        float: Mean silhouette score.
    """
    n_samples = X.shape[0]
    scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        scores[i] = compute_silhouette_sample(X, labels, i)
    
    mean_score = np.mean(scores)
    return mean_score


def cluster_separation(X, labels):
    """
    Compute mean inter-cluster centroid distance.
    
    Measures how far apart the cluster centers are.
    Higher = more separated clusters.
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        labels (numpy.ndarray): Cluster labels of shape (N,).
        
    Returns:
        float: Mean distance between all pairs of centroids.
    """
    unique_labels = np.unique(labels)
    K = len(unique_labels)
    
    # Compute centroids
    centroids = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
    
    # Mean pairwise distance between centroids
    total_dist = 0.0
    n_pairs = 0
    for i in range(K):
        for j in range(i + 1, K):
            total_dist += euclidean_distance(centroids[i], centroids[j])
            n_pairs += 1
    
    if n_pairs == 0:
        return 0.0
    
    return total_dist / n_pairs

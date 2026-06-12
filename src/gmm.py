"""
gmm.py — Gaussian Mixture Model parameter representation.

A GMM models data as a weighted sum of K Gaussian distributions:
    p(x) = sum_{k=1}^{K} pi_k * N(x | mu_k, Sigma_k)

Parameters:
    - weights (pi): mixing coefficients, sum to 1
    - means (mu): center of each Gaussian component
    - covariances (Sigma): spread/shape of each component
"""
import numpy as np


class GMMParams:
    """
    Container for GMM parameters.
    
    Attributes:
        K (int): Number of mixture components.
        weights (numpy.ndarray): Mixing weights of shape (K,), sum to 1.
        means (numpy.ndarray): Component means of shape (K, D).
        covariances (numpy.ndarray): Component covariances of shape (K, D, D).
    """
    
    def __init__(self, K, weights, means, covariances):
        self.K = K
        self.weights = weights
        self.means = means
        self.covariances = covariances
    
    def __str__(self):
        """Pretty-print GMM parameters."""
        lines = [f"GMM with K={self.K} components:"]
        for k in range(self.K):
            lines.append(f"\n  Component {k+1}:")
            lines.append(f"    Weight (pi):  {self.weights[k]:.4f}")
            lines.append(f"    Mean (mu):    {self.means[k]}")
            lines.append(f"    Covariance:")
            for row in self.covariances[k]:
                lines.append(f"      {row}")
        return "\n".join(lines)


def initialize_random(X, K, seed):
    """
    Random initialization: pick K random data points as means.
    
    - Weights: uniform (1/K each)
    - Means: K randomly selected data points
    - Covariances: identity matrices
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        K (int): Number of components.
        seed (int): Random seed for reproducibility.
        
    Returns:
        GMMParams: Initialized parameters.
    """
    rng = np.random.RandomState(seed)
    n_samples, n_features = X.shape
    
    # Uniform weights
    weights = np.ones(K) / K
    
    # Random data points as initial means
    indices = rng.choice(n_samples, size=K, replace=False)
    means = X[indices].copy()
    
    # Identity covariances
    covariances = np.array([np.eye(n_features) for _ in range(K)])
    
    return GMMParams(K, weights, means, covariances)


def initialize_kmeans(X, K, seed, max_iters=100):
    """
    KMeans-based initialization for GMM.
    
    Run KMeans first, then use:
    - Weights: proportion of points in each cluster
    - Means: KMeans centroids
    - Covariances: sample covariance of each cluster
    
    This gives EM a much better starting point than random init,
    typically reducing iterations from ~40 to ~15 on this dataset.
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        K (int): Number of components.
        seed (int): Random seed.
        max_iters (int): Max KMeans iterations.
        
    Returns:
        GMMParams: Initialized parameters.
    """
    # Import here to avoid circular dependency
    from src.kmeans import fit_kmeans
    
    labels, centroids = fit_kmeans(X, K, max_iters, seed)
    n_samples, n_features = X.shape
    
    weights = np.zeros(K)
    means = centroids.copy()
    covariances = np.zeros((K, n_features, n_features))
    
    for k in range(K):
        # Points assigned to cluster k
        cluster_points = X[labels == k]
        n_k = len(cluster_points)
        
        # Weight = fraction of points in this cluster
        weights[k] = n_k / n_samples
        
        # Covariance of this cluster's points
        if n_k > 1:
            diff = cluster_points - means[k]
            covariances[k] = np.dot(diff.T, diff) / n_k
        else:
            covariances[k] = np.eye(n_features)
        
        # Add small regularization
        covariances[k] += 1e-6 * np.eye(n_features)
    
    return GMMParams(K, weights, means, covariances)


def initialize_params(X, K, method="kmeans", seed=42):
    """
    Initialize GMM parameters using the specified method.
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        K (int): Number of components.
        method (str): "kmeans" or "random".
        seed (int): Random seed.
        
    Returns:
        GMMParams: Initialized parameters.
    """
    print(f"  Initialization method: {method}")
    
    if method == "kmeans":
        return initialize_kmeans(X, K, seed)
    elif method == "random":
        return initialize_random(X, K, seed)
    else:
        raise ValueError(f"Unknown init method: {method}. Use 'kmeans' or 'random'.")

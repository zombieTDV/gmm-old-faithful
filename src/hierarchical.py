"""
hierarchical.py — Agglomerative Hierarchical Clustering from scratch.

Bottom-up approach:
1. Start with each point as its own cluster
2. Repeatedly merge the two closest clusters
3. Stop when K clusters remain

Linkage methods determine how "closeness" between clusters is measured:
- Single linkage: min distance between any pair of points
- Average linkage: mean distance between all pairs of points

Compared to GMM: hierarchical clustering makes no distributional assumptions
but produces only hard assignments (no probabilities).
"""
import numpy as np


def compute_distance_matrix(X):
    """
    Compute pairwise Euclidean distance matrix.
    
    D[i,j] = ||x_i - x_j||_2
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        
    Returns:
        numpy.ndarray: Distance matrix of shape (N, N).
    """
    n_samples = X.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            diff = X[i] - X[j]
            dist = np.sqrt(np.sum(diff * diff))
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # Symmetric
    
    return dist_matrix


def find_closest_clusters(dist_matrix, active_clusters):
    """
    Find the two closest clusters to merge.
    
    Args:
        dist_matrix (numpy.ndarray): Inter-cluster distances.
        active_clusters (list): List of currently active cluster indices.
        
    Returns:
        tuple: (cluster_i, cluster_j, min_distance)
    """
    min_dist = np.inf
    merge_i, merge_j = -1, -1
    
    for idx_a in range(len(active_clusters)):
        for idx_b in range(idx_a + 1, len(active_clusters)):
            i = active_clusters[idx_a]
            j = active_clusters[idx_b]
            if dist_matrix[i, j] < min_dist:
                min_dist = dist_matrix[i, j]
                merge_i = i
                merge_j = j
    
    return merge_i, merge_j, min_dist


def update_distance_matrix_average(dist_matrix, cluster_members, 
                                    merged_id, removed_id, active_clusters):
    """
    Update distances after merging two clusters (average linkage).
    
    Average linkage: distance between clusters A and B is the mean
    of all pairwise distances between points in A and points in B.
    
    After merging clusters i and j into i, update distances from
    the merged cluster to all other active clusters.
    
    Args:
        dist_matrix (numpy.ndarray): Distance matrix to update in-place.
        cluster_members (dict): Maps cluster ID to list of member indices.
        merged_id (int): ID of the cluster that absorbed the other.
        removed_id (int): ID of the cluster that was absorbed.
        active_clusters (list): Currently active cluster IDs.
    """
    members_merged = cluster_members[merged_id]
    
    for other in active_clusters:
        if other == merged_id:
            continue
        
        members_other = cluster_members[other]
        
        # Average distance between all pairs
        total_dist = 0.0
        n_pairs = len(members_merged) * len(members_other)
        
        for m in members_merged:
            for o in members_other:
                diff = m - o  # These are actual point coordinates
                total_dist += np.sqrt(np.sum(diff * diff))
        
        avg_dist = total_dist / n_pairs if n_pairs > 0 else np.inf
        dist_matrix[merged_id, other] = avg_dist
        dist_matrix[other, merged_id] = avg_dist


def fit_agglomerative(X, K, linkage="average"):
    """
    Run agglomerative hierarchical clustering.
    
    Algorithm:
    1. Initialize: each point is its own cluster
    2. Compute pairwise distance matrix
    3. Repeat until K clusters remain:
       a. Find two closest clusters
       b. Merge them
       c. Update distance matrix
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        K (int): Target number of clusters.
        linkage (str): Linkage method ("average" or "single").
        
    Returns:
        numpy.ndarray: Cluster labels of shape (N,).
    """
    n_samples = X.shape[0]
    
    # Initialize: each point is its own cluster
    # cluster_members maps cluster_id -> list of point coordinates
    cluster_members = {i: [X[i].copy()] for i in range(n_samples)}
    point_to_cluster = list(range(n_samples))  # which cluster each point belongs to
    active_clusters = list(range(n_samples))
    
    # Initial pairwise distance matrix
    dist_matrix = compute_distance_matrix(X)
    # Set diagonal to infinity so we don't merge a cluster with itself
    np.fill_diagonal(dist_matrix, np.inf)
    
    n_merges = n_samples - K
    
    for step in range(n_merges):
        # Find closest pair of clusters
        merge_i, merge_j, min_dist = find_closest_clusters(
            dist_matrix, active_clusters
        )
        
        # Merge j into i
        cluster_members[merge_i].extend(cluster_members[merge_j])
        
        # Update point-to-cluster mapping
        for idx in range(n_samples):
            if point_to_cluster[idx] == merge_j:
                point_to_cluster[idx] = merge_i
        
        # Remove j from active clusters
        active_clusters.remove(merge_j)
        
        # Invalidate distances for removed cluster
        dist_matrix[merge_j, :] = np.inf
        dist_matrix[:, merge_j] = np.inf
        
        # Update distances for merged cluster
        update_distance_matrix_average(
            dist_matrix, cluster_members,
            merge_i, merge_j, active_clusters
        )
    
    # Convert cluster IDs to sequential labels 0, 1, ..., K-1
    unique_clusters = sorted(set(point_to_cluster))
    cluster_map = {old: new for new, old in enumerate(unique_clusters)}
    labels = np.array([cluster_map[c] for c in point_to_cluster])
    
    return labels

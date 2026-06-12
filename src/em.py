"""
em.py — Expectation-Maximization algorithm for GMM optimization.

The EM algorithm iteratively optimizes GMM parameters by alternating between:
    E-step: Compute responsibilities (posterior probability of each component)
    M-step: Update parameters to maximize expected log-likelihood

EM is guaranteed to monotonically increase log-likelihood at each iteration,
converging to a local maximum.
"""
import numpy as np
from src.gaussian import gaussian_pdf_batch
from src.gmm import GMMParams, initialize_params


def e_step(X, params):
    """
    E-step: Compute responsibilities (posterior probabilities).
    
    For each data point x_n and component k:
        gamma(n,k) = pi_k * N(x_n | mu_k, Sigma_k)
                     ------------------------------------
                     sum_j pi_j * N(x_n | mu_j, Sigma_j)
    
    gamma(n,k) represents "how much does component k explain point n?"
    Each row sums to 1 (probabilities over components).
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        params (GMMParams): Current GMM parameters.
        
    Returns:
        numpy.ndarray: Responsibilities of shape (N, K).
    """
    n_samples = X.shape[0]
    K = params.K
    
    # Compute weighted PDF for each component
    weighted_pdfs = np.zeros((n_samples, K))
    for k in range(K):
        pdf_values = gaussian_pdf_batch(X, params.means[k], params.covariances[k])
        weighted_pdfs[:, k] = params.weights[k] * pdf_values
    
    # Normalize: each row sums to 1
    row_sums = np.sum(weighted_pdfs, axis=1, keepdims=True)
    
    # Prevent division by zero
    row_sums = np.maximum(row_sums, 1e-300)
    
    responsibilities = weighted_pdfs / row_sums
    
    return responsibilities


def m_step(X, responsibilities, reg_covar):
    """
    M-step: Update GMM parameters using current responsibilities.
    
    For each component k:
        N_k = sum_n gamma(n,k)                           (effective count)
        mu_k = (1/N_k) * sum_n gamma(n,k) * x_n          (weighted mean)
        Sigma_k = (1/N_k) * sum_n gamma(n,k) * (x_n - mu_k)(x_n - mu_k)^T
        pi_k = N_k / N                                    (mixing weight)
    
    Covariance regularization (+ epsilon*I) prevents singularity.
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        responsibilities (numpy.ndarray): Responsibilities of shape (N, K).
        reg_covar (float): Covariance regularization term.
        
    Returns:
        GMMParams: Updated parameters.
    """
    n_samples, n_features = X.shape
    K = responsibilities.shape[1]
    
    weights = np.zeros(K)
    means = np.zeros((K, n_features))
    covariances = np.zeros((K, n_features, n_features))
    
    for k in range(K):
        # Effective number of points assigned to component k
        N_k = np.sum(responsibilities[:, k])
        
        # Prevent division by zero
        if N_k < 1e-32:
            N_k = 1e-32
        
        # Update weight: pi_k = N_k / N
        weights[k] = N_k / n_samples
        
        # Update mean: mu_k = sum(gamma * x) / N_k
        means[k] = np.zeros(n_features)
        for n in range(n_samples):
            means[k] += responsibilities[n, k] * X[n]
        means[k] /= N_k
        
        # Update covariance: Sigma_k = sum(gamma * (x-mu)(x-mu)^T) / N_k
        for n in range(n_samples):
            diff = (X[n] - means[k]).reshape(-1, 1)
            covariances[k] += responsibilities[n, k] * np.dot(diff, diff.T)
        covariances[k] /= N_k
        
        # Regularization: add epsilon * I to prevent singular matrix
        covariances[k] += reg_covar * np.eye(n_features)
    
    return GMMParams(K, weights, means, covariances)


def compute_log_likelihood(X, params):
    """
    Compute the log-likelihood of data under the current GMM.
    
    log L = sum_n log( sum_k pi_k * N(x_n | mu_k, Sigma_k) )
    
    Higher is better. EM guarantees this increases each iteration.
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        params (GMMParams): Current GMM parameters.
        
    Returns:
        float: Total log-likelihood.
    """
    n_samples = X.shape[0]
    K = params.K
    
    # Compute mixture density for each point
    mixture_density = np.zeros(n_samples)
    for k in range(K):
        pdf_values = gaussian_pdf_batch(X, params.means[k], params.covariances[k])
        mixture_density += params.weights[k] * pdf_values
    
    # Prevent log(0)
    mixture_density = np.maximum(mixture_density, 1e-300)
    
    log_likelihood = np.sum(np.log(mixture_density))
    
    return log_likelihood


def fit_gmm(X, K, max_iters, tol, reg_covar, init_method="kmeans", seed=42):
    """
    Fit a Gaussian Mixture Model using the EM algorithm.
    
    Algorithm:
    1. Initialize parameters (KMeans or random)
    2. Repeat until convergence:
       a. E-step: compute responsibilities
       b. M-step: update parameters
       c. Check convergence: |delta log-likelihood| < tol
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        K (int): Number of mixture components.
        max_iters (int): Maximum EM iterations.
        tol (float): Convergence tolerance.
        reg_covar (float): Covariance regularization.
        init_method (str): Initialization method.
        seed (int): Random seed.
        
    Returns:
        tuple: (params, responsibilities, log_likelihoods, n_iters)
    """
    # Initialize
    params = initialize_params(X, K, method=init_method, seed=seed)
    
    prev_log_likelihood = -np.inf
    log_likelihoods = []
    
    print(f"\n  Running EM (K={K}, max_iters={max_iters}, tol={tol})...")
    
    for iteration in range(1, max_iters + 1):
        # E-step
        responsibilities = e_step(X, params)
        
        # M-step
        params = m_step(X, responsibilities, reg_covar)
        
        # Compute log-likelihood
        log_likelihood = compute_log_likelihood(X, params)
        log_likelihoods.append(log_likelihood)
        
        # Check convergence
        delta = abs(log_likelihood - prev_log_likelihood)
        
        if iteration <= 5 or iteration % 10 == 0:
            print(f"    Iter {iteration:3d}: log-likelihood = {log_likelihood:.6f}, "
                  f"delta = {delta:.2e}")
        
        if delta < tol and iteration > 1:
            print(f"    Converged at iteration {iteration} "
                  f"(delta={delta:.2e} < tol={tol})")
            break
        
        prev_log_likelihood = log_likelihood
    else:
        print(f"    Reached max iterations ({max_iters})")
    
    return params, responsibilities, log_likelihoods, iteration

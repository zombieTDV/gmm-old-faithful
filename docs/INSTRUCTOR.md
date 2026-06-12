# PROJECT INSTRUCTION — Gaussian Mixture Model (GMM) from Scratch (Old Faithful)

## 1. Project Goal

You are required to implement a complete machine learning system from scratch that builds a **Gaussian Mixture Model (GMM)** trained using the **Expectation-Maximization (EM) algorithm** on the **Old Faithful Geyser dataset**.

The system must:
- Be fully implemented without any machine learning libraries
- Be modular, clean, and easy to understand
- Include preprocessing, training, evaluation, visualization, and experiments
- Include comparison against multiple baseline algorithms
- Be fully reproducible and self-contained

# THE CODE MUST BE EASY VERY EASY TO READ 

---

## 2. Hard Constraints (STRICT)

You are NOT allowed to use:

- ❌ scikit-learn
- ❌ scipy
- ❌ pandas
- ❌ seaborn
- ❌ any ML / statistics libraries
DO NOT USE ANY libraries FOR THIS PROJECT STRICT TO THIS 

Allowed:
- Python Standard Library only
- NumPy ONLY for:
  - linear algebra
  - matrix operations
  - determinant / inverse

All probabilistic and statistical functions MUST be implemented manually.

---

## 3. Dataset
- Raw data on data folder check and preproccesed and save to .csv with python code STRICT TO THIS RULE DO NOT DOWNLOAD USE ANY SOURCE FROM OTHER 
Input file:


data/raw/faithful.csv


Format:


eruptions,waiting
3.6,79
1.8,54
...


This dataset must be parsed manually using Python file I/O (`open()`).

---

## 4. Project Structure (MANDATORY)


gmm-old-faithful/
│
├── data/
│ ├── raw/
│ │ └── faithful.csv
│ ├── processed/
│ │ └── faithful_clean.csv
│
├── src/
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── gaussian.py
│ ├── gmm.py
│ ├── em.py
│ ├── kmeans.py
│ ├── knn.py
│ ├── hierarchical.py
│ ├── metrics.py
│ ├── visualization.py
│
├── experiments/
│ ├── run_gmm.py
│ ├── run_baselines.py
│ ├── run_experiments.py
│ ├── compare_models.py
│
├── outputs/
│ ├── plots/
│ ├── logs/
│
├── docs/
│ └── instructor.md <-- THIS FILE
│
├── config.py
└── README.md


---

## 5. Data Pipeline (MANDATORY IMPLEMENTATION)

You must implement a full preprocessing pipeline:

### Steps:

1. Load raw CSV using Python `open()`
2. Parse numeric values manually (no pandas)
3. Remove invalid / missing rows
4. Convert dataset into NumPy array
5. Normalize features:

\[
x' = \frac{x - \mu}{\sigma}
\]

6. Save processed dataset to:


data/processed/faithful_clean.csv


---

## 6. Gaussian Distribution (CORE COMPONENT)

Implement multivariate Gaussian distribution from scratch:

\[
\mathcal{N}(x|\mu,\Sigma)
\]

You MUST manually implement:
- covariance matrix computation
- determinant
- matrix inverse
- exponent term
- normalization constant

No external statistical functions allowed.

---

## 7. Gaussian Mixture Model (CORE)

Model definition:

\[
p(x)=\sum_{k=1}^{K} \pi_k \mathcal{N}(x|\mu_k,\Sigma_k)
\]

Parameters:
- μ (means)
- Σ (covariances)
- π (mixing weights)

---

## 8. EM Algorithm (CORE REQUIREMENT)

### E-step:

Compute responsibilities:

\[
\gamma_{nk} = \frac{\pi_k \mathcal{N}(x_n|\mu_k,\Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_n|\mu_j,\Sigma_j)}
\]

---

### M-step:

Update parameters:

- Mean:
\[
\mu_k = \frac{\sum_n \gamma_{nk} x_n}{\sum_n \gamma_{nk}}
\]

- Covariance:
\[
\Sigma_k = \frac{\sum_n \gamma_{nk}(x_n-\mu_k)(x_n-\mu_k)^T}{\sum_n \gamma_{nk}}
\]

- Mixing weights:
\[
\pi_k = \frac{N_k}{N}
\]

---

## 9. Initialization Strategy

Implement:

- Random initialization
- KMeans initialization (from scratch)

Default:

init_method = "kmeans"


---

## 10. Baseline Models (MANDATORY COMPARISON)

You must implement all baselines from scratch.

---

### 10.1 K-Means

- Euclidean distance
- centroid update
- hard assignment

Purpose:
Compare hard clustering vs probabilistic clustering

---

### 10.2 K-Nearest Neighbors (KNN)

Since dataset has no labels:

Procedure:
- Generate pseudo-labels from KMeans output
- Train manual KNN classifier
- Evaluate consistency of clustering separation

Purpose:
Show instance-based vs generative modeling differences

---

### 10.3 Hierarchical Clustering (MANDATORY 3rd method)

- Agglomerative clustering
- Simple linkage (single or average linkage)

Purpose:
Compare:
- distance-based hierarchy vs EM-based probabilistic clustering

---

## 11. Visualization Requirements

Use matplotlib only.

You MUST generate:

1. Raw dataset scatter plot
2. GMM clustering result
3. KMeans clustering result
4. Hierarchical clustering result
5. Gaussian ellipses visualization for GMM
6. Final comparison plot

---

## 12. Evaluation Metrics (NO LIBRARIES)

Implement manually:

### 12.1 Log-likelihood

\[
\log P(X|\theta)
\]

---

### 12.2 Silhouette Score (simplified manual version allowed)

---

### 12.3 Cluster separation distance

Mean inter-cluster distance

---

## 13. EXPERIMENT SECTION (MANDATORY)

You MUST include a structured experimental study.

---

### 13.1 Experiment Goals

- Evaluate GMM vs KMeans vs KNN vs Hierarchical clustering
- Understand performance differences
- Validate Gaussian mixture assumption

---

### 13.2 Hyperparameter Selection Justification

You must justify:

#### K (number of clusters)
- Why K=2 for Old Faithful:
  - eruption patterns are bimodal
  - short vs long eruption behavior

#### EM iterations (max_iters)
- Tradeoff between convergence stability and computation

#### Covariance regularization (reg_covar)
- Prevent singular matrix
- Ensure numerical stability

#### Initialization method
- KMeans init improves convergence speed and stability

---

### 13.3 Experimental Outputs

Generate:

- plots/gmm_result.png
- plots/kmeans_result.png
- plots/hierarchical_result.png
- plots/comparison.png
- logs/metrics.txt

---

### 13.4 Required Analysis

Explain:

- Why GMM performs better than KMeans on probabilistic clusters
- Why KNN is not suitable for unsupervised structure
- Why hierarchical clustering behaves differently
- Effect of covariance modeling in GMM

---

## 14. Code Quality Requirements

- No global variables
- No magic numbers
- Functions must be modular (< 40 lines)
- Clear naming conventions
- Fully deterministic when seed is fixed
- No hidden dependencies

---

## 15. Core Concept Constraint

The entire system must strictly follow:

> GMM is a probabilistic generative model optimized via Expectation-Maximization.

No shortcuts or library abstractions allowed.

---

## 16. Final Deliverable

A fully working ML system that:

1. Loads raw Old Faithful dataset
2. Preprocesses it manually
3. Implements Gaussian distribution from scratch
4. Implements GMM with EM algorithm
5. Compares against:
   - KMeans
   - KNN
   - Hierarchical clustering
6. Produces full visual + quantitative experiment report

---

## END OF INSTRUCTION
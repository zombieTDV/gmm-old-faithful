# Complete Guide — Gaussian Mixture Model from Scratch

## Old Faithful Geyser Dataset

> A full mathematical and engineering reference for every component
> of this project: theory, derivations, algorithms, pipeline, and experiments.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Data Pipeline](#3-data-pipeline)
4. [Probability Theory Foundations](#4-probability-theory-foundations)
5. [Multivariate Gaussian Distribution](#5-multivariate-gaussian-distribution)
6. [Gaussian Mixture Model (GMM)](#6-gaussian-mixture-model-gmm)
7. [Expectation-Maximization (EM) Algorithm](#7-expectation-maximization-em-algorithm)
8. [K-Means Clustering](#8-k-means-clustering)
9. [K-Nearest Neighbors (KNN)](#9-k-nearest-neighbors-knn)
10. [Agglomerative Hierarchical Clustering](#10-agglomerative-hierarchical-clustering)
11. [Evaluation Metrics](#11-evaluation-metrics)
12. [Hyperparameter Selection](#12-hyperparameter-selection)
13. [Visualization Theory](#13-visualization-theory)
14. [Experiment Pipeline](#14-experiment-pipeline)
15. [Results and Analysis](#15-results-and-analysis)

---

## 1. Project Overview

### 1.1 Goal

Build a complete machine learning system that:
- Implements a **Gaussian Mixture Model (GMM)** from scratch
- Trains it using the **Expectation-Maximization (EM)** algorithm
- Applies it to the **Old Faithful Geyser** dataset
- Compares it against three baseline clustering algorithms

### 1.2 Constraints

- **No ML libraries**: No scikit-learn, scipy, pandas, seaborn
- **Allowed**: Python standard library, NumPy (linear algebra only), Matplotlib (plotting only)
- **All statistical and probabilistic functions are implemented manually**

### 1.3 Why This Matters

This project demonstrates understanding of:
- Probabilistic generative modeling
- Iterative optimization (EM)
- The difference between hard and soft clustering
- Manual implementation of statistical computations
- Experimental methodology in machine learning

---

## 2. Dataset Description

### 2.1 Old Faithful Geyser

Old Faithful is a cone geyser in Yellowstone National Park, Wyoming, USA. It erupts approximately every 44 to 125 minutes, with eruptions lasting 1.5 to 5 minutes.

**Source file**: `data/raw/Old_Faithful_data.xls` → converted to `data/raw/faithful.csv`

### 2.2 Features

| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| `eruptions` | Duration of eruption | minutes | ~1.6 to ~5.1 |
| `waiting` | Time until next eruption | minutes | ~43 to ~96 |

**N = 298 observations**

### 2.3 Bimodal Structure

The dataset exhibits a **bimodal distribution** — two distinct clusters:

| Cluster | Eruption Duration | Waiting Time | Physical Cause |
|---------|------------------|--------------|----------------|
| **Short** | ~2.0 min | ~55 min | Partial water chamber discharge |
| **Long** | ~4.5 min | ~80 min | Full water chamber discharge |

This bimodality arises from the geyser's underground plumbing: the water chamber either partially or fully empties during an eruption, leading to two distinct refill patterns.

### 2.4 Why This Dataset is Ideal for GMM

1. **Two clear Gaussian-like clusters** → K=2 is well-motivated
2. **Elliptical clusters** → eruption duration and waiting time are positively correlated within each cluster
3. **Overlapping boundaries** → soft (probabilistic) assignment is meaningful
4. **Low dimensionality** (D=2) → easy to visualize all results
5. **No ground-truth labels** → true unsupervised learning problem

---

## 3. Data Pipeline

### 3.1 Pipeline Architecture

```
Raw XLS → CSV extraction → Manual CSV parsing → Cleaning → Standardization → Processed CSV
```

| Step | File | Input | Output |
|------|------|-------|--------|
| 1. Extract | (one-time script) | `Old_Faithful_data.xls` | `faithful.csv` |
| 2. Load | `src/data_loader.py` | `faithful.csv` | List of `[float, float]` |
| 3. Clean | `src/preprocessing.py` | Raw list | NumPy array (N, 2) |
| 4. Standardize | `src/preprocessing.py` | Clean array | Standardized array |
| 5. Save | `src/preprocessing.py` | Standardized array | `faithful_clean.csv` |

### 3.2 Step 1: CSV Loading (Manual)

**No pandas, no csv module** — pure Python file I/O:

```python
with open(filepath, 'r') as file:
    lines = file.readlines()

for line in lines[1:]:          # Skip header
    parts = line.strip().split(',')
    eruptions = float(parts[0])
    waiting = float(parts[1])
```

**Edge cases handled**:
- Empty lines → skipped
- Non-numeric values → skipped with warning
- Wrong number of columns → skipped with warning

### 3.3 Step 2: Data Cleaning

Remove rows where any value is:
- Zero or negative (physically impossible for duration/waiting)
- Missing or NaN

**Result**: 298 → 298 rows (dataset is already clean)

### 3.4 Step 3: Standardization (Z-Score Normalization)

**Why standardize?**

The two features have different scales:
- Eruptions: range ~1.6–5.1 (span ≈ 3.5)
- Waiting: range ~43–96 (span ≈ 53)

Without standardization:
- Waiting time would **dominate** distance calculations (15× larger range)
- Covariance matrices would be poorly conditioned
- EM convergence would be slow and numerically unstable

**Formula**:

For each feature j:

```
x'_j = (x_j - μ_j) / σ_j
```

Where:
- **μ_j** (mean) = (1/N) × Σᵢ xᵢⱼ
- **σ_j** (standard deviation) = √[(1/N) × Σᵢ (xᵢⱼ - μ_j)²]

**After standardization**:
- Each feature has mean = 0 and standard deviation = 1
- Both features contribute equally to distance-based computations
- Covariance matrix starts close to the identity matrix

**Mathematical detail — Population vs Sample std**:

We use **population** standard deviation (divide by N, not N-1) because:
- We are transforming the entire dataset, not estimating a population parameter
- It matches the convention used in most ML preprocessing pipelines
- The difference is negligible for N=298 (1/298 vs 1/297)

---

## 4. Probability Theory Foundations

### 4.1 Random Variables and Distributions

A **continuous random variable** X has a probability density function (PDF) f(x) such that:

```
P(a ≤ X ≤ b) = ∫ₐᵇ f(x) dx
```

Properties:
- f(x) ≥ 0 for all x
- ∫₋∞^∞ f(x) dx = 1

### 4.2 Univariate Gaussian (Normal) Distribution

The simplest Gaussian distribution for a single variable:

```
N(x | μ, σ²) = (1 / √(2πσ²)) × exp(-(x - μ)² / (2σ²))
```

Parameters:
- **μ** (mu): mean — center of the distribution
- **σ²** (sigma squared): variance — spread of the distribution

Properties:
- Bell-shaped, symmetric around μ
- 68% of data within μ ± σ
- 95% of data within μ ± 2σ
- 99.7% of data within μ ± 3σ

### 4.3 Why Gaussian?

The Gaussian distribution is fundamental because of the **Central Limit Theorem**: the sum of many independent random variables converges to a Gaussian, regardless of the original distributions. Physical processes (like geyser eruptions influenced by many small factors) often produce approximately Gaussian data.

### 4.4 Maximum Likelihood Estimation (MLE)

Given data X = {x₁, ..., xₙ}, MLE finds parameters θ that maximize:

```
L(θ) = ∏ᵢ p(xᵢ | θ)
```

In practice, we maximize the **log-likelihood** (monotonic transform):

```
log L(θ) = Σᵢ log p(xᵢ | θ)
```

**Why log?**
- Products become sums (easier to differentiate)
- Avoids numerical underflow (products of small probabilities → 0)
- Same maximizer as the original likelihood

### 4.5 Latent Variables

A **latent variable** is an unobserved variable that explains structure in the data. In GMM:
- **Observed**: data points x₁, ..., xₙ
- **Latent**: which cluster each point belongs to (z₁, ..., zₙ)

We can't directly maximize the log-likelihood because it involves summing over all possible latent variable assignments. The EM algorithm handles this elegantly.

---

## 5. Multivariate Gaussian Distribution

### 5.1 Definition

For a D-dimensional random vector **x**, the multivariate Gaussian PDF is:

```
N(x | μ, Σ) = (1 / ((2π)^(D/2) × |Σ|^(1/2))) × exp(-½ (x - μ)ᵀ Σ⁻¹ (x - μ))
```

**Parameters**:
- **μ** ∈ ℝᴰ — mean vector (center of the distribution)
- **Σ** ∈ ℝᴰˣᴰ — covariance matrix (shape and orientation)

### 5.2 Anatomy of the PDF

The formula has three parts:

#### Part 1: Normalization Constant

```
1 / ((2π)^(D/2) × |Σ|^(1/2))
```

- **(2π)^(D/2)**: scales with dimensionality (D=2 in our case → 2π)
- **|Σ|**: determinant of the covariance matrix
  - Represents the "volume" of the distribution's spread
  - Larger determinant → more spread out → lower peak density

#### Part 2: Mahalanobis Distance

```
(x - μ)ᵀ Σ⁻¹ (x - μ)
```

This is the **squared Mahalanobis distance** — a generalization of Euclidean distance that accounts for the shape of the distribution:

- **Σ⁻¹** (precision matrix): the inverse of the covariance
- Unlike Euclidean distance, it accounts for:
  - Different variances along different axes
  - Correlations between features
- Points at equal Mahalanobis distance from μ form an **ellipse** (in 2D)

**Example**: If eruption duration and waiting time are correlated, the Mahalanobis distance "rotates" the distance metric to align with the data's natural axes.

#### Part 3: Exponential

```
exp(-½ × Mahalanobis²)
```

- Converts distance to probability density
- Points closer to μ get higher density
- Decay rate controlled by the covariance

### 5.3 Covariance Matrix

For D=2 (our case):

```
Σ = | σ₁²    σ₁₂ |
    | σ₁₂    σ₂²  |
```

Where:
- **σ₁²**: variance of eruption duration
- **σ₂²**: variance of waiting time  
- **σ₁₂**: covariance between eruption duration and waiting time

**Covariance formula** (computed manually):

```
Σ = (1/N) × Σᵢ (xᵢ - μ)(xᵢ - μ)ᵀ
```

This is an outer product sum:

```
(xᵢ - μ)(xᵢ - μ)ᵀ = | (x₁ᵢ-μ₁)²         (x₁ᵢ-μ₁)(x₂ᵢ-μ₂) |
                      | (x₂ᵢ-μ₂)(x₁ᵢ-μ₁)  (x₂ᵢ-μ₂)²         |
```

### 5.4 Properties of the Covariance Matrix

1. **Symmetric**: Σᵀ = Σ (covariance is commutative)
2. **Positive semi-definite**: xᵀΣx ≥ 0 for all x (variances are non-negative)
3. **Determinant |Σ| > 0** for non-degenerate distributions (required for PDF to be finite)

### 5.5 Geometric Interpretation (2D)

The covariance matrix determines the **shape** of the distribution:

| Σ Type | Shape | Example |
|--------|-------|---------|
| σ₁₂ = 0, σ₁ = σ₂ | Circle | Features are independent, equal variance |
| σ₁₂ = 0, σ₁ ≠ σ₂ | Axis-aligned ellipse | Features independent, different variance |
| σ₁₂ ≠ 0 | Rotated ellipse | Features are correlated |

**For Old Faithful**: σ₁₂ > 0 (positive correlation) → rotated ellipses tilted upward-right, because longer eruptions correlate with longer waiting times.

### 5.6 Determinant and Inverse

**Determinant** (2×2 case):

```
|Σ| = σ₁² × σ₂² - σ₁₂²
```

- Must be > 0 (otherwise the distribution is degenerate)
- If σ₁₂² approaches σ₁² × σ₂², the determinant approaches 0 → **singular matrix**
- This is why we add regularization: Σ + εI

**Inverse** (2×2 case):

```
Σ⁻¹ = (1/|Σ|) × | σ₂²   -σ₁₂ |
                  | -σ₁₂   σ₁²  |
```

In our implementation, we use `np.linalg.det()` and `np.linalg.inv()` for these operations, as allowed by the project constraints (NumPy for linear algebra only).

---

## 6. Gaussian Mixture Model (GMM)

### 6.1 Motivation

A single Gaussian distribution can model one cluster. But Old Faithful has **two** clusters. A **Gaussian Mixture Model** solves this by combining multiple Gaussians:

```
p(x) = Σₖ₌₁ᴷ πₖ × N(x | μₖ, Σₖ)
```

This says: "The probability of observing point x is a weighted sum of K Gaussian densities."

### 6.2 Parameters

A GMM with K components has three sets of parameters:

| Parameter | Symbol | Shape | Description |
|-----------|--------|-------|-------------|
| Mixing weights | πₖ | (K,) | Probability of each component; Σₖ πₖ = 1 |
| Means | μₖ | (K, D) | Center of each Gaussian |
| Covariances | Σₖ | (K, D, D) | Shape of each Gaussian |

**Total parameters** for K=2, D=2:
- Weights: K-1 = 1 (the last is determined by Σπ=1)
- Means: K×D = 4
- Covariances: K×D×(D+1)/2 = 6 (symmetric matrix has D(D+1)/2 unique entries)
- **Total: 11 parameters** to learn from 298 data points

### 6.3 Generative Story

The GMM defines a **generative process** — how the data was supposedly created:

1. **Choose a component** k with probability πₖ (e.g., "is this a short or long eruption?")
2. **Sample a data point** x from N(μₖ, Σₖ) (e.g., "generate duration and waiting time from that component's distribution")

This is a **generative model**: it defines a full probability distribution over the data space, allowing us to:
- Compute the probability of any new observation
- Generate synthetic data
- Perform soft clustering (probabilistic membership)

### 6.4 Hard vs Soft Clustering

| Aspect | Hard (K-Means) | Soft (GMM) |
|--------|----------------|------------|
| Assignment | Each point belongs to exactly 1 cluster | Each point has a probability for each cluster |
| Boundary points | Forced into one cluster | Can be 60% cluster A, 40% cluster B |
| Output | Labels: {0, 1} | Responsibilities: [0.6, 0.4] |
| Cluster shape | Spherical (Voronoi cells) | Elliptical (full covariance) |

### 6.5 Why Not Just Use K-Means?

K-Means is a special case of GMM where:
- All covariances are equal to σ²I (spherical clusters)
- All weights are equal (πₖ = 1/K)
- Assignments are hard (0 or 1 instead of probabilities)

Old Faithful clusters are **elliptical and correlated**, so K-Means's spherical assumption is violated. GMM captures this structure.

---

## 7. Expectation-Maximization (EM) Algorithm

### 7.1 The Problem

We want to find parameters θ = {π, μ, Σ} that maximize the log-likelihood:

```
log L(θ) = Σₙ₌₁ᴺ log [Σₖ₌₁ᴷ πₖ × N(xₙ | μₖ, Σₖ)]
```

**Why is this hard?** The log of a sum has no closed-form solution. Unlike simple MLE for a single Gaussian, we can't take derivatives, set them to zero, and solve directly.

### 7.2 The EM Idea

EM introduces **latent variables** zₙ (which component generated each point) and alternates between:

1. **E-step**: "If the current parameters were correct, which component most likely generated each point?"
2. **M-step**: "Given those responsibilities, what are the best parameters?"

### 7.3 E-Step: Computing Responsibilities

For each data point xₙ and component k, compute the **responsibility** (posterior probability):

```
γ(n,k) = πₖ × N(xₙ | μₖ, Σₖ) / Σⱼ₌₁ᴷ πⱼ × N(xₙ | μⱼ, Σⱼ)
```

**Interpretation**: γ(n,k) = "probability that component k generated point xₙ"

**Properties**:
- 0 ≤ γ(n,k) ≤ 1
- Σₖ γ(n,k) = 1 for each n (probabilities sum to 1)
- This is Bayes' theorem applied to mixture components

**Derivation via Bayes' theorem**:

```
P(z=k | xₙ, θ) = P(xₙ | z=k, θ) × P(z=k | θ) / P(xₙ | θ)
                = N(xₙ | μₖ, Σₖ) × πₖ / Σⱼ πⱼ × N(xₙ | μⱼ, Σⱼ)
```

### 7.4 M-Step: Updating Parameters

Using the responsibilities from the E-step, update each parameter:

#### Effective count for component k:

```
Nₖ = Σₙ₌₁ᴺ γ(n,k)
```

This is the "soft count" of points assigned to component k. Unlike K-Means (where each point is counted once), here each point contributes fractionally to each component.

#### Update mixing weights:

```
πₖ = Nₖ / N
```

**Intuition**: The weight of component k equals the fraction of (soft) data it explains.

#### Update means:

```
μₖ = (1/Nₖ) × Σₙ₌₁ᴺ γ(n,k) × xₙ
```

**Intuition**: The mean of component k is a weighted average of all data points, where the weights are the responsibilities. Points strongly assigned to component k pull its mean toward them.

#### Update covariances:

```
Σₖ = (1/Nₖ) × Σₙ₌₁ᴺ γ(n,k) × (xₙ - μₖ)(xₙ - μₖ)ᵀ + εI
```

**Intuition**: The covariance of component k is a responsibility-weighted scatter matrix. Points far from μₖ that are strongly assigned to k increase the covariance.

**The εI term** (regularization): prevents the covariance matrix from becoming singular if a component collapses onto a single point. See Section 12.4 for justification.

### 7.5 Convergence

EM is guaranteed to **monotonically increase** the log-likelihood at each iteration:

```
log L(θ⁽ᵗ⁺¹⁾) ≥ log L(θ⁽ᵗ⁾)
```

**Proof sketch**: Each E-step computes the exact posterior (tightest lower bound on log L), and each M-step maximizes this bound. Together, they cannot decrease the objective.

**Stopping criteria**:
- |log L(θ⁽ᵗ⁺¹⁾) - log L(θ⁽ᵗ⁾)| < TOL (change is negligible)
- OR iteration count > MAX_ITERS (safety cap)

**Caveat**: EM converges to a **local** maximum, not necessarily the global maximum. This is why initialization matters (see Section 12.5).

### 7.6 Log-Likelihood Computation

At each iteration, we evaluate model quality:

```
log L = Σₙ₌₁ᴺ log [Σₖ₌₁ᴷ πₖ × N(xₙ | μₖ, Σₖ)]
```

**Step-by-step**:
1. For each point xₙ, compute the weighted PDF for each component: πₖ × N(xₙ | μₖ, Σₖ)
2. Sum across components: p(xₙ) = Σₖ πₖ × N(xₙ | μₖ, Σₖ)
3. Take the log: log p(xₙ)
4. Sum across all points: log L = Σₙ log p(xₙ)

**Numerical safety**: We clamp p(xₙ) ≥ 1e-300 before taking the log to prevent log(0) = -∞.

### 7.7 Complete EM Algorithm

```
Algorithm: EM for GMM
────────────────────────────────
Input: Data X, K, max_iters, tol, reg_covar
Output: Trained GMM parameters θ* = {π*, μ*, Σ*}

1. INITIALIZE θ⁰ (using KMeans or random)
2. prev_ll ← -∞
3. FOR t = 1, 2, ..., max_iters:
   
   4. E-STEP: Compute responsibilities
      FOR each n, k:
        γ(n,k) ← πₖ N(xₙ|μₖ,Σₖ) / Σⱼ πⱼ N(xₙ|μⱼ,Σⱼ)
   
   5. M-STEP: Update parameters
      FOR each k:
        Nₖ ← Σₙ γ(n,k)
        πₖ ← Nₖ / N
        μₖ ← Σₙ γ(n,k)xₙ / Nₖ
        Σₖ ← Σₙ γ(n,k)(xₙ-μₖ)(xₙ-μₖ)ᵀ / Nₖ + εI
   
   6. EVALUATE: Compute log-likelihood
      ll ← Σₙ log(Σₖ πₖ N(xₙ|μₖ,Σₖ))
   
   7. CHECK CONVERGENCE:
      IF |ll - prev_ll| < tol:
        BREAK (converged)
      prev_ll ← ll

8. RETURN θ* = {π, μ, Σ}, γ
```

---

## 8. K-Means Clustering

### 8.1 Algorithm

K-Means is a hard clustering algorithm that minimizes within-cluster sum of squared distances:

```
J = Σₖ₌₁ᴷ Σₙ∈Cₖ ||xₙ - cₖ||²
```

Where cₖ is the centroid of cluster k and Cₖ is the set of points assigned to cluster k.

### 8.2 Steps

```
Algorithm: K-Means
────────────────────────────────
Input: Data X, K, max_iters, seed
Output: Labels, Centroids

1. INITIALIZE: Pick K random data points as centroids
2. REPEAT until convergence or max_iters:
   
   3. ASSIGN: For each point, find nearest centroid
      label(n) ← argminₖ ||xₙ - cₖ||₂
   
   4. UPDATE: Recompute centroids as cluster means
      cₖ ← (1/|Cₖ|) × Σₙ∈Cₖ xₙ
   
   5. CHECK: If centroids haven't moved, stop
```

### 8.3 Euclidean Distance

```
d(a, b) = √(Σᵢ (aᵢ - bᵢ)²)
```

For D=2:

```
d(a, b) = √((a₁ - b₁)² + (a₂ - b₂)²)
```

### 8.4 K-Means as a Special Case of GMM

K-Means can be derived from EM for a restricted GMM where:
- All Σₖ = σ²I (spherical, equal covariance)
- As σ² → 0, soft assignments become hard assignments
- The E-step simplifies to nearest-centroid assignment
- The M-step simplifies to computing cluster means

This reveals **K-Means' limitation**: it cannot model elliptical or correlated clusters.

### 8.5 Comparison with GMM

| Property | K-Means | GMM |
|----------|---------|-----|
| Assignment | Hard (0 or 1) | Soft (probability) |
| Cluster shape | Spherical | Elliptical |
| Parameters | Centroids only | Weights + Means + Covariances |
| Optimization | Minimize distance | Maximize likelihood |
| Probabilistic | No | Yes (generative model) |

---

## 9. K-Nearest Neighbors (KNN)

### 9.1 Algorithm

KNN is a **supervised, instance-based** (lazy) classifier. For each query point:

1. Compute distance to all training points
2. Find the K nearest neighbors
3. Predict the majority label among neighbors

### 9.2 Why KNN in an Unsupervised Problem?

Old Faithful has **no ground-truth labels**. We use KNN to test **clustering consistency**:

1. Generate pseudo-labels using K-Means
2. For each point (leave-one-out):
   - Remove it from the dataset
   - Predict its label using KNN on the remaining points
   - Check if the prediction matches the K-Means label
3. High consistency → clusters are locally coherent and well-separated

### 9.3 Leave-One-Out Cross-Validation

```
For each point i:
    X_train ← X without point i
    y_train ← labels without label i
    ŷᵢ ← KNN(X_train, y_train, xᵢ, k)
    correct += (ŷᵢ == yᵢ)

consistency = correct / N
```

### 9.4 Majority Voting

For K nearest neighbors with labels {y₁, ..., yₖ}:

```
ŷ = argmax_c |{j : yⱼ = c}|
```

The predicted label is the most frequent label among the K neighbors.

**Why k must be odd**: With 2 classes and even k, ties are possible (e.g., k=4 with 2 votes each). An odd k guarantees a clear majority.

### 9.5 Comparison with GMM

| Property | KNN | GMM |
|----------|-----|-----|
| Type | Supervised, discriminative | Unsupervised, generative |
| Training | None (stores data) | EM optimization |
| New points | Distance computation | PDF evaluation |
| Discovers clusters? | No (needs labels) | Yes |
| Probabilistic? | No (majority vote) | Yes (posterior probabilities) |

---

## 10. Agglomerative Hierarchical Clustering

### 10.1 Algorithm

Bottom-up approach that builds a cluster tree (dendrogram):

```
Algorithm: Agglomerative Hierarchical Clustering
────────────────────────────────
Input: Data X, K (target clusters), linkage method
Output: Cluster labels

1. INITIALIZE: Each point is its own cluster (N clusters)
2. Compute pairwise distance matrix D
3. REPEAT until K clusters remain:
   
   4. FIND: The two closest clusters (i, j)
   5. MERGE: Combine clusters i and j
   6. UPDATE: Recompute distances to the merged cluster
```

### 10.2 Linkage Methods

How to measure distance between two **clusters** (sets of points):

| Linkage | Formula | Behavior |
|---------|---------|----------|
| **Single** | min d(a,b) for a∈A, b∈B | Tends to create elongated chains |
| **Average** | mean d(a,b) for a∈A, b∈B | Balanced, moderate-sized clusters |
| **Complete** | max d(a,b) for a∈A, b∈B | Tends to create compact, equal-size clusters |

**We use average linkage** — it produces balanced clusters without the chaining problem of single linkage.

### 10.3 Average Linkage Formula

```
d(A, B) = (1 / (|A| × |B|)) × Σₐ∈ₐ Σᵦ∈ᵦ ||a - b||₂
```

After merging clusters A and B into cluster C = A ∪ B, the distance from C to any other cluster E must be recomputed:

```
d(C, E) = (1 / (|C| × |E|)) × Σ_c∈C Σ_e∈E ||c - e||₂
```

### 10.4 Pairwise Distance Matrix

For N data points, the initial distance matrix is N×N:

```
D[i,j] = ||xᵢ - xⱼ||₂
```

**Complexity**: O(N²) space, O(N² × D) time to compute.

For N=298, this is 298 × 298 = 88,804 distances — manageable.

### 10.5 Comparison with GMM

| Property | Hierarchical | GMM |
|----------|-------------|-----|
| Approach | Bottom-up merging | Iterative optimization |
| Assumptions | None (distance-based) | Gaussian clusters |
| Assignment | Hard | Soft (probabilistic) |
| Dendrogram | Yes (full tree) | No |
| Cluster shape | Arbitrary (linkage-dependent) | Elliptical |
| Number of K | Can vary post-hoc | Must be fixed upfront |

---

## 11. Evaluation Metrics

### 11.1 Log-Likelihood (GMM Only)

```
log L = Σₙ₌₁ᴺ log [Σₖ₌₁ᴷ πₖ × N(xₙ | μₖ, Σₖ)]
```

**Interpretation**:
- Higher is better (data is more probable under the model)
- Only applies to probabilistic models (not K-Means or Hierarchical)
- Can be used to compare different GMM configurations (different K, initializations)

**Range**: (-∞, 0) for normalized data (log of densities < 1)

### 11.2 Silhouette Score

Measures how well each point fits its assigned cluster versus the nearest alternative cluster.

#### For a single point i:

**a(i)** — mean distance to other points in the **same** cluster (cohesion):

```
a(i) = (1 / (|Cₖ| - 1)) × Σⱼ∈Cₖ,j≠i ||xᵢ - xⱼ||₂
```

Small a(i) = point is close to its cluster members (good).

**b(i)** — mean distance to points in the **nearest other** cluster (separation):

```
b(i) = min_{k'≠k} [(1 / |Cₖ'|) × Σⱼ∈Cₖ' ||xᵢ - xⱼ||₂]
```

Large b(i) = point is far from other clusters (good).

**Silhouette score**:

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

#### Interpretation:

| s(i) value | Meaning |
|-----------|---------|
| s ≈ +1 | Point is well-matched to its cluster, far from others |
| s ≈ 0 | Point is on the boundary between two clusters |
| s ≈ -1 | Point may be assigned to the wrong cluster |

#### Mean Silhouette Score:

```
S = (1/N) × Σᵢ s(i)
```

**Range**: [-1, 1], higher is better.

### 11.3 Cluster Separation Distance

Mean distance between cluster centroids:

```
Sep = (2 / (K(K-1))) × Σᵢ Σⱼ>ᵢ ||cᵢ - cⱼ||₂
```

For K=2:

```
Sep = ||c₁ - c₂||₂
```

**Interpretation**: Higher = clusters are farther apart = better separation.

### 11.4 KNN Consistency Score

```
Consistency = (Number of correct leave-one-out predictions) / N
```

**Range**: [0, 1], where 1 = perfect consistency.

**Interpretation**: If KNN can perfectly reproduce the clustering labels using only local neighborhood information, the clusters are locally coherent.

### 11.5 Bayesian Information Criterion (BIC)

```
BIC = -2 × log L + p × log(N)
```

Where:
- log L = log-likelihood
- p = number of free parameters
- N = number of data points

**Lower BIC is better** — it penalizes model complexity (number of parameters) to prevent overfitting.

For GMM with K components and D features:
- p = K×D + K×D×(D+1)/2 + (K-1) = K(D + D(D+1)/2 + 1) - 1

---

## 12. Hyperparameter Selection

### 12.1 K = 2 (Number of Clusters)

**Theoretical justification**:
- Old Faithful has two distinct eruption regimes (short and long)
- The bimodal distribution is well-documented in geophysics

**Experimental evidence** (from `hyperparameter_study.py`):
- K=2 achieves the **highest silhouette score** among K=2,3,4,5
- K=2 achieves the **lowest BIC** (best model complexity tradeoff)
- K≥3 splits natural clusters without physical justification

### 12.2 MAX_ITERS = 100 (EM Iteration Cap)

**Theoretical justification**:
- EM is guaranteed to converge (monotonically increasing log-likelihood)
- The cap prevents infinite loops in edge cases

**Experimental evidence**:
- EM converges in **~38 iterations** with KMeans init
- 100 provides a **~3× safety margin**
- Log-likelihood changes become < 1e-6 well before iteration 100

### 12.3 TOL = 1e-6 (Convergence Tolerance)

**Theoretical justification**:
- When |Δ log L| < 1e-6, parameters have stabilized to ~6 decimal places
- This is well above float64 machine epsilon (~2.2e-16), avoiding false non-convergence due to floating-point noise
- It's well below 1e-4, ensuring genuine convergence (not premature stopping)

**Experimental evidence**:
- The convergence curve shows log-likelihood plateaus are correctly detected at this tolerance

### 12.4 REG_COVAR = 1e-6 (Covariance Regularization)

**Theoretical justification**:

**Problem**: During EM, if a Gaussian component collapses onto a single data point:
- The covariance matrix becomes Σ → 0 (all points have zero spread)
- Determinant |Σ| → 0 (singular matrix)
- Inverse Σ⁻¹ → ∞ (undefined)
- PDF → ∞ for that point (infinite density spike)
- **The algorithm crashes**

**Solution**: Add εI to the diagonal:

```
Σₖ ← Σₖ + εI = Σₖ + ε × | 1 0 |
                           | 0 1 |
```

This ensures:
- |Σₖ| ≥ ε² > 0 (never singular)
- All eigenvalues ≥ ε (positive definite)
- The regularization adds a tiny isotropic variance (~1e-6) that is negligible compared to the actual data variance (~1.0 after standardization)

**Experimental evidence**:
- ε = 0 works on this dataset (no collapse occurs) but is risky
- ε = 1e-6 has negligible effect on log-likelihood and silhouette
- ε = 1e-2 starts degrading model fit (too much artificial variance)

### 12.5 INIT_METHOD = "kmeans" (Initialization Strategy)

**Theoretical justification**:

**Problem**: EM converges to a **local** maximum, not the global maximum. The initial parameters determine which local maximum is found.

**Random initialization** risks:
- Placing initial means far from data clusters → slow convergence
- Both means in the same cluster → poor local optimum
- Non-reproducible results across seeds

**KMeans initialization** solves this by:
- Running K-Means first to find cluster centroids
- Using these centroids as initial means for GMM
- Using cluster membership proportions as initial weights
- Using within-cluster covariance as initial covariance matrices

**Experimental evidence**:
- KMeans init: **avg 38 iterations** to converge
- Random init: **avg 59 iterations** to converge (55% slower)
- Both methods reach the same final log-likelihood on this dataset
- KMeans init is **consistent** across different seeds (same result every time)
- Random init has **variable** convergence behavior

### 12.6 RANDOM_SEED = 42 (Reproducibility)

**Justification**:
- Fixed seed ensures deterministic results across runs
- Required by the instructor specification (Section 14)
- Convention: 42 is the most common seed in ML (from "The Hitchhiker's Guide to the Galaxy")

### 12.7 K_NEIGHBORS = 5 (KNN Neighborhood Size)

**Theoretical justification**:
- **Odd number**: With 2 classes, odd k avoids ties in majority voting
- **k=5 ≈ 1.7% of dataset**: Local enough to respect decision boundaries, large enough to smooth noise
- **Bias-variance tradeoff**: k=1 is noisy (high variance), k=N is trivial (high bias)

**Experimental evidence**:
- k=1: 99.66% consistency (1 error due to noise)
- k=3,5,7,9: 100% consistency (perfect)
- k=15: 99.66% (starts losing local sensitivity)
- k=5 is the standard default that balances all factors

---

## 13. Visualization Theory

### 13.1 Gaussian Confidence Ellipses

To visualize a 2D Gaussian, we draw **confidence ellipses** at n standard deviations:

**Step 1: Eigendecomposition of Σ**

```
Σ = V Λ Vᵀ
```

Where:
- V = eigenvectors (columns) → direction of principal axes
- Λ = diag(λ₁, λ₂) → eigenvalues = variance along each axis

**Step 2: Compute ellipse semi-axes**

```
Semi-axis lengths = n × √λₖ
```

At n=1 (1σ): ellipse contains ~68% of the Gaussian density
At n=2 (2σ): ellipse contains ~95% of the Gaussian density
At n=3 (3σ): ellipse contains ~99.7% of the Gaussian density

**Step 3: Generate ellipse points**

1. Create a unit circle: (cos θ, sin θ) for θ ∈ [0, 2π]
2. Scale by semi-axis lengths: diag(n√λ₁, n√λ₂)
3. Rotate by eigenvectors: multiply by V
4. Translate to mean μ

```
Ellipse = V × diag(n√Λ) × unit_circle + μ
```

### 13.2 Plot Types

| Plot | Purpose | Key Information |
|------|---------|-----------------|
| Raw data scatter | Show bimodal structure | Visible two-cluster pattern |
| GMM result + ellipses | Show probabilistic clustering | Ellipse orientation reveals correlation |
| K-Means result | Show hard clustering | Voronoi-like boundaries |
| Hierarchical result | Show distance-based clustering | Potentially different boundaries |
| Comparison | Side-by-side model evaluation | Visual differences in assignments |
| EM convergence | Show optimization progress | Log-likelihood plateau = convergence |

---

## 14. Experiment Pipeline

### 14.1 Full Pipeline Flow

```
┌──────────────────┐
│  1. PREPROCESSING │
│  Load → Clean →   │
│  Standardize →    │
│  Save CSV         │
└────────┬─────────┘
         │
    ┌────▼────┐          ┌───────────┐
    │ 2. GMM  │          │ 3. BASE-  │
    │ Train   │          │   LINES   │
    │ EM algo │          │ KMeans    │
    │ Plot    │          │ Hierarch  │
    └────┬────┘          │ KNN      │
         │               └─────┬─────┘
         │                     │
    ┌────▼─────────────────────▼────┐
    │  4. HYPERPARAMETER EXPERIMENTS │
    │  Vary K, test convergence,     │
    │  regularization, init method,  │
    │  KNN neighborhood size         │
    └──────────────┬────────────────┘
                   │
    ┌──────────────▼────────────────┐
    │  5. MODEL COMPARISON           │
    │  Compute metrics for all       │
    │  Write metrics.txt             │
    │  Generate comparison plot      │
    └───────────────────────────────┘
```

### 14.2 Running the Pipeline

```bash
# Install dependencies
python3 -m pip install numpy matplotlib olefile --user

# Run everything
python3 experiments/run_experiments.py
```

### 14.3 Output Files

| File | Content |
|------|---------|
| `data/processed/faithful_clean.csv` | Standardized dataset |
| `outputs/plots/raw_data.png` | Raw data scatter |
| `outputs/plots/gmm_result.png` | GMM with confidence ellipses |
| `outputs/plots/em_convergence.png` | Log-likelihood convergence |
| `outputs/plots/kmeans_result.png` | K-Means clustering |
| `outputs/plots/hierarchical_result.png` | Hierarchical clustering |
| `outputs/plots/comparison.png` | Side-by-side comparison |
| `outputs/plots/exp1_vary_k.png` | Experiment: optimal K |
| `outputs/plots/exp2_convergence.png` | Experiment: convergence analysis |
| `outputs/plots/exp3_regularization.png` | Experiment: regularization effect |
| `outputs/plots/exp4_initialization.png` | Experiment: init method comparison |
| `outputs/plots/exp5_knn_k.png` | Experiment: KNN k selection |
| `outputs/logs/metrics.txt` | Full metrics report + analysis |

### 14.4 Code Architecture

```
config.py ─────────────────────────────────────► All experiments
     │
     ├── src/data_loader.py ◄── open() only
     │        │
     ├── src/preprocessing.py ◄── z-score normalization
     │        │
     ├── src/gaussian.py ◄── N(x|μ,Σ) PDF from scratch
     │        │
     ├── src/gmm.py ◄── GMMParams + initialization
     │        │
     ├── src/em.py ◄── E-step, M-step, fit_gmm
     │        │
     ├── src/kmeans.py ◄── K-Means from scratch
     │        │
     ├── src/knn.py ◄── KNN from scratch
     │        │
     ├── src/hierarchical.py ◄── Agglomerative from scratch
     │        │
     ├── src/metrics.py ◄── Silhouette, Separation
     │        │
     └── src/visualization.py ◄── All plots (Matplotlib)
```

---

## 15. Results and Analysis

### 15.1 Final Metrics

| Metric | GMM | K-Means | Hierarchical |
|--------|-----|---------|-------------|
| Log-Likelihood | -442.07 | N/A | N/A |
| Silhouette Score | 0.7304 | 0.7380 | 0.7292 |
| Cluster Separation | 2.7175 | 2.7100 | 2.7274 |

**KNN Consistency**: 1.0000 (100%)

### 15.2 Analysis

#### Why GMM is the best model for this data

1. **Probabilistic output**: GMM assigns probabilities (e.g., "this point is 73% likely from cluster 1"), while K-Means and Hierarchical force hard assignment. For boundary points between the two eruption regimes, soft assignment is more physically meaningful.

2. **Covariance modeling**: GMM's full covariance matrices capture the positive correlation between eruption duration and waiting time within each cluster. The confidence ellipses are tilted, matching the data's natural axes. K-Means assumes spherical clusters.

3. **Generative model**: GMM defines p(x), so we can compute the likelihood of new observations, detect outliers, and generate synthetic data. K-Means and Hierarchical cannot.

#### Why K-Means achieves slightly higher silhouette

K-Means' silhouette score (0.738) is marginally higher than GMM's (0.730) because:
- Silhouette score uses **Euclidean distance**, which aligns with K-Means' objective
- GMM's soft assignments create slightly fuzzy boundaries that Euclidean-based silhouette penalizes
- This does NOT mean K-Means is a better model — just that silhouette favors K-Means' distance metric

#### Why KNN achieves 100% consistency

The two Old Faithful clusters are **well-separated** — there's a clear gap between them in the standardized feature space. Local neighbors almost always belong to the same cluster, so KNN perfectly reproduces the labeling. This confirms the clustering is robust and locally coherent.

#### Why Hierarchical produces similar results

Average linkage hierarchical clustering produces results comparable to K-Means because:
- The two clusters are clearly separated (no chaining problems)
- Average linkage doesn't make distributional assumptions, so it captures the natural separation
- The main disadvantage is computational cost: O(N²) pairwise distances, while K-Means is O(N×K)

#### Physical interpretation

The two learned Gaussian components correspond to:
- **Short eruptions** (μ ≈ 2 min, wait ≈ 55 min): The underground water chamber partially empties, requiring less time to refill
- **Long eruptions** (μ ≈ 4.5 min, wait ≈ 80 min): The chamber fully empties, requiring more time to refill

The positive correlation within each cluster (tilted ellipses) reflects the physical relationship: within each regime, longer eruptions deplete slightly more water, requiring slightly longer refill times.

---

## Glossary

| Term | Definition |
|------|-----------|
| **BIC** | Bayesian Information Criterion — model selection metric penalizing complexity |
| **Cluster** | A group of similar data points |
| **Covariance** | Measure of how two variables change together |
| **Dendrogram** | Tree diagram showing hierarchical clustering merges |
| **Determinant** | Scalar value representing the "volume" of a matrix |
| **EM** | Expectation-Maximization — iterative optimization for latent variable models |
| **E-step** | Compute posterior probabilities (responsibilities) |
| **Gaussian** | Normal distribution — bell-shaped probability density |
| **Generative model** | Model that defines a full probability distribution over data |
| **GMM** | Gaussian Mixture Model — sum of weighted Gaussians |
| **Hard assignment** | Each point belongs to exactly one cluster |
| **Inverse** | Matrix A⁻¹ such that A×A⁻¹ = I |
| **Latent variable** | Unobserved variable explaining data structure |
| **Linkage** | Method for computing inter-cluster distance |
| **Log-likelihood** | Log of the probability of data under the model |
| **Mahalanobis distance** | Distance metric accounting for covariance structure |
| **M-step** | Update parameters to maximize expected log-likelihood |
| **MLE** | Maximum Likelihood Estimation |
| **Mixing weight** | Prior probability of each GMM component (πₖ) |
| **PDF** | Probability Density Function |
| **Precision matrix** | Inverse of covariance matrix (Σ⁻¹) |
| **Regularization** | Adding small value to diagonal to prevent singularity |
| **Responsibility** | Posterior probability that component k generated point n |
| **Silhouette score** | Cluster quality metric based on cohesion vs separation |
| **Soft assignment** | Each point has a probability for each cluster |
| **Standardization** | Transform features to zero mean and unit variance |
| **Z-score** | Standardized value: (x - μ) / σ |

---

*Document generated for the GMM from Scratch project on Old Faithful Geyser dataset.*
*All algorithms implemented without ML libraries — Python stdlib + NumPy (linear algebra) + Matplotlib (plots) only.*

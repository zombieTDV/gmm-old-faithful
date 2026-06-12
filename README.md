# Gaussian Mixture Model (GMM) from Scratch on Old Faithful Dataset

A complete, from-scratch Python machine learning system implementing a multivariate **Gaussian Mixture Model (GMM)** optimized via the **Expectation-Maximization (EM) algorithm**. The model is trained on the **Old Faithful Geyser dataset** and compared against three baseline clustering models: **K-Means**, **K-Nearest Neighbors (KNN)**, and **Agglomerative Hierarchical Clustering**.

All probability computations, model logic, preprocessing pipelines, and evaluation metrics are written completely from scratch in pure Python using NumPy (strictly for basic linear algebra/matrix operations) and Matplotlib (for visualization).

## 🚀 Getting Started

### 📋 Prerequisites

- Python 3.10+
- NumPy
- Matplotlib
- olefile (for reading the binary XLS spreadsheet without third-party Excel libraries)

### 💻 Installation

Install the required packages using `pip`:
```bash
python3 -m pip install numpy matplotlib olefile --user
```

### 🏃 Running the Experiments

To run the entire end-to-end data pipeline, model training, baseline evaluations, and final comparisons:
```bash
python3 experiments/run_experiments.py
```

This master script will automatically run:
1. **Preprocessing (`src/preprocessing.py`)**: Loads `data/raw/Old_Faithful_data.xls`, cleans it, standardizes it manually, and saves it to `data/processed/faithful_clean.csv`.
2. **GMM Training (`experiments/run_gmm.py`)**: Initializes and trains the GMM on the processed data, printing learned weights, means, and covariance matrices.
3. **Baselines (`experiments/run_baselines.py`)**: Runs K-Means, Hierarchical clustering, and KNN consistency checks.
4. **Comparison (`experiments/compare_models.py`)**: Consolidates evaluation metrics (Log-Likelihood, Silhouette Score, Separation Distance) and logs them to `outputs/logs/metrics.txt` while plotting the final comparisons.

---

## 📂 Project Structure

```
gmm-old-faithful/
│
├── data/
│   ├── raw/
│   │   └── Old_Faithful_data.xls   # Raw binary Excel spreadsheet
│   ├── processed/
│   │   └── faithful_clean.csv      # Cleaned and standardized dataset
│   
├── src/
│   ├── data_loader.py              # Manual OLE/BIFF8 binary Excel parser
│   ├── preprocessing.py            # Manual cleaning and normalization pipeline
│   ├── gaussian.py                 # Manual multivariate Gaussian PDF & covariance calculations
│   ├── gmm.py                      # GMM parameter representation
│   ├── em.py                       # EM optimization algorithm
│   ├── kmeans.py                   # Manual K-Means clustering algorithm
│   ├── knn.py                      # Manual KNN classifier for consistency check
│   ├── hierarchical.py             # Manual Agglomerative Hierarchical Clustering
│   ├── metrics.py                  # Manual evaluation metrics (Silhouette, Separation, etc.)
│   └── visualization.py            # Matplotlib plotting functions
│
├── experiments/
│   ├── run_gmm.py                  # Driver script for GMM
│   ├── run_baselines.py            # Driver script for K-Means, Hierarchical, and KNN
│   ├── run_experiments.py          # Master orchestrator script
│   └── compare_models.py           # Metrics compiler and report generator
│
├── outputs/
│   ├── plots/                      # Generated visualization PNG files
│   └── logs/                       # Written experimental logs
│
├── docs/
│   ├── INSTRUCTOR.md               # Original project instruction sheet
│   └── implementation_plan.md      # Detailed approved design document
│
├── config.py                       # Global settings and hyperparameters
└── README.md                       # Project overview documentation
```

---

## 🧮 Mathematical Formulas Implemented

### 1. Standardization
$$ x' = \frac{x - \mu}{\sigma} $$

### 2. Multivariate Gaussian Probability Density Function (PDF)
$$ \mathcal{N}(x|\mu,\Sigma) = \frac{1}{(2\pi)^{D/2} |\Sigma|^{1/2}} \exp\left( -\frac{1}{2} (x-\mu)^T \Sigma^{-1} (x-\mu) \right) $$

### 3. Expectation-Maximization (EM) Updates
- **E-Step (Responsibilities)**:
  $$ \gamma_{nk} = \frac{\pi_k \mathcal{N}(x_n|\mu_k,\Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_n|\mu_j,\Sigma_j)} $$
- **M-Step (Parameter Updates)**:
  - **Means**:
    $$ \mu_k = \frac{\sum_{n=1}^N \gamma_{nk} x_n}{\sum_{n=1}^N \gamma_{nk}} $$
  - **Covariances** (with regularization $\epsilon I$):
    $$ \Sigma_k = \frac{\sum_{n=1}^N \gamma_{nk}(x_n-\mu_k)(x_n-\mu_k)^T}{\sum_{n=1}^N \gamma_{nk}} + \epsilon I $$
  - **Mixing Weights**:
    $$ \pi_k = \frac{1}{N} \sum_{n=1}^N \gamma_{nk} $$

### 4. Silhouette Score
$$ s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} $$
- $a(i)$: mean distance to other points in the same cluster.
- $b(i)$: min mean distance to points in another cluster.
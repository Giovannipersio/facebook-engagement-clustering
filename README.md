# Facebook Engagement Clustering

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![UCI](https://img.shields.io/badge/Dataset-UCI-2C3E50?style=flat)](https://archive.ics.uci.edu/dataset/488/facebook+live+sellers+in+thailand)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=flat)](https://github.com/Giovannipersio/facebook-engagement-clustering)

## Overview

This repository contains an unsupervised machine learning project for exploring engagement patterns in Facebook posts published by Thai fashion and cosmetics sellers.

The workflow covers data preprocessing, feature encoding and scaling, dimensionality reduction with Principal Component Analysis (PCA), cluster visualization with t-SNE, and the comparison of K-Means, Agglomerative Clustering, and DBSCAN using the Silhouette Score.

## Project Objective

The objective is to identify groups of Facebook posts with similar engagement behavior based on reactions, comments, shares, content type, and publishing time.

The analysis aims to:

- preprocess numerical, categorical, and temporal features;
- reduce the dimensionality of the feature space while preserving relevant information;
- compare clustering algorithms and hyperparameter configurations;
- evaluate cluster cohesion and separation;
- interpret patterns associated with engagement and post type.

## Dataset

The project uses the [Facebook Live Sellers in Thailand](https://archive.ics.uci.edu/dataset/488/facebook+live+sellers+in+thailand) dataset from the UCI Machine Learning Repository. The notebook downloads it through the `ucimlrepo` package.

The dataset contains Facebook posts from 10 Thai fashion and cosmetics sellers, including photos, videos, status updates, and links.

| Item | Value |
| --- | ---: |
| Rows | 7,050 |
| Original features used | 11 |
| Numerical engagement features | 9 |
| Categorical features | 1 |
| Temporal feature | 1 |
| Missing values | None |

The engagement variables include reactions, comments, shares, likes, loves, wows, hahas, sads, and angry reactions.

## Repository Structure

```text
facebook-engagement-clustering/
|-- notebooks/
|   `-- facebook_engagement_clustering.ipynb
|-- src/                         # Reusable analysis utilities
|-- requirements.txt
|-- make_env.py
|-- .gitignore
`-- README.md
```

## Methodology

### 1. Data Preprocessing

The publication timestamp is decomposed into year, month, day, hour, and weekday features. Categorical variables are transformed with one-hot encoding, and the resulting feature matrix is standardized with `StandardScaler` before dimensionality reduction and clustering.

| Feature Type | Strategy |
| --- | --- |
| Publication timestamp | Extraction of calendar and time features |
| Categorical features | One-hot encoding |
| Numerical features | Standard scaling |
| Missing values | Completeness check |

### 2. Dimensionality Reduction

PCA is applied to reduce the feature space to 15 principal components, preserving approximately 85% of the cumulative variance. Two-dimensional t-SNE projections are also used to inspect the resulting clusters visually.

### 3. Clustering

Three clustering approaches are evaluated:

- **K-Means:** the number of clusters is investigated using inertia, the elbow method, and silhouette analysis.
- **Agglomerative Clustering:** combinations of cluster count, distance metric, and linkage method are compared.
- **DBSCAN:** `eps` and `min_samples` configurations are tested to identify density-based structures and noise.

### 4. Evaluation

The models are compared primarily with the Silhouette Score. Silhouette plots, reordered distance matrices, cluster profiles, post-type distributions, PCA projections, and t-SNE visualizations complement the numerical evaluation.

## Results

PCA reduced the data to 15 components while retaining approximately 85% of its cumulative variance.

| Model | Configuration | Silhouette Score |
| --- | --- | ---: |
| Agglomerative Clustering | 2 clusters, Manhattan distance, average linkage | 0.8191 |
| K-Means | 11 clusters | 0.3320 |
| DBSCAN | `eps=1.9`, `min_samples=5` | 0.3022 |

Agglomerative Clustering obtained the highest score, but its result was strongly imbalanced, with most observations assigned to one cluster. The score must therefore be interpreted alongside cluster sizes, individual silhouette values, and domain relevance rather than as conclusive evidence of the best segmentation.

The exploratory profiles indicate differences in engagement levels, content types, and publication timing across groups. High-engagement clusters are often associated with photos or videos, while other groups contain posts with lower interaction levels.

## How to Reproduce

### 1. Clone the repository

```bash
git clone https://github.com/Giovannipersio/facebook-engagement-clustering.git
cd facebook-engagement-clustering
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

Open and execute:

```text
notebooks/facebook_engagement_clustering.ipynb
```

The notebook downloads the dataset from UCI, preprocesses the features, performs dimensionality reduction, compares clustering models, and generates the evaluation visualizations.

## Requirements

Main libraries used in this project:

| Library | Purpose |
| --- | --- |
| `numpy`, `pandas` | Data manipulation and numerical operations |
| `matplotlib`, `seaborn` | Data and cluster visualization |
| `scikit-learn` | Preprocessing, PCA, t-SNE, clustering, and metrics |
| `missingno` | Missing-value visualization |
| `ucimlrepo` | Dataset download |
| `tqdm` | Progress tracking during parameter searches |
| `ipykernel` | Notebook execution |

See `requirements.txt` for dependency specifications.

## Limitations

- The analysis uses a single dataset from a specific commercial and geographic context.
- The data represents posts from only 10 Facebook sellers.
- A high average Silhouette Score may conceal highly imbalanced or very small clusters.
- t-SNE projections are intended for visual exploration and do not preserve every high-dimensional relationship.
- The identified groups require business and domain validation before being used in marketing decisions.

## Next Steps

- Refactor the original academic notebook into a reproducible portfolio notebook.
- Move reusable preprocessing, visualization, and evaluation functions into `src/`.
- Add cluster-size and stability analyses to complement the Silhouette Score.
- Improve cluster interpretation with systematic feature profiles.
- Compare additional dimensionality-reduction and clustering techniques.
- Add automated checks for notebook reproducibility and code quality.

## Author

Developed by **Giovanni Persio**.

The original academic project was developed in collaboration with **Rodrigo Corrêa Fardin** and **Marcelo Alves Otaviano Botelho** as part of a data science specialization course.

This repository is a portfolio-oriented revision focused on unsupervised learning, dimensionality reduction, clustering evaluation, and reproducible data analysis.

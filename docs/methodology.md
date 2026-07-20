# Methodology

[Back to the project README](../README.md)

This document describes the analytical decisions implemented by the five
project notebooks. The executed notebooks and persisted artifacts are the
authoritative specification when implementation details change.

## 1. Problem Definition

The project is an unsupervised analysis of Facebook post engagement. It seeks
groups of posts with similar interaction, content-format, and publication-time
patterns without using a target variable.

The intended outcome is an interpretable exploratory segmentation. It is not a
causal study, a recommendation engine, or a production scoring service.

## 2. Data Collection and Contract

Notebook 01 downloads the feature table for UCI dataset 488 and requires the
following 11 fields:

- `status_type` and `status_published`;
- `num_reactions`, `num_comments`, and `num_shares`;
- `num_likes`, `num_loves`, `num_wows`, `num_hahas`, `num_sads`, and
  `num_angrys`.

The collected table contains 7,050 rows. The project validates its structure
before saving `data/raw/facebook_live_sellers.csv`.

## 3. Preprocessing

Notebook 02 creates a deterministic `record_id` so every downstream artifact
can be checked and joined safely.

### 3.1. Data-quality decisions

- Missing values are rejected.
- Numeric engagement columns must be non-negative.
- Column labels must be unique.
- The 54 exact feature duplicates are retained. Without original post and
  seller identifiers, identical feature values are not sufficient evidence of
  duplicated observations.
- `num_reactions` remains in the cleaned analytical table but is excluded from
  the model matrix. It is almost fully explained by the individual reaction
  counts and would otherwise duplicate their influence on distance.

### 3.2. Feature engineering

The publication timestamp is parsed into:

- year;
- month;
- weekday name and weekday number;
- fractional publication hour.

Month, weekday number, and hour receive sine/cosine encodings so the end and
beginning of each cycle remain close in the feature space.

### 3.3. Transformation pipelines

| Feature group | Transformation |
| --- | --- |
| Comments, shares, and individual reactions | `log1p` followed by standard scaling |
| Publication year and cyclical features | Standard scaling |
| Status type | Dense one-hot encoding with unknown-category handling |

The final modeling table has 7,050 rows and 19 numeric features, excluding
`record_id`. The fitted `ColumnTransformer` is persisted for reproducibility.

## 4. Dimensionality Reduction

Notebook 03 first fits an unrestricted PCA model to the 19-feature matrix. It
then selects the smallest number of components whose cumulative explained
variance reaches the configured threshold of 85%.

The current run retains 10 principal components and 87.79% of the cumulative
variance. The notebook also exports:

- variance explained by every component;
- feature weights for the retained components;
- reconstruction error and diagnostic plots;
- the fitted PCA object.

The two-dimensional PCA chart is only a visual projection. All clustering
models are fitted in the complete retained 10-component space.

## 5. Clustering Search

Notebook 04 evaluates 122 candidate configurations.

### 5.1. Search spaces

| Algorithm | Search |
| --- | --- |
| K-Means | 2–15 clusters, `n_init=20`, `random_state=42` |
| Agglomerative Clustering | 2–15 clusters across Ward, complete, and average linkage configurations with compatible metrics |
| DBSCAN | Four `min_samples` values and six data-derived neighborhood-distance quantiles |

This produces 14 K-Means, 84 agglomerative, and 24 DBSCAN candidates.

### 5.2. Evaluation metrics

- sampled Silhouette Score, using up to 2,000 observations and a fixed seed;
- Calinski–Harabasz Score;
- Davies–Bouldin Score;
- number and relative size of clusters;
- noise share and non-noise coverage for DBSCAN.

### 5.3. Eligibility policy

A candidate must produce at least two valid groups, a minimum cluster size of
50, and no cluster larger than 90% of evaluated observations. DBSCAN must also
cover at least 80% of the data after noise is excluded.

These rules prevent a high internal metric from automatically winning when it
comes from a nearly single-cluster solution or a tiny residual group.

### 5.4. Perturbation stability

The best eligible candidate from each algorithm family is fitted repeatedly
after five small random perturbations of the PCA scores. Agreement with the
baseline assignments is measured with the Adjusted Rand Index (ARI).

A mean ARI of at least 0.80 is considered stable. Final ranking prioritizes:

1. Silhouette Score, descending;
2. Calinski–Harabasz Score, descending;
3. Davies–Bouldin Score, ascending.

The current policy selects a two-cluster Ward agglomerative model.

## 6. Cluster Interpretation

Notebook 05 joins cleaned records, PCA scores, and selected labels through
`record_id` with one-to-one validation. It analyzes:

- cluster count, size, and imbalance;
- mean and median engagement profiles;
- within-cluster content-type composition;
- weekday and publication-daypart distributions;
- PCA-space cluster centroids and component weights;
- Cramér's V for categorical association strength.

Means and medians are retained together because engagement counts are strongly
right-skewed. Cluster identifiers remain arbitrary and should not be treated as
ratings.

## 7. Reproducibility

- Python and package constraints are declared in `pyproject.toml`.
- Exact resolved package versions are stored in `uv.lock`.
- Notebooks locate the project root dynamically and use project-relative
  artifact paths.
- Operational steps use structured logging.
- Every downstream notebook verifies required files, columns, row alignment,
  and output contracts.
- Randomized model evaluation uses `random_state=42`.

Run the complete workflow with the commands and notebook order documented in
the [main README](../README.md#reproduce-the-project).

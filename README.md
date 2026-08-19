# Stratified_GM

The main implementation is contained in `stratified_gm.py`. The accompanying `Stratified_GW_demo.ipynb` notebook demonstrates the pipeline on synthetic geometric surfaces and CEBRA embeddings.


## Requirements

- Python 3.10 or later
- NumPy
- SciPy
- POT (Python Optimal Transport)
- hdbscan
- Matplotlib
- umap-learn
- Plotly
- Jupyter, for running the demonstration notebook

Install the dependencies with:

```bash
python -m pip install numpy scipy POT hdbscan matplotlib umap-learn plotly jupyter
```

The package imported as `ot` is installed from PyPI under the name `POT`.

## Extracting CEBRA embeddings

To extract embeddings from a MATLAB experiment file, import `CEBRAAnalysis` from `cebra_dim_reduction`, instantiate it with the data path and `session_choose=True`, and call `run_analysis` with the desired embedding output directory. The complete call can be written as `CEBRAAnalysis(data_path, session_choose=True).run_analysis(embedding_folder_path)`.


## Basic usage

Run the notebook from the repository root so that Python can find `stratified_gm.py`:

```bash
jupyter notebook Stratified_GW_demo.ipynb
```

Alternatively, import the analysis function into another notebook or Python script:

```python
from stratified_gm import run_analysis

results = run_analysis(
    point_clouds,
    plot_path="results/validation_mmd.png",
    validation_fraction=0.5,
    n_quantile=100,
    metric="Euclidean",
    normalize=False,
    min_cluster_size=5,
    min_samples=5,
    n_permutations=10_000,
    random_state=42,
)
```

`point_clouds` should be a collection of two-dimensional arrays:

```python
point_clouds = [
    cloud_0,  # shape (n_points_0, n_features_0)
    cloud_1,  # shape (n_points_1, n_features_1)
    ...
]
```

Point clouds may contain different numbers of points. Each cloud must contain at least two points so that nonempty discovery and validation subsets can be formed.
## Output

`run_analysis` returns a dictionary keyed by pairs of HDBSCAN cluster labels:

```python
{
    (0, 1): {
        "mmd2": 0.2868,
        "p_value": 0.0001,
        "n_clouds_cluster_1": 20,
        "n_clouds_cluster_2": 20,
    },
    (0, 2): {
        "mmd2": 0.9614,
        "p_value": 0.0001,
        "n_clouds_cluster_1": 20,
        "n_clouds_cluster_2": 20,
    },
}
```

The function also saves an annotated MMD² matrix to `plot_path`. Parent directories are created automatically when necessary.

The reported p-values use the finite-permutation correction

```text
(number of permuted statistics >= observed statistic + 1)
----------------------------------------------------------------
                    (number of permutations + 1)
```

Consequently, the smallest possible p-value is `1 / (n_permutations + 1)`.

## Main Functions

| Function | Purpose |
|---|---|
| `split_point_clouds` | Randomly divides every cloud into non-overlapping discovery and validation sub-clouds. |
| `geodesic_distance_matrix` | Computes pairwise angular distances after projecting points onto the unit sphere. |
| `compute_stratified_wasserstein_distances` | Computes the pairwise optimal-transport distance matrix between point-cloud distance profiles. |
| `cluster` | Runs HDBSCAN using a precomputed distance matrix and returns cluster labels and index groups. |
| `distance_to_rbf_median` | Converts a distance matrix into an RBF kernel using the median positive distance as the bandwidth. |
| `precomputed_kernel_permutation_test` | Performs a two-sample MMD permutation test using a precomputed kernel. |
| `pairwise_cluster_mmd_precomputed` | Applies the MMD permutation test to every pair of clusters. |
| `mmd_results_to_matrix` | Converts pairwise MMD² results into a symmetric matrix. |
| `plot_pairwise_matrices` | Plots and optionally saves one or two annotated pairwise matrices. |
| `run_analysis` | Runs the complete discovery-validation analysis pipeline. |


## Distance construction

For a point cloud \(X=\{x_1,\ldots,x_n\}\), the code first calculates its within-cloud distance matrix. Each point \(x_i\) is then represented by

```text
[Q_0(d(x_i, X)), Q_1/n_quantile(d(x_i, X)), ..., Q_1(d(x_i, X))],
```

where `Q` denotes an empirical quantile. The optimal transport cost between the resulting quantile-vector distributions defines the distance between two clouds. Uniform mass is assigned to every point in each cloud.

This construction depends on within-cloud distances rather than the original coordinates. With the Euclidean metric it is therefore invariant to point ordering, translation, rotation, and reflection.

Two within-cloud metrics are supported:

- `metric="Euclidean"`: ordinary Euclidean distances.
- `metric="Geodesic"`: angular distances after projection onto the unit sphere.

When `normalize=True`, each within-cloud distance matrix is divided by its median before quantiles are calculated. This removes overall scale, but requires every cloud to have a strictly positive median distance.

## Discovery-validation testing

The MMD test is performed only on validation data. HDBSCAN labels are learned from the discovery sub-clouds and transferred by index to their matched validation sub-clouds; the validation clouds are not reclustered.

The validation distance matrix is converted into a kernel using

```text
K[i, j] = exp(-D[i, j]² / (2 sigma²)),
```

## Demonstration notebook

`Stratified_GW_demo.ipynb` contains two examples:

1. Synthetic point clouds sampled from sphere, torus, cube, and tetrahedron surfaces, optionally subjected to independent random rotations.
2. Collections of sub-clouds sampled from CEBRA time and behavior embeddings (obtained from the CEBRA documentation) stored as NumPy arrays.

The CEBRA example expects files with names such as:

```text
Consistency_test_data/time3_embedding_values_rat_0.npy
Consistency_test_data/posdir3_embedding_values_rat_0.npy
```

through rat index `3`. These data files are not required for the synthetic demonstration.

## Reproducibility

Set `random_state` in `run_analysis` to reproduce the discovery-validation split and the permutation tests:

```python
results = run_analysis(
    point_clouds,
    plot_path="results/mmd.png",
    random_state=42,
)
```




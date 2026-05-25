# Cebra_GW

Cebra_dim_R created by Deven Shidfar. 
CEBRAAnalysis.run_analysis() extracts neural embeddings through CEBRA.

## Installation

Install the main dependencies:

```bash
pip install numpy scipy matplotlib plotly umap-learn hdbscan pot scikit-learn torch
```

Optional/project-specific dependencies:

```bash
pip install utonia
```

Some parts of the notebook also refer to local paths and a `CEBRAAnalysis` class. Those may need to be adjusted before running on a new machine.

## Basic Usage

Import the functions from the file or run the notebook cells first.

```python
import numpy as np

# Generate point-cloud samples
samples = sample_four_surfaces(
    N=10,
    n=200,
    random_rotate=True,
    rng=42
)

print(samples.shape)
# (40, 200, 3)
```

Each sample is a point cloud with shape:

```python
(n_points, 3)
```

The full dataset has shape:

```python
(n_clouds, n_points, 3)
```

## Main Functions

### `sample_sphere_surface(N, n, d=3, rng=None)`

Generates `N` point clouds sampled uniformly from the surface of a unit sphere.

```python
sphere_samples = sample_sphere_surface(N=5, n=100, rng=0)
```

Returns:

```python
shape = (N, n, d)
```

### `sample_torus_surface(N, n, R=2.0, r=1.0, rng=None)`

Generates point clouds sampled from the surface of a torus in 3D.

```python
torus_samples = sample_torus_surface(
    N=5,
    n=100,
    R=2.0,
    r=0.5,
    rng=0
)
```

Arguments:

- `R`: major radius
- `r`: minor radius
- Must satisfy `R > r > 0`

Returns:

```python
shape = (N, n, 3)
```

### `sample_cube_surface(N, n, side_length=2.0, rng=None)`

Generates point clouds sampled uniformly from the surface of a cube centered at the origin.

```python
cube_samples = sample_cube_surface(
    N=5,
    n=100,
    side_length=2.0,
    rng=0
)
```

Returns:

```python
shape = (N, n, 3)
```

### `sample_tetrahedron_surface(N, n, side_length=2.0, rng=None)`

Generates point clouds sampled from the surface of a regular tetrahedron.

```python
tetra_samples = sample_tetrahedron_surface(
    N=5,
    n=100,
    side_length=2.0,
    rng=0
)
```

Returns:

```python
shape = (N, n, 3)
```

### `sample_three_surfaces(N, n, ...)`

Generates point clouds from three geometric surfaces:

1. sphere
2. torus
3. cube

```python
samples = sample_three_surfaces(
    N=10,
    n=200,
    rng=42
)
```

Returns:

```python
shape = (3 * N, n, 3)
```

The ordering is:

```python
samples[0:N]       # sphere
samples[N:2*N]     # torus
samples[2*N:3*N]   # cube
```

### `sample_four_surfaces(N, n, ..., random_rotate=True, rng=None)`

Generates point clouds from four geometric surfaces:

1. sphere
2. torus
3. cube
4. tetrahedron

```python
samples = sample_four_surfaces(
    N=10,
    n=200,
    random_rotate=True,
    rng=42
)
```

Returns:

```python
shape = (4 * N, n, 3)
```

If `random_rotate=True`, each point cloud is randomly rotated in 3D.

### `apply_random_rotations(samples, rng=None)`

Applies a different random 3D rotation to each point cloud.

```python
rotated_samples = apply_random_rotations(samples, rng=42)
```

Input:

```python
shape = (num_clouds, n_points, 3)
```

Output:

```python
shape = (num_clouds, n_points, 3)
```

## Distance and Kernel Functions

### `geodesic_distance_matrix(points, eps=1e-12)`

Projects points onto the unit sphere and computes the pairwise geodesic distance matrix.

```python
D = geodesic_distance_matrix(samples[0])
```

Returns:

```python
shape = (n_points, n_points)
```

Distances are measured in radians.

### `compute_WassKernel_stratified_improved(data, n_quantile=100, metric='Euclidean', normalize=False)`

Computes a pairwise distance matrix between point clouds using quantiles of internal point-cloud distances and optimal transport.

```python
D = compute_WassKernel_stratified_improved(
    data=samples,
    n_quantile=50,
    metric="Euclidean",
    normalize=False
)
```

Arguments:

- `data`: list or array of point clouds
- `n_quantile`: number of quantile bins
- `metric`: `"Euclidean"` or `"Geodesic"`
- `normalize`: whether to normalize each distance matrix by its median

Returns:

```python
shape = (n_clouds, n_clouds)
```

### `compute_kernel_matrix(data, normalize=False, n_align=100, metric='Euclidean', norm_const=100)`

Wrapper around `compute_WassKernel_stratified_improved`.

```python
K = compute_kernel_matrix(
    data=samples,
    normalize=False,
    n_align=50,
    metric="Euclidean"
)
```

Despite the name, this currently returns a pairwise distance matrix.

### `pairwise_hausdorff_distance_matrix(point_clouds, return_directed=False)`

Computes the pairwise Hausdorff distance between point clouds.

```python
H = pairwise_hausdorff_distance_matrix(samples)
```

To also return directed Hausdorff distances:

```python
H, directed_H = pairwise_hausdorff_distance_matrix(
    samples,
    return_directed=True
)
```

Returns:

```python
H.shape = (n_clouds, n_clouds)
```

## Clustering and Visualization

### `cluster(D, min_cluster_size, min_samples)`

Runs HDBSCAN clustering using a precomputed distance matrix.

```python
cluster_labels, cluster_indices = cluster(
    D,
    min_cluster_size=5,
    min_samples=2
)
```

Returns:

- `cluster_labels`: unique cluster IDs
- `cluster_indices`: dictionary mapping each cluster ID to sample indices

Noise points are labeled `-1`.

### `plot_umap_from_cluster_indices(data, cluster_labels, cluster_indices, ...)`

Fits UMAP and plots the clustered data in 2D.

```python
embedding = plot_umap_from_cluster_indices(
    data=D,
    cluster_labels=cluster_labels,
    cluster_indices=cluster_indices,
    metric="precomputed",
    title="UMAP of Point-Cloud Distances"
)
```

Returns:

```python
embedding.shape = (n_samples, 2)
```

### `plot_points_3d(points, index, mode="markers", marker_size=4, show_axes=True, title="3D Points")`

Displays a 3D point cloud using Plotly.

```python
fig = plot_points_3d(
    samples[0],
    index=0,
    title="Example Point Cloud"
)
```

This function also writes an HTML file to a hardcoded local path. You may want to replace that path before using it on another machine.

## MMD Testing

### `median_heuristic(X)`

Estimates an RBF kernel bandwidth using the median pairwise squared distance.

```python
gamma = median_heuristic(X)
```

### `mmd_permutation_test(X, Y, n_permutations=1000, gamma=None, random_state=None)`

Runs a two-sample MMD permutation test between two groups of data.

```python
mmd2, p_value = mmd_permutation_test(
    X,
    Y,
    n_permutations=1000,
    random_state=42
)
```

Returns:

- `mmd2`: observed MMD statistic
- `p_value`: permutation-test p-value

### `pairwise_cluster_mmd_tests(data, cluster_labels, n_permutations=1000, random_state=None)`

Runs MMD tests between every pair of clusters.

```python
results = pairwise_cluster_mmd_tests(
    data=embedding,
    cluster_labels=labels,
    n_permutations=1000,
    random_state=42
)
```

Returns a dictionary:

```python
{
    (cluster_a, cluster_b): {
        "mmd2": ...,
        "p_value": ...,
        "n_cluster_1": ...,
        "n_cluster_2": ...
    }
}
```

## Example Workflow

```python
# 1. Generate synthetic geometric point clouds
samples = sample_four_surfaces(
    N=10,
    n=200,
    random_rotate=True,
    rng=42
)

# 2. Compute pairwise point-cloud distances
D = compute_kernel_matrix(
    samples,
    n_align=50,
    metric="Euclidean"
)

# 3. Cluster using HDBSCAN
cluster_labels, cluster_indices = cluster(
    D,
    min_cluster_size=5,
    min_samples=2
)

# 4. Visualize with UMAP
embedding = plot_umap_from_cluster_indices(
    D,
    cluster_labels,
    cluster_indices,
    metric="precomputed",
    title="UMAP of Stratified Geometric Samples"
)

# 5. Plot one point cloud
plot_points_3d(samples[0], index=0, title="Example Surface Sample")
```

## Notes / Known Issues

- `compute_kernel_matrix` currently returns a distance matrix, not a kernel matrix.
- `plot_points_3d` saves HTML output to a hardcoded local Windows path. Update this path before running elsewhere.
- `extract_embeddings` depends on `CEBRAAnalysis`, but that import is commented out in the notebook.
- `pairwise_cluster_mmd_tests` uses `combinations`, so make sure to import it:

```python
from itertools import combinations

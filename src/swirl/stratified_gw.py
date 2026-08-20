import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import umap
from itertools import combinations
from numpy.typing import ArrayLike, NDArray
from pathlib import Path
from matplotlib.colors import Normalize
import numpy as np
import plotly.graph_objects as go
import ot
import hdbscan
from scipy.spatial.transform import Rotation




def split_point_clouds(
    point_clouds: list[ArrayLike],
    validation_fraction: float = 0.5,
    random_state: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Randomly partition every point cloud into non-overlapping discovery
    and validation sub-clouds.

    Parameters
    ----------
    point_clouds
        Collection of point clouds. Each cloud must have shape
        (n_points, n_features).

    validation_fraction
        Fraction of each cloud assigned to validation. Must be strictly
        between 0 and 1.

    random_state
        Seed for reproducible splitting.

    Returns
    -------
    discovery_clouds
        List containing the discovery portion of each point cloud.

    validation_clouds
        List containing the matched validation portion of each point cloud.

    Notes
    -----
    discovery_clouds[i] and validation_clouds[i] are non-overlapping
    partitions of point_clouds[i].
    """
    if not 0 < validation_fraction < 1:
        raise ValueError(
            "validation_fraction must be strictly between 0 and 1."
        )

    rng = np.random.default_rng(random_state)

    discovery_clouds = []
    validation_clouds = []

    for cloud_index, cloud in enumerate(point_clouds):
        cloud = np.asarray(cloud)

        if cloud.ndim != 2:
            raise ValueError(
                f"Point cloud {cloud_index} must have shape "
                "(n_points, n_features)."
            )

        n_points = len(cloud)

        if n_points < 2:
            raise ValueError(
                f"Point cloud {cloud_index} must contain at least two points."
            )

        n_validation = int(np.floor(n_points * validation_fraction))

        # Ensure that neither partition is empty.
        n_validation = min(max(n_validation, 1), n_points - 1)

        shuffled_indices = rng.permutation(n_points)
        validation_indices = shuffled_indices[:n_validation]
        discovery_indices = shuffled_indices[n_validation:]

        discovery_clouds.append(cloud[discovery_indices])
        validation_clouds.append(cloud[validation_indices])

    return discovery_clouds, validation_clouds


def geodesic_distance_matrix(points, eps=1e-12):
    """
    Project points onto the unit sphere, then compute the pairwise
    geodesic distance matrix.

    Parameters
    ----------
    points : array-like, shape (n_points, dim)
        Input points.
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    D : ndarray, shape (n_points, n_points)
        Pairwise geodesic distance matrix, in radians.
    """
    points = np.asarray(points, dtype=float)

    # Project onto unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    unit_points = points / np.clip(norms, eps, None)

    # Cosine of pairwise angles
    cos_theta = unit_points @ unit_points.T

    # Numerical safety
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Geodesic distance on unit sphere
    D = np.arccos(cos_theta)

    return D


def compute_stratified_wasserstein_distances(data, n_quantile = 100, metric='Euclidean',normalize=False):
    """
    Compute pairwise stratified Wasserstein distances between point clouds.

    Each point in a cloud is represented by the quantiles of its distances
    to all other points in the same cloud. A point cloud is consequently
    represented as an empirical distribution of these distance-quantile
    vectors. The distance between two point clouds is the optimal transport
    cost between their respective collections of quantile vectors, using
    uniform weights and Euclidean ground cost.

    Parameters
    ----------
    data : sequence of array-like
        Collection of point clouds. ``data[i]`` must be a two-dimensional
        array with shape ``(n_points_i, n_features_i)``. Point clouds may
        contain different numbers of points.

    n_quantile : int, default=100
        Number of intervals in the quantile grid. Quantiles are evaluated
        at ``n_quantile + 1`` equally spaced levels between 0 and 1,
        including both endpoints.

    metric : {"Euclidean", "Geodesic"}, default="Euclidean"
        Metric used to calculate distances between points within each cloud.

        - ``"Euclidean"`` uses ordinary Euclidean distance.
        - ``"Geodesic"`` uses ``geodesic_distance_matrix`` and therefore
          assumes that the points can be projected onto the unit sphere.

    normalize : bool, default=False
        If True, divide each within-cloud distance matrix by its median
        before calculating the quantile representation. This removes the
        overall distance scale of each point cloud. The median must be
        strictly positive.

    Returns
    -------
    distances : ndarray of shape (n_clouds, n_clouds)
        Symmetric matrix of pairwise stratified Wasserstein distances.
        ``distances[i, j]`` is the optimal transport cost between the
        distance-quantile representations of point clouds ``i`` and ``j``.
        The diagonal entries are zero.

    Raises
    ------
    ValueError
        If ``metric`` is not ``"Euclidean"`` or ``"Geodesic"``.

    ValueError
        If ``n_quantile`` is not a positive integer.

    Notes
    -----
    This function does not calculate ordinary Wasserstein distance directly
    between the original point coordinates. Instead, it compares empirical
    distributions of within-cloud distance profiles. Consequently, the
    representation is invariant to point ordering and, for Euclidean
    distances, to translations, rotations, and reflections of a cloud.

    ``ot.emd2`` returns the optimal transport objective associated with the
    supplied Euclidean cost matrix. Despite its name, the returned value is
    not squared here because the ground-cost matrix contains ordinary
    Euclidean distances rather than squared distances.
    """
    if metric not in {"Euclidean", "Geodesic"}:
        raise ValueError(
            "metric must be either 'Euclidean' or 'Geodesic'."
        )

    if not isinstance(n_quantile, int) or n_quantile < 1:
        raise ValueError("n_quantile must be a positive integer.")

    n = len(data)


    quantiles = [0]*n
    for i in range(n):
        if metric == 'Euclidean':
            C1 = sp.spatial.distance.cdist(data[i], data[i])
        elif metric == 'Geodesic':
            C1 = geodesic_distance_matrix(data[i])
        if normalize:
            median_distance = float(np.median(C1))

            if not np.isfinite(median_distance):
                raise ValueError(
                    f"Point cloud {i} has a non-finite median distance."
                )

            if median_distance <= 0:
                raise ValueError(
                f"Point cloud {i} has a median distance of "
                f"{median_distance}. Normalization requires a "
                "strictly positive median distance."
            )

            C1 = C1 / median_distance

        quantiles[i] = np.zeros((C1.shape[0],n_quantile+1))
        for j in range(C1.shape[0]):
            quantiles[i][j,:] = np.quantile(C1[j,:], np.linspace(0, 1, n_quantile+1))
            
        
    
    distances = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            M = sp.spatial.distance.cdist(quantiles[i], quantiles[j])
            
            a = np.ones((quantiles[i].shape[0],))
            a = a/a.sum()
            b = np.ones((quantiles[j].shape[0],))
            b = b/b.sum()
            distances[i,j] = ot.emd2(a, b, M)
            distances[j,i] = distances[i,j]


    return distances

def cluster(D, min_cluster_size, min_samples):

    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,  
        min_samples=min_samples,         
        cluster_selection_method='leaf'
    )

    clusters = clusterer.fit_predict(D)
    cluster_labels = np.unique(clusters)    
    cluster_indices = {
    c: np.where(clusters == c)[0]
    for c in cluster_labels
    }
    return cluster_labels, cluster_indices


def plot_umap_from_cluster_indices(
    data,
    cluster_labels,
    cluster_indices,
    n_neighbors=15,
    min_dist=0.1,
    metric="precomputed",
    random_state=42,
    figsize=(8, 6),
    s=20,
    alpha=0.8,
    title="UMAP plot by cluster",
    plot_noise=True,
):
    """
    Fit UMAP on data and plot clusters using precomputed cluster_indices.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Original data matrix.

    cluster_labels : array-like
        Unique cluster labels returned by your cluster() function.

    cluster_indices : dict
        Dictionary mapping cluster label -> sample indices.

    plot_noise : bool
        Whether to plot HDBSCAN noise points labeled as -1.

    Returns
    -------
    embedding : ndarray, shape (n_samples, 2)
        The UMAP embedding.
    """

    data = np.asarray(data)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )

    embedding = reducer.fit_transform(data)

    plt.figure(figsize=figsize)

    for c in cluster_labels:
        if c == -1 and not plot_noise:
            print('Unassigned data points, possible noise')
            continue

        idx = cluster_indices[c]

        label = "Noise" if c == -1 else f"Cluster {c}"

        plt.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            s=s,
            alpha=alpha,
            label=label,
        )

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return embedding

def distance_to_rbf_median(D):
    D = np.asarray(D, dtype=float)

    distances = D[np.triu_indices_from(D, k=1)]
    distances = distances[distances > 0]

    if distances.size == 0:
        raise ValueError(
            "Cannot estimate sigma because all distances are zero."
        )

    sigma = float(np.median(distances))

    K = np.exp(
        -(D**2) / (2.0 * sigma**2)
    )
    np.fill_diagonal(K, 1.0)
    
    return K, sigma

def compute_kernel_matrix(data, normalize=False, n_align=100, metric='Euclidean',norm_const=100):
    K = compute_stratified_wasserstein_distances(data, metric=metric,normalize=normalize,n_quantile=n_align)
    K = distance_to_rbf_median(K)[0]
    return K

def mmd2_from_kernel(K, indices_x, indices_y):
    """
    Compute biased MMD² between two groups using a precomputed kernel.

    Each observation represented by K is one complete point cloud.
    """
    indices_x = np.asarray(indices_x)
    indices_y = np.asarray(indices_y)

    K_xx = K[np.ix_(indices_x, indices_x)]
    K_yy = K[np.ix_(indices_y, indices_y)]
    K_xy = K[np.ix_(indices_x, indices_y)]

    mmd2 = (
        K_xx.mean()
        + K_yy.mean()
        - 2.0 * K_xy.mean()
    )

    return mmd2

def precomputed_kernel_permutation_test(
    K,
    indices_x,
    indices_y,
    n_permutations=1000,
    random_state=None,
):
    """
    Permutation MMD test using a precomputed point-cloud kernel.

    Parameters
    ----------
    K : ndarray, shape (n_clouds, n_clouds)
        Precomputed kernel between whole point clouds.

    indices_x, indices_y : array-like
        Indices of point clouds belonging to the two clusters.
    """
    rng = np.random.default_rng(random_state)

    indices_x = np.asarray(indices_x, dtype=int)
    indices_y = np.asarray(indices_y, dtype=int)

    n_x = len(indices_x)
    n_y = len(indices_y)

    combined_indices = np.concatenate([indices_x, indices_y])

    observed_mmd2 = mmd2_from_kernel(
        K,
        indices_x,
        indices_y,
    )

    permuted_mmd2 = np.empty(n_permutations)

    for permutation in range(n_permutations):
        permuted_indices = rng.permutation(combined_indices)

        permuted_x = permuted_indices[:n_x]
        permuted_y = permuted_indices[n_x:n_x + n_y]

        permuted_mmd2[permutation] = mmd2_from_kernel(
            K,
            permuted_x,
            permuted_y,
        )

    p_value = (
        np.count_nonzero(permuted_mmd2 >= observed_mmd2) + 1
    ) / (
        n_permutations + 1
    )

    return observed_mmd2, p_value


def pairwise_cluster_mmd_precomputed(
    kernel_matrix,
    cluster_indices,
    n_permutations=1000,
    random_state=None,
):
    """
    Pairwise MMD permutation tests between clusters of point clouds.

    Parameters
    ----------
    kernel_matrix : ndarray, shape (n_clouds, n_clouds)
        kernel_matrix[i, j] compares entire point clouds i and j.

    cluster_indices : dict
        Maps each cluster label to its point-cloud indices.

    Returns
    -------
    results : dict
        Pairwise MMD² statistics and permutation p-values.
    """
    K = np.asarray(kernel_matrix, dtype=float)


    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(
            "kernel_matrix must be a square matrix with shape "
            "(n_clouds, n_clouds)."
        )

    if not np.allclose(K, K.T, rtol=1e-7, atol=1e-10):
        raise ValueError("kernel_matrix must be symmetric.")

    validated_indices = {}

    for label, indices in cluster_indices.items():
        indices = np.asarray(indices, dtype=int)

        if indices.ndim != 1:
            raise ValueError(
                f"Indices for cluster {label} must be one-dimensional."
            )

        if len(indices) == 0:
            raise ValueError(f"Cluster {label} is empty.")

        if np.any(indices < 0) or np.any(indices >= K.shape[0]):
            raise IndexError(
                f"Cluster {label} contains indices outside the kernel "
                f"matrix range 0 to {K.shape[0] - 1}."
            )

        validated_indices[label] = indices

    rng = np.random.default_rng(random_state)
    results = {}

    for cluster_1, cluster_2 in combinations(
        validated_indices.keys(),
        2,
    ):
        indices_1 = validated_indices[cluster_1]
        indices_2 = validated_indices[cluster_2]

        mmd2, p_value = precomputed_kernel_permutation_test(
            K,
            indices_1,
            indices_2,
            n_permutations=n_permutations,
            random_state=rng.integers(0, 2**32 - 1),
        )

        results[(cluster_1, cluster_2)] = {
            "mmd2": mmd2,
            "p_value": p_value,
            "n_clouds_cluster_1": len(indices_1),
            "n_clouds_cluster_2": len(indices_2),
        }

    return results

def mmd_results_to_matrix(results, cluster_labels=None):
    """
    Convert pairwise MMD² results into a symmetric matrix.

    Parameters
    ----------
    results : dict
        Dictionary of the form:
        {
            (cluster_1, cluster_2): {
                "mmd2": value,
                ...
            },
            ...
        }

    cluster_labels : sequence or None
        Desired cluster ordering. If None, labels are inferred and sorted.

    Returns
    -------
    mmd_matrix : ndarray, shape (n_clusters, n_clusters)
        Symmetric matrix of pairwise MMD² values.

    label_order : ndarray
        Cluster label corresponding to each matrix row and column.
    """
    if cluster_labels is None:
        cluster_labels = sorted({
            label
            for pair in results
            for label in pair
        })

    cluster_labels = np.asarray(cluster_labels)
    label_to_position = {
        label: i for i, label in enumerate(cluster_labels)
    }

    n_clusters = len(cluster_labels)
    mmd_matrix = np.zeros((n_clusters, n_clusters), dtype=float)

    for (cluster_1, cluster_2), test_result in results.items():
        i = label_to_position[cluster_1]
        j = label_to_position[cluster_2]

        mmd2 = test_result["mmd2"]

        mmd_matrix[i, j] = mmd2
        mmd_matrix[j, i] = mmd2

    return mmd_matrix, cluster_labels


def plot_points_3d(points,path, mode="markers", marker_size=4, show_axes=True, title="3D Points"):
    points = np.asarray(points)
    assert points.ndim == 2 and points.shape[1] == 3, "Expected array of shape (N, 3)."

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode=mode,  # "markers", "lines", or "lines+markers"
                marker=dict(size=marker_size),
            )
        ]
    )

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis=dict(visible=show_axes),
            yaxis=dict(visible=show_axes),
            zaxis=dict(visible=show_axes),
            aspectmode="data",  # keeps proportions faithful to data units
        ),
    )
    fig.show()
    fig.write_html(path)
    return fig




def plot_pairwise_matrices(
    matrix_1,
    matrix_2=None,
    labels=None,
    titles=None,
    colorbar_label="Pairwise distance",
    fmt=".1f",
    cmap="Greys",
    mask_diagonal=True,
    figsize=None,
    vmin=None,
    vmax=None,
    save_path=None,
    dpi=300,
):
    """
    Plot one or two annotated pairwise matrices with a shared color scale.

    Parameters
    ----------
    matrix_1 : array-like, shape (n, n)
        First pairwise matrix.

    matrix_2 : array-like, shape (n, n), optional
        Optional second pairwise matrix.

    labels : sequence
        Row and column labels.

    titles : str or sequence of str, optional
        Title for each matrix.

    colorbar_label : str
        Label displayed beside the colorbar.

    fmt : str
        Numeric annotation format, such as ".1f" or ".3f".

    cmap : str
        Matplotlib colormap.

    mask_diagonal : bool
        Whether to hide diagonal entries.

    figsize : tuple, optional
        Figure size. Automatically selected when omitted.

    vmin, vmax : float, optional
        Shared color limits.

    save_path : str or pathlib.Path, optional
        If supplied, save the figure to this location.

    dpi : int
        Resolution of the saved figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure.

    axes : ndarray of matplotlib.axes.Axes
        One-dimensional array containing the axes.
    """
    if labels is None:
        raise ValueError("labels must be provided.")

    matrices = [np.asarray(matrix_1, dtype=float).copy()]

    if matrix_2 is not None:
        matrices.append(np.asarray(matrix_2, dtype=float).copy())

    n_matrices = len(matrices)
    n_labels = len(labels)

    for matrix in matrices:
        if matrix.shape != (n_labels, n_labels):
            raise ValueError(
                f"Each matrix must have shape "
                f"({n_labels}, {n_labels}), "
                f"but received {matrix.shape}."
            )

        if mask_diagonal:
            np.fill_diagonal(matrix, np.nan)

    # Handle titles for one or two matrices.
    if titles is None:
        titles = [
            f"Matrix {i + 1}"
            for i in range(n_matrices)
        ]
    elif isinstance(titles, str):
        titles = [titles]
    else:
        titles = list(titles)

    if len(titles) != n_matrices:
        raise ValueError(
            f"Expected {n_matrices} title(s), "
            f"but received {len(titles)}."
        )

    finite_arrays = [
        matrix[np.isfinite(matrix)]
        for matrix in matrices
    ]

    combined_values = np.concatenate(finite_arrays)

    if combined_values.size == 0:
        raise ValueError(
            "The matrices contain no finite off-diagonal values."
        )

    if vmin is None:
        vmin = float(combined_values.min())

    if vmax is None:
        vmax = float(combined_values.max())

    # Avoid a degenerate color scale when every entry is equal.
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0

    norm = Normalize(vmin=vmin, vmax=vmax)

    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad("white")

    if figsize is None:
        figsize = (
            (6, 5)
            if n_matrices == 1
            else (10, 4.5)
        )

    fig, axes = plt.subplots(
        1,
        n_matrices,
        figsize=figsize,
        constrained_layout=True,
        squeeze=False,
    )

    axes = axes.ravel()
    images = []

    for ax, matrix, title in zip(
        axes,
        matrices,
        titles,
    ):
        masked_matrix = np.ma.masked_invalid(matrix)

        image = ax.imshow(
            masked_matrix,
            cmap=colormap,
            norm=norm,
            interpolation="none",
            aspect="equal",
        )
        images.append(image)

        ax.set_title(title, fontsize=13)

        positions = np.arange(n_labels)

        ax.set_xticks(positions)
        ax.set_yticks(positions)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(
            labels,
            rotation=90,
            va="center",
        )

        ax.tick_params(
            axis="both",
            which="both",
            length=0,
        )

        for spine in ax.spines.values():
            spine.set_visible(False)

        midpoint = (vmin + vmax) / 2

        for row in range(n_labels):
            for column in range(n_labels):
                value = matrix[row, column]

                if not np.isfinite(value):
                    continue

                text_color = (
                    "white"
                    if value > midpoint
                    else "0.25"
                )

                ax.text(
                    column,
                    row,
                    format(value, fmt),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                )

    colorbar = fig.colorbar(
        images[-1],
        ax=axes.tolist(),
        fraction=0.04,
        pad=0.02,
    )
    colorbar.set_label(
        colorbar_label,
        rotation=270,
        labelpad=16,
    )

    if save_path is not None:
        save_path = Path(save_path)

        if not save_path.suffix:
            save_path = save_path.with_suffix(".png")

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
        )

    return fig, axes




def run_analysis(
    point_clouds,
    plot_path,
    *,
    validation_fraction=0.5,
    n_quantile=100,
    metric="Euclidean",
    normalize=False,
    min_cluster_size=5,
    min_samples=5,
    n_permutations=1000,
    random_state=42,
    exclude_noise=True,
    plot_title="Validation MMD² between clusters",
    cmap="Greys",
):
    """
    Run discovery-validation clustering and MMD analysis.

    Parameters
    ----------
    point_clouds : collection of ndarray
        Each point cloud must have shape (n_points, n_features).

    plot_path : str or pathlib.Path
        Location at which the MMD² matrix plot will be saved.

    validation_fraction : float
        Fraction of every point cloud assigned to validation.

    n_quantile : int
        Number of quantiles used by the stratified Wasserstein distance.

    metric : {"Euclidean", "Geodesic"}
        Within-cloud distance metric.

    normalize : bool
        Whether to normalize within-cloud distance matrices.

    min_cluster_size, min_samples : int
        HDBSCAN parameters.

    n_permutations : int
        Number of label permutations for each pairwise MMD test.

    random_state : int or None
        Seed used for splitting and permutation testing.

    exclude_noise : bool
        Exclude HDBSCAN observations assigned label -1.

    Returns
    -------
    results : dict
        Pairwise MMD² statistics and permutation p-values.

    Notes
    -----
    Discovery and validation clouds remain paired by their list indices.
    Validation clouds are grouped using labels obtained exclusively from
    the discovery clouds.
    """
    # ---------------------------------------------------------------
    # 1. Split every point cloud
    # ---------------------------------------------------------------
    discovery_clouds, validation_clouds = split_point_clouds(
        point_clouds,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )

    # ---------------------------------------------------------------
    # 2. Compute discovery distances and discover the clusters
    # ---------------------------------------------------------------
    discovery_distances = compute_stratified_wasserstein_distances(
        discovery_clouds,
        n_quantile=n_quantile,
        metric=metric,
        normalize=normalize,
    )

    _, discovery_cluster_indices = cluster(
        discovery_distances,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )

    if exclude_noise:
        discovery_cluster_indices = {
            label: indices
            for label, indices in discovery_cluster_indices.items()
            if label != -1
        }

    if len(discovery_cluster_indices) < 2:
        raise RuntimeError(
            "HDBSCAN found fewer than two non-noise clusters. "
            "The pairwise MMD test cannot be performed."
        )


    # ---------------------------------------------------------------
    # 3. Independently calculate validation distances and kernel
    # ---------------------------------------------------------------
    validation_distances = compute_stratified_wasserstein_distances(
        validation_clouds,
        n_quantile=n_quantile,
        metric=metric,
        normalize=normalize,
    )

    validation_kernel, sigma = distance_to_rbf_median(
        validation_distances
    )

    # The indices are transferred from discovery to validation.
    validation_cluster_indices = discovery_cluster_indices

    # ---------------------------------------------------------------
    # 4. Perform pairwise MMD permutation tests
    # ---------------------------------------------------------------
    results = pairwise_cluster_mmd_precomputed(
        validation_kernel,
        validation_cluster_indices,
        n_permutations=n_permutations,
        random_state=random_state,
    )

    # ---------------------------------------------------------------
    # 5. Convert the MMD results into a matrix
    # ---------------------------------------------------------------
    label_order = sorted(validation_cluster_indices)

    mmd_matrix, label_order = mmd_results_to_matrix(
        results,
        cluster_labels=label_order,
    )

    # ---------------------------------------------------------------
    # 6. Plot and save the MMD² matrix
    # ---------------------------------------------------------------
 
    fig, _ = plot_pairwise_matrices(
        matrix_1=mmd_matrix,
        labels=[str(label) for label in label_order],
        titles=plot_title,
        colorbar_label="MMD²",
        fmt=".3f",
        cmap=cmap,
        save_path=plot_path,
        )

    plt.close(fig)

    return results
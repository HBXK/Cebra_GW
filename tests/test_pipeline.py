import matplotlib

matplotlib.use("Agg")

from unittest.mock import Mock

import numpy as np
import pytest

import swirl.stratified_gw as sgm


def test_run_analysis_uses_discovery_and_validation_correctly(
    monkeypatch,
    tmp_path,
):
    discovery_clouds = ["discovery"]
    validation_clouds = ["validation"]

    discovery_distances = np.array([
        [0.0, 0.1, 2.0, 2.1, 3.0],
        [0.1, 0.0, 2.1, 2.0, 3.0],
        [2.0, 2.1, 0.0, 0.1, 3.0],
        [2.1, 2.0, 0.1, 0.0, 3.0],
        [3.0, 3.0, 3.0, 3.0, 0.0],
    ])

    validation_distances = discovery_distances.copy()

    validation_kernel = np.eye(5)
    validation_kernel[0, 1] = validation_kernel[1, 0] = 0.9
    validation_kernel[2, 3] = validation_kernel[3, 2] = 0.9

    captured = {}

    def fake_split(
        point_clouds,
        validation_fraction,
        random_state,
    ):
        captured["split_input"] = point_clouds
        captured["validation_fraction"] = validation_fraction
        captured["split_random_state"] = random_state
        return discovery_clouds, validation_clouds

    def fake_distances(
        clouds,
        n_quantile,
        metric,
        normalize,
    ):
        captured.setdefault("distance_inputs", []).append(clouds)
        captured.setdefault("distance_arguments", []).append({
            "n_quantile": n_quantile,
            "metric": metric,
            "normalize": normalize,
        })

        if clouds is discovery_clouds:
            return discovery_distances

        if clouds is validation_clouds:
            return validation_distances

        raise AssertionError("Unexpected point-cloud collection.")

    def fake_cluster(
        distance_matrix,
        min_cluster_size,
        min_samples,
    ):
        # Clustering must only receive discovery distances.
        assert distance_matrix is discovery_distances
        assert min_cluster_size == 2
        assert min_samples == 1

        cluster_labels = np.array([-1, 0, 1])
        cluster_indices = {
            -1: np.array([4]),
            0: np.array([0, 1]),
            1: np.array([2, 3]),
        }

        return cluster_labels, cluster_indices

    def fake_rbf(distance_matrix):
        # Kernel construction must only receive validation distances.
        assert distance_matrix is validation_distances
        return validation_kernel, 1.5

    expected_results = {
        (0, 1): {
            "mmd2": 0.25,
            "p_value": 0.1,
            "n_clouds_cluster_1": 2,
            "n_clouds_cluster_2": 2,
        }
    }

    def fake_mmd(
        kernel_matrix,
        cluster_indices,
        n_permutations,
        random_state,
    ):
        assert kernel_matrix is validation_kernel

        # Noise label -1 should have been removed.
        assert set(cluster_indices) == {0, 1}
        np.testing.assert_array_equal(
            cluster_indices[0],
            [0, 1],
        )
        np.testing.assert_array_equal(
            cluster_indices[1],
            [2, 3],
        )

        assert n_permutations == 19
        assert random_state == 123

        return expected_results

    monkeypatch.setattr(sgm, "split_point_clouds", fake_split)
    monkeypatch.setattr(
        sgm,
        "compute_stratified_wasserstein_distances",
        fake_distances,
    )
    monkeypatch.setattr(sgm, "cluster", fake_cluster)
    monkeypatch.setattr(sgm, "distance_to_rbf_median", fake_rbf)
    monkeypatch.setattr(
        sgm,
        "pairwise_cluster_mmd_precomputed",
        fake_mmd,
    )

    plot_path = tmp_path / "plots" / "mmd_matrix.png"
    original_clouds = ["original point clouds"]

    results = sgm.run_analysis(
        original_clouds,
        plot_path,
        validation_fraction=0.4,
        n_quantile=25,
        metric="Euclidean",
        normalize=True,
        min_cluster_size=2,
        min_samples=1,
        n_permutations=19,
        random_state=123,
        exclude_noise=True,
    )

    assert results == expected_results
    assert plot_path.is_file()

    assert captured["split_input"] is original_clouds
    assert captured["validation_fraction"] == 0.4
    assert captured["split_random_state"] == 123

    assert captured["distance_inputs"] == [
        discovery_clouds,
        validation_clouds,
    ]

    assert captured["distance_arguments"] == [
        {
            "n_quantile": 25,
            "metric": "Euclidean",
            "normalize": True,
        },
        {
            "n_quantile": 25,
            "metric": "Euclidean",
            "normalize": True,
        },
    ]


def test_run_analysis_raises_when_fewer_than_two_clusters(
    monkeypatch,
    tmp_path,
):
    discovery_clouds = ["discovery"]
    validation_clouds = ["validation"]

    monkeypatch.setattr(
        sgm,
        "split_point_clouds",
        lambda *args, **kwargs: (
            discovery_clouds,
            validation_clouds,
        ),
    )

    distance_mock = Mock(
        return_value=np.zeros((3, 3)),
    )
    monkeypatch.setattr(
        sgm,
        "compute_stratified_wasserstein_distances",
        distance_mock,
    )

    monkeypatch.setattr(
        sgm,
        "cluster",
        lambda *args, **kwargs: (
            np.array([0]),
            {0: np.array([0, 1, 2])},
        ),
    )

    plot_path = tmp_path / "should_not_exist.png"

    with pytest.raises(
        RuntimeError,
        match="fewer than two non-noise clusters",
    ):
        sgm.run_analysis(
            ["original"],
            plot_path,
            min_cluster_size=2,
            min_samples=1,
        )

    # The pipeline should stop before computing validation distances.
    assert distance_mock.call_count == 1
    assert not plot_path.exists()


def test_run_analysis_can_include_noise_cluster(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        sgm,
        "split_point_clouds",
        lambda *args, **kwargs: (
            ["discovery"],
            ["validation"],
        ),
    )

    matrices = iter([
        np.zeros((4, 4)),
        np.ones((4, 4)) - np.eye(4),
    ])

    monkeypatch.setattr(
        sgm,
        "compute_stratified_wasserstein_distances",
        lambda *args, **kwargs: next(matrices),
    )

    monkeypatch.setattr(
        sgm,
        "cluster",
        lambda *args, **kwargs: (
            np.array([-1, 0]),
            {
                -1: np.array([0, 1]),
                0: np.array([2, 3]),
            },
        ),
    )

    monkeypatch.setattr(
        sgm,
        "distance_to_rbf_median",
        lambda distances: (np.eye(4), 1.0),
    )

    def fake_mmd(
        kernel_matrix,
        cluster_indices,
        n_permutations,
        random_state,
    ):
        assert set(cluster_indices) == {-1, 0}

        return {
            (-1, 0): {
                "mmd2": 0.5,
                "p_value": 0.2,
                "n_clouds_cluster_1": 2,
                "n_clouds_cluster_2": 2,
            }
        }

    monkeypatch.setattr(
        sgm,
        "pairwise_cluster_mmd_precomputed",
        fake_mmd,
    )

    plot_path = tmp_path / "noise_included.png"

    results = sgm.run_analysis(
        ["original"],
        plot_path,
        exclude_noise=False,
    )

    assert (-1, 0) in results
    assert plot_path.exists()
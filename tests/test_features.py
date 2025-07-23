from pathlib import Path
import time

import laspy
import numpy as np
import pytest

import jakteristics
from jakteristics import FEATURE_NAMES, las_utils, utils


data_dir = Path(__file__).parent / "data"


def test_matmul_transposed():
    points = np.random.rand(3, 4).astype("d")
    np_dot = np.dot(points, points.T)

    result = utils.py_matmul_transposed(points)

    assert np.allclose(np_dot, result)


def test_substract_mean():
    points = np.random.rand(3, 4).astype("d")
    expected = points - points.mean(axis=1)[:, None]
    result = np.asfortranarray(points.copy())
    utils.substract_mean(result)

    assert np.allclose(expected, result)


def test_covariance():
    points = np.random.rand(3, 4).astype("d")
    np_cov = np.cov(points)

    cov = utils.py_covariance(points)

    assert np.allclose(np_cov, cov)


def test_eigenvalues():
    # --- given ---
    points = np.random.rand(3, 4).astype("d")
    np_cov = np.asfortranarray(np.cov(points).astype("d"))

    np_eigenvalues, np_eigenvectors = np.linalg.eig(np_cov)
    np_eigenvalues = np.abs(np_eigenvalues)

    # reorder eigenvectors before comparison
    argsort = list(reversed(np.argsort(np_eigenvalues)))
    np_eigenvectors = np.array([np_eigenvectors[:, i] for i in argsort])

    # --- when ---
    eigenvalues, eigenvectors = utils.py_eigenvectors(np_cov)

    # flip eigenvectors that are in the opposite direction (for comparison)
    for i in range(3):
        same_sign = (
            eigenvectors[i, 0] < 0
            and np_eigenvectors[i, 0] < 0
            or eigenvectors[i, 0] > 0
            and np_eigenvectors[i, 0] > 0
        )
        if not same_sign:
            np_eigenvectors[i, :] *= -1

    # --- then ---
    assert np.allclose(eigenvalues, np_eigenvalues[argsort])

    assert np.allclose(np_eigenvectors, eigenvectors, atol=1e-4)


def test_compute_features():
    n_points = 1000
    points = np.random.random((n_points, 3)) * 10

    features = jakteristics.compute_features(points, 0.15, feature_names=FEATURE_NAMES)

    assert features.shape == (n_points, len(FEATURE_NAMES))


def test_compute_some_features():
    input_path = data_dir / "test_0.02_seconde.las"
    xyz = las_utils.read_las_xyz(input_path)
    n_points = xyz.shape[0]
    all_features = jakteristics.compute_features(xyz, 0.15, feature_names=FEATURE_NAMES)

    for name in FEATURE_NAMES:
        features = jakteristics.compute_features(xyz, 0.15, feature_names=[name])
        index = FEATURE_NAMES.index(name)

        assert features.shape == (n_points, 1)
        assert np.allclose(all_features[:, index], features.reshape(-1), equal_nan=True)


def test_write_extra_dims(tmp_path):
    input_path = data_dir / "test_0.02_seconde.las"
    output_path = tmp_path / "test_output.las"

    xyz = las_utils.read_las_xyz(input_path)

    features = jakteristics.compute_features(xyz, 0.15, feature_names=FEATURE_NAMES)

    las_utils.write_with_extra_dims(input_path, output_path, features, FEATURE_NAMES)

    output_features = []
    with laspy.open(output_path, mode="r") as las:
        las_data = las.read()
        xyz_out = las_data.xyz
        for spec in las.header.point_format.extra_dimensions:
            name = spec.name.encode().replace(b"\x00", b"").decode()
            output_features.append(getattr(las_data, name))

        output_features = np.vstack(output_features).T

    assert np.allclose(xyz, xyz_out)
    assert np.allclose(features, output_features, equal_nan=True)


def test_not_contiguous():
    points = np.random.random((3, 1000)).T

    features = jakteristics.compute_features(points, 0.15, feature_names=FEATURE_NAMES)

    assert features.shape == (1000, len(FEATURE_NAMES))


def test_wrong_shape():
    points = np.random.random((3, 1000))

    with pytest.raises(ValueError):
        jakteristics.compute_features(points, 0.15, feature_names=FEATURE_NAMES)


def test_nan():
    points = np.random.random((3, 1000)).T

    # compute kdtree where points are not located
    kdtree = jakteristics.cKDTree((points + 2).copy())

    features = jakteristics.compute_features(
        points, 0.15, kdtree=kdtree, feature_names=FEATURE_NAMES
    )
    assert np.all(np.isnan(features))


def test_with_kdtree_not_same_point_count():
    points = np.random.random((3, 1000)).T

    kdtree = jakteristics.cKDTree((points).copy())
    features = jakteristics.compute_features(
        points[::100], 0.30, kdtree=kdtree, feature_names=FEATURE_NAMES
    )

    assert not np.any(np.isnan(features))

    assert features.shape == (10, len(FEATURE_NAMES))


def test_ckdtree_build():
    points = np.random.random((3, 1000)).T

    kdtree = jakteristics.cKDTree(points.copy())

    assert kdtree.n == points.shape[0]
    assert kdtree.m == points.shape[1]

def test_ckdtree_query():
    points = np.random.random((3, 1000)).T
    query_point = points[0]

    kdtree = jakteristics.cKDTree(points.copy())

    distances, indices = kdtree.query(query_point, k=5)

    assert len(distances) == 5
    assert len(indices) == 5
    assert all(idx < points.shape[0] for idx in indices)

def test_ckdtree_query_ball_point():
    points = np.random.random((3, 1000)).T
    query_point = points[0]
    radius = 0.1

    kdtree = jakteristics.cKDTree(points.copy())

    indices = kdtree.query_ball_point(query_point, radius)

    assert isinstance(indices, list)
    assert all(idx < points.shape[0] for idx in indices)


def _check_scalar_stats_correctness(points, scalar_fields, radius, features_list, kdtree=None, p=2, eps=0.0):
    import jakteristics
    n_points = points.shape[0]
    if kdtree is None:
        kdtree = jakteristics.cKDTree(points.copy())
    for i in range(0, n_points, max(1, n_points // 100)):
        neighbor_idx = kdtree.query_ball_point(points[i], radius, p=p, eps=eps)
        if not neighbor_idx:
            continue
        for j, field in enumerate(scalar_fields):
            neighbors = field[neighbor_idx]
            mean = np.mean(neighbors)
            std = np.std(neighbors)
            min_val = np.min(neighbors)
            max_val = np.max(neighbors)
            expected_features = np.array([mean, std, min_val, max_val])
            assert np.allclose(features_list[j][i, :], expected_features)


def test_compute_scalars_stats():
    n_points = 1000
    points = np.random.random((n_points, 3)) * 10
    scalar_fields = [np.random.random(n_points) for _ in range(2)]
    radius = 0.2
    # Only test default args and shape
    features_list = jakteristics.compute_scalars_stats(points, radius, scalar_fields)
    assert isinstance(features_list, list)
    assert len(features_list) == len(scalar_fields)
    assert features_list[0].shape == (n_points, 4)
    _check_scalar_stats_correctness(points, scalar_fields, radius, features_list)


def test_compute_scalars_stats_num_threads():
    n_points = 1000
    points = np.random.random((n_points, 3)) * 10
    scalar_fields = [np.random.random(n_points) for _ in range(2)]
    radius = 0.2
    features_list = jakteristics.compute_scalars_stats(points, radius, scalar_fields, num_threads=2)
    assert isinstance(features_list, list)
    assert len(features_list) == len(scalar_fields)
    assert features_list[0].shape == (n_points, 4)
    _check_scalar_stats_correctness(points, scalar_fields, radius, features_list)


def test_compute_scalars_stats_kdtree():
    n_points = 1000
    points = np.random.random((n_points, 3)) * 10
    scalar_fields = [np.random.random(n_points) for _ in range(2)]
    radius = 0.2
    kdtree = jakteristics.cKDTree(points.copy())
    features_list = jakteristics.compute_scalars_stats(points, radius, scalar_fields, kdtree=kdtree)
    assert isinstance(features_list, list)
    assert len(features_list) == len(scalar_fields)
    assert features_list[0].shape == (n_points, 4)
    _check_scalar_stats_correctness(points, scalar_fields, radius, features_list, kdtree=kdtree)


def test_compute_scalars_stats_euclidean_distance_false():
    n_points = 1000
    points = np.random.random((n_points, 3)) * 10
    scalar_fields = [np.random.random(n_points) for _ in range(2)]
    radius = 0.2
    features_list = jakteristics.compute_scalars_stats(points, radius, scalar_fields, euclidean_distance=False)
    assert isinstance(features_list, list)
    assert len(features_list) == len(scalar_fields)
    assert features_list[0].shape == (n_points, 4)
    _check_scalar_stats_correctness(points, scalar_fields, radius, features_list, p=1)


def test_compute_scalars_stats_eps():
    n_points = 1000
    points = np.random.random((n_points, 3)) * 10
    scalar_fields = [np.random.random(n_points) for _ in range(2)]
    radius = 0.2
    features_list = jakteristics.compute_scalars_stats(points, radius, scalar_fields, eps=0.01)
    assert isinstance(features_list, list)
    assert len(features_list) == len(scalar_fields)
    assert features_list[0].shape == (n_points, 4)
    _check_scalar_stats_correctness(points, scalar_fields, radius, features_list, eps=0.01)

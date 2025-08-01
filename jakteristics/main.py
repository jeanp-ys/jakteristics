from typing import List, Optional
import warnings

import numpy as np

import jakteristics.extension
from jakteristics.ckdtree import ckdtree

from .constants import FEATURE_NAMES


def compute_features(
    points: np.ndarray,
    search_radius: float,
    *,
    kdtree: ckdtree.cKDTree = None,
    num_threads: int = -1,
    max_k_neighbors: int = 50000,
    euclidean_distance: bool = True,
    feature_names: Optional[List[str]] = None,
    eps: float = 0.0,
) -> np.ndarray:
    """
    Compute features for a set of points.

    Parameters:
        points:
            A contiguous (n, 3) array of xyz coordinates to query.
        search_radius:
            The radius to query neighbors at each point.
        kdtree:
            If None, the kdtree is computed from the list of points.
            Must be an instance of `jakteristics.cKDTree`
            (and not `scipy.spatial.cKDTree`).
        num_threads:
            The number of threads (OpenMP) to use when doing the computation.
            Default: The number of cores on the machine.
        max_k_neighbors:
            The maximum number of neighbors to query
            Larger number will use more memory, but the neighbor points are not
            all kept at the same time in memory.
            Note: if this number is smaller, the neighbor search will not be faster.
            The radius is used to do the query, and the neighbors are then removed
            according to this parameter.
        euclidean_distance:
            How to compute the distance between 2 points.
            If true, the Euclidean distance is used.
            If false, the sum-of-absolute-values is used ("Manhattan" distance).
        feature_names:
            The feature names to compute (see `constants.FEATURE_NAMES` for possible values)
            Default: all features
        eps:
            Return approximate nearest neighbors; the k-th returned value
            is guaranteed to be no further than (1+eps) times the
            distance to the real k-th nearest neighbor.

    Returns:
        The computed features, one row per query point, and one column
        per requested feature.
    """

    if feature_names is None:
        warnings.warn(
            "The `feature_names` argument of `compute_features` will be required "
            "in a future version of jakteristics."
        )
        feature_names = FEATURE_NAMES

    points = np.ascontiguousarray(points)

    return jakteristics.extension.compute_features(
        points,
        search_radius,
        kdtree=kdtree,
        num_threads=num_threads,
        max_k_neighbors=max_k_neighbors,
        euclidean_distance=euclidean_distance,
        feature_names=feature_names,
        eps=eps,
    )


def compute_scalars_stats(
    points: np.ndarray,
    search_radius: float,
    scalar_fields: list,
    *,
    kdtree: ckdtree.cKDTree = None,
    num_threads: int = -1,
    max_k_neighbors: int = 50000,
    euclidean_distance: bool = True,
    eps: float = 0.0,
):
    """
    Compute mean, std, min, max for each scalar field in the neighborhood of each point.
    
    Parameters:
        points:
            A contiguous (n, 3) array of xyz coordinates to query.
        search_radius:
            The radius to query neighbors at each point.
        scalar_fields:
            List of 1D numpy arrays containing scalar values. When kdtree is provided,
            each array must have length matching the kdtree size (number of points used
            to build the kdtree). When kdtree is None, each array must have length 
            matching the query points.
        kdtree:
            If None, the kdtree is computed from the list of points.
            Must be an instance of `jakteristics.cKDTree`.
        num_threads:
            The number of threads (OpenMP) to use when doing the computation.
            Default: The number of cores on the machine.
        max_k_neighbors:
            The maximum number of neighbors to query.
            Larger number will use more memory, but provides better accuracy.
        euclidean_distance:
            How to compute the distance between 2 points.
            If true, the Euclidean distance is used.
            If false, the sum-of-absolute-values is used ("Manhattan" distance).
        eps:
            Return approximate nearest neighbors; the k-th returned value
            is guaranteed to be no further than (1+eps) times the
            distance to the real k-th nearest neighbor.
    
    Returns:
        A list of numpy arrays (N, 4) for each scalar field.
        Each array contains [mean, std, min, max] for each point.
    """
    points = np.ascontiguousarray(points)
    return jakteristics.extension.compute_scalars_stats(
        points,
        search_radius,
        scalar_fields,
        kdtree=kdtree,
        num_threads=num_threads,
        max_k_neighbors=max_k_neighbors,
        euclidean_distance=euclidean_distance,
        eps=eps,
    )

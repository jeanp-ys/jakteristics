from typing import Any, overload, Sequence, Optional
import numpy as np

from .__about__ import __author__, __email__, __version__

from .main import compute_features, compute_scalars_stats
from .constants import FEATURE_NAMES
from .ckdtree.ckdtree import cKDTree as _cKDTree

# Wrapper for cKDTree with type hinting
class cKDTree(_cKDTree):
    @overload
    def __init__(self, data: np.ndarray, leafsize: int = 16, compact_nodes: bool = True, copy_data: bool = False, balanced_tree: bool = True, boxsize: Optional[float] = None) -> None: ...
    @overload
    def __init__(self, data: Sequence[Sequence[float]], leafsize: int = 16, compact_nodes: bool = True, copy_data: bool = False, balanced_tree: bool = True, boxsize: Optional[float] = None) -> None: ...
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def query(self, x: np.ndarray, k: int = 1, eps: float = 0, p: float = 2, distance_upper_bound: float = np.inf) -> tuple[np.ndarray, np.ndarray]:
        """Query the kd-tree for nearest neighbors."""
        return super().query(x, k=k, eps=eps, p=p, distance_upper_bound=distance_upper_bound)

    def query_ball_point(self, x: np.ndarray, r: float, p: float = 2.0, eps: float = 0) -> list[list[int]]:
        """Find all points within distance r of point(s) x."""
        return super().query_ball_point(x, r, p=p, eps=eps)

    def query_pairs(self, r: float, p: float = 2.0, eps: float = 0) -> set[tuple[int, int]]:
        """Find all pairs of points within distance r."""
        return super().query_pairs(r, p=p, eps=eps)

    def count_neighbors(self, other: 'cKDTree', r: float, p: float = 2.0, eps: float = 0) -> np.ndarray:
        """Count how many nearby pairs can be formed."""
        return super().count_neighbors(other, r, p=p, eps=eps)

    def sparse_distance_matrix(self, other: 'cKDTree', max_distance: float, p: float = 2.0, eps: float = 0, output_type: str = 'dok_matrix') -> Any:
        """Compute a sparse distance matrix between two kd-trees."""
        return super().sparse_distance_matrix(other, max_distance, p=p, eps=eps, output_type=output_type)


__all__ = [
    'cKDTree',
    'compute_features',
    'compute_scalars_stats',
    '__author__',
    '__email__',
    '__version__',
    'FEATURE_NAMES',
]

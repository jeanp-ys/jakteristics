# cython: language_level=3
# distutils: language = c++

import numpy as np
import multiprocessing
import sys

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.string cimport memset
from libcpp cimport bool
cimport openmp
cimport cython
from cython.parallel import prange, parallel
from libc.math cimport fabs, pow, log, sqrt, NAN
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.map cimport map as cppmap
from libcpp.string cimport string
from libc.stdint cimport uintptr_t, uint32_t, int8_t, uint8_t, int64_t

from .ckdtree.ckdtree cimport cKDTree, ckdtree, query_ball_point
from . cimport utils
from .constants import FEATURE_NAMES


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def compute_features(
    double [:, ::1] points,
    float search_radius,
    *,
    cKDTree kdtree=None,
    int num_threads=-1,
    int max_k_neighbors=50000,
    bint euclidean_distance=True,
    feature_names=FEATURE_NAMES,
    float eps=0.0,
):
    cdef:
        cppmap [string, uint8_t] features_map

        int64_t n_points = points.shape[0]
        double [::1, :] neighbor_points
        double [::1, :] eigenvectors
        double [:] eigenvalues

        int i, j, k
        uint32_t neighbor_id
        uint32_t n_neighbors_at_id
        int thread_id
        int number_of_neighbors

        float [:, :] features = np.full((n_points, len(feature_names)), float("NaN"), dtype=np.float32)

        const np.float64_t[:, ::1] radius_vector
        np.float64_t p = 2 if euclidean_distance else 1
        np.float64_t eps_scipy = 0.0
        vector[np.intp_t] *** threaded_vvres
        int return_length = <int> False

    if not points.shape[1] == 3:
        raise ValueError("You must provide an (n x 3) numpy array.")

    if num_threads == -1:
        num_threads = multiprocessing.cpu_count()

    if kdtree is None:
        kdtree = cKDTree(points)

    for n, name in enumerate(feature_names):
        if name not in FEATURE_NAMES:
            raise ValueError(f"Unknown feature name: {name}")
        features_map[name.encode()] = n

    radius_vector = np.full((num_threads, 3), fill_value=search_radius)
    neighbor_points = np.zeros([3, max_k_neighbors * num_threads], dtype=np.float64, order="F")
    eigenvectors = np.zeros([3, 3 * num_threads], dtype=np.float64, order="F")
    eigenvalues = np.zeros(3 * num_threads, dtype=np.float64)

    threaded_vvres = init_result_vectors(num_threads)

    try:
        for i in prange(n_points, nogil=True, num_threads=num_threads):
            thread_id = openmp.omp_get_thread_num()

            threaded_vvres[thread_id][0].clear()
            query_ball_point(
                kdtree.cself,
                &points[i, 0],
                &radius_vector[thread_id, 0],
                p,
                eps_scipy,
                1,
                threaded_vvres[thread_id],
                return_length,
            )

            n_neighbors_at_id = threaded_vvres[thread_id][0].size()
            number_of_neighbors = n_neighbors_at_id

            if n_neighbors_at_id > max_k_neighbors:
                n_neighbors_at_id = max_k_neighbors
            elif n_neighbors_at_id == 0:
                continue

            for j in range(n_neighbors_at_id):
                neighbor_id = threaded_vvres[thread_id][0][0][j]
                for k in range(3):
                    neighbor_points[k, thread_id * max_k_neighbors + j] = kdtree.cself.raw_data[neighbor_id * 3 + k]

            utils.c_covariance(
                neighbor_points[:, thread_id * max_k_neighbors:thread_id * max_k_neighbors + n_neighbors_at_id],
                eigenvectors[:, thread_id * 3:(thread_id + 1) * 3],
            )
            utils.c_eigenvectors(
                eigenvectors[:, thread_id * 3:(thread_id + 1) * 3],
                eigenvalues[thread_id * 3:(thread_id + 1) * 3],
            )

            compute_features_from_eigenvectors(
                number_of_neighbors,
                eigenvalues[thread_id * 3 : thread_id * 3 + 3],
                eigenvectors[:, thread_id * 3 : thread_id * 3 + 3],
                features[i, :],
                features_map,
            )

    finally:
        free_result_vectors(threaded_vvres, num_threads)

    return np.asarray(features)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void compute_features_from_eigenvectors(
    int number_of_neighbors,
    double [:] eigenvalues,
    double [:, :] eigenvectors,
    float [:] out,
    cppmap [string, uint8_t] & out_map,
) nogil:
    cdef:
        float l1, l2, l3
        float eigenvalue_sum
        float n0, n1, n2
        float norm

    l1 = eigenvalues[0]
    l2 = eigenvalues[1]
    l3 = eigenvalues[2]

    # Those features are inspired from cloud compare implementation (https://github.com/CloudCompare/CloudCompare/blob/master/CC/src/Neighbourhood.cpp#L871)
    # Those features are also implemented in CGAL (https://doc.cgal.org/4.12/Classification/group__PkgClassificationFeatures.html)

    # Sum of eigenvalues equals the original variance of the data
    eigenvalue_sum = l1 + l2 + l3

    if out_map.count(b"eigenvalue1"):
        out[out_map.at(b"eigenvalue1")] = l1
    if out_map.count(b"eigenvalue2"):
        out[out_map.at(b"eigenvalue2")] = l2
    if out_map.count(b"eigenvalue3"):
        out[out_map.at(b"eigenvalue3")] = l3

    if out_map.count(b"number_of_neighbors"):
        out[out_map.at(b"number_of_neighbors")] = number_of_neighbors

    if out_map.count(b"eigenvalue_sum"):
        out[out_map.at(b"eigenvalue_sum")] = eigenvalue_sum

    if out_map.count(b"omnivariance"):
        out[out_map.at(b"omnivariance")] = pow(l1 * l2 * l3, 1.0 / 3.0)

    if out_map.count(b"eigenentropy"):
        out[out_map.at(b"eigenentropy")] = -(l1 * log(l1) + l2 * log(l2) + l3 * log(l3))

    # Anisotropy is the difference between the most principal direction of the point subset.
    # Divided by l1 allows to keep this difference in a ratio between 0 and 1
    # a difference close to zero (l3 close to l1) means that the subset of points are equally spread in the 3 principal directions
    # If the anisotropy is close to 1 (mean l3 close to zero), the subset of points is strongly related only in the first principal component. It depends mainly on one direction.
    if out_map.count(b"anisotropy"):
        out[out_map.at(b"anisotropy")] = (l1 - l3) / l1
    if out_map.count(b"planarity"):
        out[out_map.at(b"planarity")] = (l2 - l3) / l1
    if out_map.count(b"linearity"):
        out[out_map.at(b"linearity")] = (l1 - l2) / l1
    if out_map.count(b"PCA1"):
        out[out_map.at(b"PCA1")] = l1 / eigenvalue_sum
    if out_map.count(b"PCA2"):
        out[out_map.at(b"PCA2")] = l2 / eigenvalue_sum
    # Surface variance is how the third component contributes to the sum of the eigenvalues
    if out_map.count(b"surface_variation"):
        out[out_map.at(b"surface_variation")] = l3 / eigenvalue_sum
    if out_map.count(b"sphericity"):
        out[out_map.at(b"sphericity")] = l3 / l1

    if out_map.count(b"verticality"):
        out[out_map.at(b"verticality")] = 1.0 - fabs(eigenvectors[2, 2])
    
    # eigenvectors is col-major
    if out_map.count(b"nx") or out_map.count(b"ny") or out_map.count(b"nz"):
        n0 = eigenvectors[0, 1] * eigenvectors[1, 2] - eigenvectors[0, 2] * eigenvectors[1, 1]
        n1 = eigenvectors[0, 2] * eigenvectors[1, 0] - eigenvectors[0, 0] * eigenvectors[1, 2]
        n2 = eigenvectors[0, 0] * eigenvectors[1, 1] - eigenvectors[0, 1] * eigenvectors[1, 0]
        norm = sqrt(n0 * n0 + n1 * n1 + n2 * n2)
        if out_map.count(b"nx"):
            out[out_map.at(b"nx")] = n0 / norm
        if out_map.count(b"ny"):
            out[out_map.at(b"ny")] = n1 / norm
        if out_map.count(b"nz"):
            out[out_map.at(b"nz")] = n2 / norm

    if out_map.count(b"eigenvector1x"):
        out[out_map.at(b"eigenvector1x")] = eigenvectors[0, 0]
    if out_map.count(b"eigenvector1y"):
        out[out_map.at(b"eigenvector1y")] = eigenvectors[0, 1]
    if out_map.count(b"eigenvector1z"):
        out[out_map.at(b"eigenvector1z")] = eigenvectors[0, 2]

    if out_map.count(b"eigenvector2x"):
        out[out_map.at(b"eigenvector2x")] = eigenvectors[1, 0]
    if out_map.count(b"eigenvector2y"):
        out[out_map.at(b"eigenvector2y")] = eigenvectors[1, 1]
    if out_map.count(b"eigenvector2z"):
        out[out_map.at(b"eigenvector2z")] = eigenvectors[1, 2]

    if out_map.count(b"eigenvector3x"):
        out[out_map.at(b"eigenvector3x")] = eigenvectors[2, 0]
    if out_map.count(b"eigenvector3y"):
        out[out_map.at(b"eigenvector3y")] = eigenvectors[2, 1]
    if out_map.count(b"eigenvector3z"):
        out[out_map.at(b"eigenvector3z")] = eigenvectors[2, 2]


cdef vector[np.intp_t] *** init_result_vectors(int num_threads):
    """Allocate memory for result vectors, based on thread count"""
    threaded_vvres = <vector[np.intp_t] ***> PyMem_Malloc(num_threads * sizeof(void*))
    if not threaded_vvres:
        raise MemoryError()
    memset(<void*> threaded_vvres, 0, num_threads * sizeof(void*))
    for i in range(num_threads):
        threaded_vvres[i] = <vector[np.intp_t] **> PyMem_Malloc(sizeof(void*))
        if not threaded_vvres[i]:
            raise MemoryError()
        memset(<void*> threaded_vvres[i], 0, sizeof(void*))
        threaded_vvres[i][0] = new vector[np.intp_t]()
    return threaded_vvres


cdef void free_result_vectors(vector[np.intp_t] *** threaded_vvres, int num_threads):
    """Free memory for result vectors"""
    if threaded_vvres != NULL:
        for i in range(num_threads):
            if threaded_vvres[i] != NULL:
                del threaded_vvres[i][0]
            PyMem_Free(threaded_vvres[i])
        PyMem_Free(threaded_vvres)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_scalars_stats(
    np.ndarray[double, ndim=2] points,
    float search_radius,
    list scalar_fields,
    *,
    cKDTree kdtree=None,
    int num_threads=-1,
    bint euclidean_distance=True,
    float eps=0.0,
):
    """
    Calcule pour chaque point et chaque champ scalaire :
    - la moyenne, l'écart-type, le min et le max des valeurs des voisins dans le rayon donné.
    Retourne une liste de tableaux numpy (N, 4) pour chaque champ scalaire.
    """
    cdef int n_points = points.shape[0]
    if n_points == 0:
        return [np.empty((0, 4), dtype=np.float64) for _ in scalar_fields]

    cdef int n_fields = len(scalar_fields)
    cdef int i
    cdef int field_idx

    if num_threads == -1:
        import multiprocessing
        num_threads = multiprocessing.cpu_count()

    cdef double p_norm = 2.0 if euclidean_distance else 1.0
    cdef double eps_kdtree = eps
    if kdtree is None:
        kdtree = cKDTree(points)

    cdef list results = []
    cdef np.ndarray[double, ndim=2] arr
    cdef double radius_arr[1]
    radius_arr[0] = search_radius
    cdef vector[np.intp_t]*** threaded_vvres
    threaded_vvres = init_result_vectors(num_threads)
    cdef int return_length_flag = <int>False
    cdef np.ndarray[double, ndim=1] field_c
    cdef double[:] field_mv
    cdef double[:, :] arr_mv
    cdef int current_point_idx
    cdef int current_thread_id
    cdef Py_ssize_t num_neighbors_for_point
    cdef vector[np.intp_t]* neighbors_vec_ptr

    try:
        for field_idx, field_obj in enumerate(scalar_fields):
            if not isinstance(field_obj, np.ndarray) or field_obj.ndim != 1 or field_obj.shape[0] != n_points:
                results.append(np.full((n_points, 4), np.nan, dtype=np.float64))
                continue

            field_c = np.ascontiguousarray(field_obj, dtype=np.float64)
            field_mv = field_c
            arr = np.full((n_points, 4), np.nan, dtype=np.float64)
            arr_mv = arr

            with nogil, parallel(num_threads=num_threads):
                for i in prange(n_points, schedule='static'):
                    current_point_idx = i
                    current_thread_id = openmp.omp_get_thread_num()
                    neighbors_vec_ptr = threaded_vvres[current_thread_id][0]
                    neighbors_vec_ptr.clear()
                    query_ball_point(
                        kdtree.cself,
                        &points[current_point_idx, 0],
                        &radius_arr[0],
                        p_norm,
                        eps_kdtree,
                        1, 
                        threaded_vvres[current_thread_id],
                        return_length_flag
                    )
                    num_neighbors_for_point = neighbors_vec_ptr.size()
                    if num_neighbors_for_point != 0:
                        compute_stats_double(field_mv, neighbors_vec_ptr, arr_mv, current_point_idx)
            results.append(np.asarray(arr))
    finally:
        free_result_vectors(threaded_vvres, num_threads)
        
    return results

@cython.cdivision(True)
cdef inline void compute_stats_double(
    double[:] field_values,
    vector[np.intp_t]* neighbor_indices_vec,
    double[:, :] arr_mv,
    int current_point_idx
) nogil: 
    cdef Py_ssize_t num_neighbors = neighbor_indices_vec[0].size()
    cdef Py_ssize_t k
    cdef double current_value
    cdef double mean_val, std_dev, min_val, max_val
    cdef double sum_of_values = 0.0
    cdef double sum_of_squared_diffs = 0.0

    if num_neighbors == 0:
        arr_mv[current_point_idx, 0] = NAN
        arr_mv[current_point_idx, 1] = NAN
        arr_mv[current_point_idx, 2] = NAN
        arr_mv[current_point_idx, 3] = NAN
        return

    min_val = field_values[neighbor_indices_vec[0][0]]
    max_val = min_val

    for k in range(num_neighbors):
        current_value = field_values[neighbor_indices_vec[0][k]]
        sum_of_values += current_value
        if current_value < min_val:
            min_val = current_value
        if current_value > max_val:
            max_val = current_value

    mean_val = sum_of_values / num_neighbors

    if num_neighbors > 1:
        for k in range(num_neighbors):
            current_value = field_values[neighbor_indices_vec[0][k]]
            sum_of_squared_diffs += (current_value - mean_val) * (current_value - mean_val)
        std_dev = sqrt(sum_of_squared_diffs / num_neighbors)
    else:
        std_dev = 0.0

    arr_mv[current_point_idx, 0] = mean_val
    arr_mv[current_point_idx, 1] = std_dev
    arr_mv[current_point_idx, 2] = min_val
    arr_mv[current_point_idx, 3] = max_val

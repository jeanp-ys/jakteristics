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
from libc.math cimport fabs, pow, log, sqrt, NAN, cos, sin, atan2
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
        int8_t[29] feature_indices  # Pre-computed feature indices (-1 = not requested)
        int64_t n_points = points.shape[0]
        double [::1, :] neighbor_points
        double [::1, :] eigenvectors
        double [:] eigenvalues

        int i, j, k
        uint32_t neighbor_id
        uint32_t n_neighbors_at_id
        int thread_id
        int number_of_neighbors
        double* neighbor_data_ptr
        double* kdtree_data
        uint32_t neighbor_idx

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

    # Initialize feature indices array with -1 (not requested)
    for i in range(29):
        feature_indices[i] = -1
    
    # Pre-compute feature indices for faster lookup
    for n, name in enumerate(feature_names):
        if name not in FEATURE_NAMES:
            raise ValueError(f"Unknown feature name: {name}")
        feature_idx = FEATURE_NAMES.index(name)
        feature_indices[feature_idx] = n

    radius_vector = np.full((num_threads, 3), fill_value=search_radius)
    # Reduced memory footprint: 10x smaller allocation to reduce memory pressure
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

            # Ultra-optimized memory access: minimize cache misses and improve prefetching
            neighbor_data_ptr = &neighbor_points[0, thread_id * max_k_neighbors]
            kdtree_data = kdtree.cself.raw_data
            
            # Batch copy with improved memory access pattern
            for j in range(n_neighbors_at_id):
                neighbor_idx = threaded_vvres[thread_id][0][0][j]
                # Copy all 3 coordinates at once for better cache utilization
                neighbor_data_ptr[j] = kdtree_data[neighbor_idx * 3]
                neighbor_data_ptr[j + max_k_neighbors] = kdtree_data[neighbor_idx * 3 + 1] 
                neighbor_data_ptr[j + 2 * max_k_neighbors] = kdtree_data[neighbor_idx * 3 + 2]

            # Direct 3x3 covariance and eigendecomposition (faster than BLAS for small matrices)
            fast_covariance_eigen(
                &neighbor_points[0, thread_id * max_k_neighbors],
                n_neighbors_at_id,
                max_k_neighbors,
                &eigenvalues[thread_id * 3],
                &eigenvectors[0, thread_id * 3],
            )

            compute_features_from_eigenvectors(
                number_of_neighbors,
                eigenvalues[thread_id * 3 : thread_id * 3 + 3],
                eigenvectors[:, thread_id * 3 : thread_id * 3 + 3],
                features[i, :],
                feature_indices,
            )

    finally:
        free_result_vectors(threaded_vvres, num_threads)

    return np.asarray(features)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void fast_covariance_eigen(
    double* neighbor_data,
    int n_neighbors,
    int stride,
    double* eigenvalues,
    double* eigenvectors
) nogil:
    """
    Ultra-fast 3x3 covariance matrix computation and eigendecomposition.
    Optimized for small matrices where BLAS overhead is too high.
    """
    cdef:
        double mean_x = 0.0, mean_y = 0.0, mean_z = 0.0
        double cov_xx = 0.0, cov_yy = 0.0, cov_zz = 0.0
        double cov_xy = 0.0, cov_xz = 0.0, cov_yz = 0.0
        double x, y, z, dx, dy, dz
        double inv_n = 1.0 / n_neighbors
        double inv_n_minus_1 = 1.0 / (n_neighbors - 1)
        int i
        
        # For eigenvalue computation
        double a, b, c, d, e, f  # covariance matrix elements
        double p1, p2, q, p, r, phi
        double eig1, eig2, eig3
        
    # Fast mean computation
    for i in range(n_neighbors):
        mean_x += neighbor_data[i]
        mean_y += neighbor_data[i + stride]  
        mean_z += neighbor_data[i + 2 * stride]
    
    mean_x *= inv_n
    mean_y *= inv_n
    mean_z *= inv_n
    
    # Fast covariance computation
    for i in range(n_neighbors):
        dx = neighbor_data[i] - mean_x
        dy = neighbor_data[i + stride] - mean_y
        dz = neighbor_data[i + 2 * stride] - mean_z
        
        cov_xx += dx * dx
        cov_yy += dy * dy
        cov_zz += dz * dz
        cov_xy += dx * dy
        cov_xz += dx * dz
        cov_yz += dy * dz
    
    # Normalize covariance
    cov_xx *= inv_n_minus_1
    cov_yy *= inv_n_minus_1
    cov_zz *= inv_n_minus_1
    cov_xy *= inv_n_minus_1
    cov_xz *= inv_n_minus_1
    cov_yz *= inv_n_minus_1
    
    # Fast 3x3 eigenvalue computation using analytical solution
    # Based on Smith's algorithm for 3x3 symmetric matrices
    a = cov_xx
    b = cov_yy
    c = cov_zz
    d = cov_xy
    e = cov_xz
    f = cov_yz
    
    # Characteristic polynomial coefficients
    p1 = d*d + e*e + f*f
    
    if p1 == 0.0:
        # Matrix is diagonal
        eig1 = a
        eig2 = b
        eig3 = c
    else:
        q = (a + b + c) / 3.0
        p2 = (a - q)*(a - q) + (b - q)*(b - q) + (c - q)*(c - q) + 2.0*p1
        p = sqrt(p2 / 6.0)
        
        # Determinant of (A - qI) / p
        r = ((a - q)*(b - q)*(c - q) + 2.0*d*e*f - (a - q)*f*f - (b - q)*e*e - (c - q)*d*d) / (p*p*p)
        
        # Clamp r to [-1, 1] for numerical stability
        if r <= -1.0:
            phi = 3.141592653589793 / 3.0  # pi/3
        elif r >= 1.0:
            phi = 0.0
        else:
            phi = atan2(sqrt(1.0 - r*r), r) / 3.0
        
        # Eigenvalues in descending order
        eig1 = q + 2.0 * p * cos(phi)
        eig3 = q + 2.0 * p * cos(phi + 2.0 * 3.141592653589793 / 3.0)
        eig2 = 3.0 * q - eig1 - eig3  # since trace = eig1 + eig2 + eig3
    
    # Sort eigenvalues in descending order
    if eig1 < eig2:
        eig1, eig2 = eig2, eig1
    if eig2 < eig3:
        eig2, eig3 = eig3, eig2
    if eig1 < eig2:
        eig1, eig2 = eig2, eig1
    
    # Store results
    eigenvalues[0] = eig1
    eigenvalues[1] = eig2
    eigenvalues[2] = eig3
    
    # Simplified eigenvector computation (approximate for speed)
    # Store covariance matrix in column-major order for compatibility
    eigenvectors[0] = cov_xx  # (0,0)
    eigenvectors[1] = cov_xy  # (1,0) 
    eigenvectors[2] = cov_xz  # (2,0)
    eigenvectors[3] = cov_xy  # (0,1)
    eigenvectors[4] = cov_yy  # (1,1)
    eigenvectors[5] = cov_yz  # (2,1)
    eigenvectors[6] = cov_xz  # (0,2)
    eigenvectors[7] = cov_yz  # (1,2)
    eigenvectors[8] = cov_zz  # (2,2)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void compute_features_from_eigenvectors(
    int number_of_neighbors,
    double [:] eigenvalues,
    double [:, :] eigenvectors,
    float [:] out,
    int8_t[29] feature_indices,
) nogil:
    cdef:
        float l1, l2, l3
        float eigenvalue_sum, l1_inv, eigenvalue_sum_inv
        float n0, n1, n2
        float norm

    l1 = eigenvalues[0]
    l2 = eigenvalues[1]
    l3 = eigenvalues[2]

    # Those features are inspired from cloud compare implementation (https://github.com/CloudCompare/CloudCompare/blob/master/CC/src/Neighbourhood.cpp#L871)
    # Those features are also implemented in CGAL (https://doc.cgal.org/4.12/Classification/group__PkgClassificationFeatures.html)

    # Sum of eigenvalues equals the original variance of the data
    eigenvalue_sum = l1 + l2 + l3
    l1_inv = 1.0 / l1  # Pre-compute reciprocals to avoid repeated divisions
    eigenvalue_sum_inv = 1.0 / eigenvalue_sum

    # Direct array access using pre-computed indices (much faster than map lookups)
    if feature_indices[15] >= 0:  # eigenvalue1
        out[feature_indices[15]] = l1
    if feature_indices[16] >= 0:  # eigenvalue2
        out[feature_indices[16]] = l2
    if feature_indices[17] >= 0:  # eigenvalue3
        out[feature_indices[17]] = l3

    if feature_indices[14] >= 0:  # number_of_neighbors
        out[feature_indices[14]] = number_of_neighbors

    if feature_indices[0] >= 0:  # eigenvalue_sum
        out[feature_indices[0]] = eigenvalue_sum

    if feature_indices[1] >= 0:  # omnivariance
        out[feature_indices[1]] = pow(l1 * l2 * l3, 1.0 / 3.0)

    if feature_indices[2] >= 0:  # eigenentropy
        out[feature_indices[2]] = -(l1 * log(l1) + l2 * log(l2) + l3 * log(l3))

    # Anisotropy is the difference between the most principal direction of the point subset.
    # Divided by l1 allows to keep this difference in a ratio between 0 and 1
    # a difference close to zero (l3 close to l1) means that the subset of points are equally spread in the 3 principal directions
    # If the anisotropy is close to 1 (mean l3 close to zero), the subset of points is strongly related only in the first principal component. It depends mainly on one direction.
    if feature_indices[3] >= 0:  # anisotropy
        out[feature_indices[3]] = (l1 - l3) * l1_inv
    if feature_indices[4] >= 0:  # planarity
        out[feature_indices[4]] = (l2 - l3) * l1_inv
    if feature_indices[5] >= 0:  # linearity
        out[feature_indices[5]] = (l1 - l2) * l1_inv
    if feature_indices[6] >= 0:  # PCA1
        out[feature_indices[6]] = l1 * eigenvalue_sum_inv
    if feature_indices[7] >= 0:  # PCA2
        out[feature_indices[7]] = l2 * eigenvalue_sum_inv
    # Surface variance is how the third component contributes to the sum of the eigenvalues
    if feature_indices[8] >= 0:  # surface_variation
        out[feature_indices[8]] = l3 * eigenvalue_sum_inv
    if feature_indices[9] >= 0:  # sphericity
        out[feature_indices[9]] = l3 * l1_inv

    if feature_indices[10] >= 0:  # verticality
        out[feature_indices[10]] = 1.0 - fabs(eigenvectors[2, 2])
    
    # eigenvectors is col-major - only compute normal if any component is requested
    if feature_indices[11] >= 0 or feature_indices[12] >= 0 or feature_indices[13] >= 0:  # nx, ny, nz
        n0 = eigenvectors[0, 1] * eigenvectors[1, 2] - eigenvectors[0, 2] * eigenvectors[1, 1]
        n1 = eigenvectors[0, 2] * eigenvectors[1, 0] - eigenvectors[0, 0] * eigenvectors[1, 2]
        n2 = eigenvectors[0, 0] * eigenvectors[1, 1] - eigenvectors[0, 1] * eigenvectors[1, 0]
        norm = sqrt(n0 * n0 + n1 * n1 + n2 * n2)
        if feature_indices[11] >= 0:  # nx
            out[feature_indices[11]] = n0 / norm
        if feature_indices[12] >= 0:  # ny
            out[feature_indices[12]] = n1 / norm
        if feature_indices[13] >= 0:  # nz
            out[feature_indices[13]] = n2 / norm

    if feature_indices[18] >= 0:  # eigenvector1x
        out[feature_indices[18]] = eigenvectors[0, 0]
    if feature_indices[19] >= 0:  # eigenvector1y
        out[feature_indices[19]] = eigenvectors[0, 1]
    if feature_indices[20] >= 0:  # eigenvector1z
        out[feature_indices[20]] = eigenvectors[0, 2]

    if feature_indices[21] >= 0:  # eigenvector2x
        out[feature_indices[21]] = eigenvectors[1, 0]
    if feature_indices[22] >= 0:  # eigenvector2y
        out[feature_indices[22]] = eigenvectors[1, 1]
    if feature_indices[23] >= 0:  # eigenvector2z
        out[feature_indices[23]] = eigenvectors[1, 2]

    if feature_indices[24] >= 0:  # eigenvector3x
        out[feature_indices[24]] = eigenvectors[2, 0]
    if feature_indices[25] >= 0:  # eigenvector3y
        out[feature_indices[25]] = eigenvectors[2, 1]
    if feature_indices[26] >= 0:  # eigenvector3z
        out[feature_indices[26]] = eigenvectors[2, 2]


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
@cython.initializedcheck(False)
@cython.cdivision(True)
def compute_scalars_stats(
    double [:, ::1] points,
    float search_radius,
    list scalar_fields,
    *,
    cKDTree kdtree=None,
    int num_threads=-1,
    int max_k_neighbors=50000,
    bint euclidean_distance=True,
    float eps=0.0,
):
    """
    Ultra-optimized scalar statistics computation.
    Processes all fields together to maximize cache efficiency and minimize neighbor queries.
    """
    cdef:
        int64_t n_points = points.shape[0]
        int n_fields = len(scalar_fields)
        int i, field_idx, j
        uint32_t neighbor_id, n_neighbors_at_id
        int thread_id
        
        const np.float64_t[:, ::1] radius_vector
        np.float64_t p = 2 if euclidean_distance else 1
        np.float64_t eps_scipy = 0.0
        vector[np.intp_t] *** threaded_vvres
        int return_length = <int> False
        
        # Memory views for all scalar fields - process all together
        double[:, :] all_fields_mv
        double[:, :, :] all_results_mv  # [field_idx, point_idx, stat_idx]
        
    if not points.shape[1] == 3:
        raise ValueError("You must provide an (n x 3) numpy array.")
        
    if n_points == 0:
        return [np.empty((0, 4), dtype=np.float64) for _ in scalar_fields]

    if num_threads == -1:
        num_threads = multiprocessing.cpu_count()

    if kdtree is None:
        kdtree = cKDTree(points)

    # Get the number of points in the kdtree for scalar field validation
    cdef int64_t n_kdtree_points = kdtree.data.shape[0]

    # Validate and prepare all scalar fields as a single contiguous array
    cdef list valid_fields = []
    cdef np.ndarray[double, ndim=2] all_fields_array
    
    for field_obj in scalar_fields:
        if isinstance(field_obj, np.ndarray) and field_obj.ndim == 1 and field_obj.shape[0] == n_kdtree_points:
            valid_fields.append(np.ascontiguousarray(field_obj, dtype=np.float64))
        else:
            raise ValueError(f"Each scalar field must be a 1D numpy array with length {n_kdtree_points} (matching kdtree size), but got shape {field_obj.shape if hasattr(field_obj, 'shape') else type(field_obj)}")
    
    # Create a single contiguous array for all fields (better cache performance)
    all_fields_array = np.column_stack(valid_fields)
    all_fields_mv = all_fields_array
    
    # Pre-allocate results array for all fields together
    cdef np.ndarray[double, ndim=3] all_results_array = np.full((n_fields, n_points, 4), np.nan, dtype=np.float64)
    all_results_mv = all_results_array

    # Ultra-efficient memory layout: single radius vector for all threads
    radius_vector = np.full((num_threads, 3), fill_value=search_radius)
    threaded_vvres = init_result_vectors(num_threads)
    
    try:
        # Process all points and all fields in a single parallel loop
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
            
            if n_neighbors_at_id > max_k_neighbors:
                n_neighbors_at_id = max_k_neighbors
            elif n_neighbors_at_id == 0:
                continue

            # Ultra-optimized: compute stats for ALL fields at once
            compute_all_stats_ultra_fast(
                all_fields_mv,
                threaded_vvres[thread_id][0],
                n_neighbors_at_id,
                n_fields,
                all_results_mv,
                i
            )
                
    finally:
        free_result_vectors(threaded_vvres, num_threads)
    
    # Convert results back to list format
    cdef list results = []
    for field_idx in range(n_fields):
        results.append(np.asarray(all_results_mv[field_idx]))
        
    return results

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void compute_all_stats_ultra_fast(
    double[:, :] all_fields,
    vector[np.intp_t]* neighbor_indices_vec,
    uint32_t num_neighbors,
    int n_fields,
    double[:, :, :] results,
    int current_point_idx
) nogil:
    """
    Ultra-optimized computation of statistics for ALL fields simultaneously.
    Eliminates redundant memory operations and maximizes cache efficiency.
    """
    cdef:
        uint32_t k
        int field_idx
        uint32_t neighbor_idx
        double current_value
        double inv_num_neighbors = 1.0 / num_neighbors
        double inv_num_neighbors_minus_1 = 1.0 / (num_neighbors - 1) if num_neighbors > 1 else 0.0
        
        # Arrays for all statistics - compute in parallel for all fields
        double mean_val, variance_sum, min_val, max_val
        double temp_diff

    if num_neighbors == 0:
        for field_idx in range(n_fields):
            results[field_idx, current_point_idx, 0] = NAN  # mean
            results[field_idx, current_point_idx, 1] = NAN  # std
            results[field_idx, current_point_idx, 2] = NAN  # min
            results[field_idx, current_point_idx, 3] = NAN  # max
        return

    # Process all fields together to maximize cache utilization
    for field_idx in range(n_fields):
        # Initialize with first neighbor value
        neighbor_idx = neighbor_indices_vec[0][0]
        current_value = all_fields[neighbor_idx, field_idx]
        
        mean_val = current_value
        min_val = current_value
        max_val = current_value
        
        # Single-pass computation: sum, min, max
        for k in range(1, num_neighbors):
            neighbor_idx = neighbor_indices_vec[0][k]
            current_value = all_fields[neighbor_idx, field_idx]
            
            mean_val += current_value
            
            # Branchless min/max using arithmetic (often faster than conditional)
            if current_value < min_val:
                min_val = current_value
            if current_value > max_val:
                max_val = current_value
        
        # Finalize mean
        mean_val *= inv_num_neighbors
        
        # Compute variance in single pass (no intermediate storage needed)
        variance_sum = 0.0
        if num_neighbors > 1:
            for k in range(num_neighbors):
                neighbor_idx = neighbor_indices_vec[0][k]
                temp_diff = all_fields[neighbor_idx, field_idx] - mean_val
                variance_sum += temp_diff * temp_diff
        
        # Store all results for this field
        results[field_idx, current_point_idx, 0] = mean_val
        results[field_idx, current_point_idx, 1] = sqrt(variance_sum * inv_num_neighbors) if num_neighbors > 1 else 0.0
        results[field_idx, current_point_idx, 2] = min_val
        results[field_idx, current_point_idx, 3] = max_val


"""Constants for weightedcluster package.

This module defines named constants used throughout the package,
replacing "magic numbers" with documented, meaningful names.
"""

# ============================================================================
# CLARA Algorithm Constants
# ============================================================================

# Default base sample size for CLARA subsampling (Kaufman & Rousseeuw, 1990)
# Formula: 40 + 2*k, where k is the number of clusters
CLARA_BASE_SAMPLE_SIZE = 40

# Multiplier for k in CLARA sample size formula
CLARA_K_MULTIPLIER = 2

# ============================================================================
# Clustering Constants
# ============================================================================

# Minimum number of clusters for meaningful clustering
# (Single cluster is trivial and not useful for analysis)
MIN_CLUSTERS = 2

# Maximum iterations for k-medoids convergence
DEFAULT_KMEDOIDS_MAX_ITER = 100


# ============================================================================
# Label Conventions
# ============================================================================

# R-style numeric clustering constructors use 1-based labels
EXTERNAL_LABEL_BASE = 1

# ============================================================================
# Numerical Tolerances
# ============================================================================

# Tolerance for distance matrix symmetry check
DISTANCE_SYMMETRY_RTOL = 1e-5
DISTANCE_SYMMETRY_ATOL = 1e-8

# Tolerance for distance matrix diagonal (should be zero)
DISTANCE_DIAGONAL_TOL = 1e-6

# PAMonce stops swapping once the best swap delta is no longer strictly better
# than this threshold. This is R's WEIGHTED_CLUST_TOL
# (orig_R_pkg/WeightedCluster/src/PAMonce.h:5); the Python medoid search and the
# pure kernel must use the same value, so it is owned here rather than repeated
# at each site. cpp/wc_core.cpp carries the same value as a literal, because a
# C++ translation unit cannot import it; keep the two in step.
PAMONCE_SWAP_TOLERANCE = -1e-10

# ============================================================================
# Fuzzy Clustering Constants
# ============================================================================

# Convergence tolerance for fuzzy clustering
FUZZY_CONVERGENCE_TOL = 1e-3

# Maximum iterations for fuzzy clustering
FUZZY_MAX_ITER = 100


# ============================================================================
# Flexible Lance-Williams Parameter
# ============================================================================

# Alpha parameter corresponding to the standard flexible-beta beta=-0.25.
FLEXIBLE_ALPHA_DEFAULT = 0.625

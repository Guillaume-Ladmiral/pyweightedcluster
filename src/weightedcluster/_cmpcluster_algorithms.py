from __future__ import annotations

import numpy as np


def _embed_cmds(D: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Return classical-MDS coordinates from a distance matrix.

    ``tol`` is a relative spectral tolerance: eigenvalues no greater than
    ``tol * max(eigenvalues)`` are discarded.  Distances are divided by their
    finite positive maximum before squaring and the resulting coordinates are
    rescaled afterwards.  These two relative operations preserve the embedding
    under any representable common positive distance scale.
    """
    # Classical MDS: B = -0.5 * J * D^2 * J, eigendecomp of B
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    distance_scale = float(np.max(D)) if D.size else 0.0
    if distance_scale <= 0.0:
        return np.zeros((n, 1), dtype=float)
    scaled_distances = D / distance_scale
    J = np.eye(n) - np.ones((n, n), dtype=float) / n
    D2 = scaled_distances * scaled_distances
    B = -0.5 * (J @ D2 @ J)
    # symmetric eigendecomposition
    evals, evecs = np.linalg.eigh(B)
    # sort descending
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    # keep positive eigenvalues
    maxeval = float(np.max(evals)) if evals.size else 0.0
    if maxeval <= 0.0:
        return np.zeros((n, 1), dtype=float)
    relative_tolerance = max(float(tol), 1e-15)
    keep = (evals > 0.0) & (evals > relative_tolerance * maxeval)
    if not np.any(keep):
        # all non-positive; return zeros (degenerate)
        return np.zeros((n, 1), dtype=float)
    L = np.sqrt(evals[keep])
    V = evecs[:, keep]
    X_coords = (V * L) * distance_scale  # broadcast columns by sqrt eigenvalues
    return X_coords


def _diana_merge_structure(Dmat: np.ndarray) -> np.ndarray:
    """Port ``cluster`` 2.1.8.3's native ``splyt`` merge construction."""
    distances = np.asarray(Dmat, dtype=float)
    n_obs = int(distances.shape[0])
    order = np.arange(n_obs, dtype=int)
    cluster_sizes = np.zeros(n_obs, dtype=int)
    banner = np.zeros(n_obs, dtype=float)
    cluster_sizes[0] = n_obs
    diameter = float(np.max(distances))
    cluster_count = 1
    cluster_start = 0

    while cluster_count < n_obs:
        cluster_end = cluster_start + int(cluster_sizes[cluster_start]) - 1
        nonsplinter_end = cluster_end

        if cluster_sizes[cluster_start] == 2:
            cluster_sizes[cluster_start] = 1
            cluster_sizes[cluster_end] = 1
            banner[cluster_end] = distances[order[cluster_start], order[cluster_end]]
        else:
            largest_sum = -1.0
            first_shift = -1
            members = order[cluster_start : cluster_end + 1]
            for position in range(cluster_start, cluster_end + 1):
                distance_sum = float(np.sum(distances[order[position], members]))
                if largest_sum < distance_sum:
                    largest_sum = distance_sum
                    first_shift = position

            cluster_sizes[cluster_start] -= 1
            cluster_sizes[cluster_end] = 1
            if cluster_end != first_shift:
                shifted = int(order[first_shift])
                order[first_shift:cluster_end] = order[first_shift + 1 : cluster_end + 1].copy()
                order[cluster_end] = shifted

            splinter_size = 0
            nonsplinter_end = cluster_end - 1
            while True:
                splinter_size += 1
                nonsplinter_denominator = nonsplinter_end - cluster_start
                next_shift = -1
                largest_difference = -1.0
                nonsplinter = order[cluster_start : nonsplinter_end + 1]
                splinter = order[nonsplinter_end + 1 : cluster_end + 1]
                for position in range(cluster_start, nonsplinter_end + 1):
                    observation = order[position]
                    mean_nonsplinter = float(
                        np.sum(distances[observation, nonsplinter]) / nonsplinter_denominator
                    )
                    mean_splinter = float(np.sum(distances[observation, splinter]) / splinter_size)
                    difference = mean_nonsplinter - mean_splinter
                    if largest_difference < difference:
                        largest_difference = difference
                        next_shift = position

                splinter_start = nonsplinter_end + 1
                if largest_difference <= 0.0:
                    break

                if nonsplinter_end != next_shift:
                    shifted = int(order[next_shift])
                    order[next_shift:nonsplinter_end] = order[
                        next_shift + 1 : nonsplinter_end + 1
                    ].copy()
                    order[nonsplinter_end] = shifted
                for position in range(splinter_start, cluster_end + 1):
                    if order[position - 1] < order[position]:
                        break
                    previous = int(order[position - 1])
                    order[position - 1] = order[position]
                    order[position] = previous

                cluster_sizes[cluster_start] -= 1
                cluster_sizes[nonsplinter_end] = cluster_sizes[splinter_start] + 1
                cluster_sizes[splinter_start] = 0
                nonsplinter_end -= 1
                splinter_start = nonsplinter_end + 1
                if nonsplinter_end == cluster_start:
                    break

            if order[cluster_start] >= order[splinter_start]:
                first_part = order[cluster_start:splinter_start].copy()
                second_part = order[splinter_start : cluster_end + 1].copy()
                order[cluster_start : cluster_end + 1] = np.concatenate((second_part, first_part))
                second_size = int(cluster_sizes[splinter_start])
                cluster_sizes[splinter_start] = 0
                nonsplinter_end = cluster_start + cluster_end - nonsplinter_end - 1
                splinter_start = nonsplinter_end + 1
                cluster_sizes[splinter_start] = cluster_sizes[cluster_start]
                cluster_sizes[cluster_start] = second_size

            if cluster_count == 1:
                banner[splinter_start] = diameter
            else:
                members = order[cluster_start : cluster_end + 1]
                banner[splinter_start] = float(np.max(distances[np.ix_(members, members)]))

        cluster_count += 1
        if cluster_count >= n_obs:
            break

        next_start = None
        if cluster_end != n_obs - 1:
            position = cluster_start + int(cluster_sizes[cluster_start])
            while position < n_obs:
                if cluster_sizes[position] > 1:
                    next_start = position
                    break
                position += int(cluster_sizes[position])
        if next_start is None:
            position = 0
            while cluster_sizes[position] <= 1:
                position += int(cluster_sizes[position])
            next_start = position
        cluster_start = next_start

    merge = np.zeros((n_obs - 1, 2), dtype=int)
    for merge_index in range(n_obs - 1):
        minimum_height = diameter
        boundary = -1
        for position in range(1, n_obs):
            if cluster_sizes[position] >= 0 and minimum_height >= banner[position]:
                minimum_height = banner[position]
                boundary = position
        cluster_sizes[boundary] = -1
        left = -(int(order[boundary - 1]) + 1)
        right = -(int(order[boundary]) + 1)
        for previous_merge in range(merge_index):
            if left in merge[previous_merge]:
                left = previous_merge + 1
            if right in merge[previous_merge]:
                right = previous_merge + 1
        merge[merge_index] = (left, right)
    return merge


def _cut_r_merge(merge: np.ndarray, n_clusters: int) -> np.ndarray:
    """Cut an R ``hclust`` merge matrix after its first ``n-k`` merges."""
    n_obs = int(merge.shape[0] + 1)
    active = {
        -(observation + 1): np.asarray([observation], dtype=int) for observation in range(n_obs)
    }
    for merge_index, (left, right) in enumerate(merge[: n_obs - n_clusters]):
        members = np.concatenate((active.pop(int(left)), active.pop(int(right))))
        active[merge_index + 1] = members

    labels = np.empty(n_obs, dtype=int)
    for label, members in enumerate(
        sorted(active.values(), key=lambda values: int(values.min())), start=1
    ):
        labels[members] = label
    return labels


def _cluster_diana(Dmat: np.ndarray, maxk: int) -> dict[int, np.ndarray]:
    """Return R-compatible DIANA partitions through ``maxk`` clusters."""
    n_obs = int(Dmat.shape[0])
    merge = _diana_merge_structure(Dmat)
    return {
        n_clusters: _cut_r_merge(merge, n_clusters)
        for n_clusters in range(2, min(max(2, int(maxk)), n_obs) + 1)
    }

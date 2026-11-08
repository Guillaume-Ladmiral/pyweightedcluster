"""R ``fastcluster::hclust`` reproduced in Python, in R's own conventions.

Why this module exists
----------------------
``orig_R_pkg/WeightedCluster/NAMESPACE`` contains ``importFrom("fastcluster",
"hclust")``, so every unqualified ``hclust`` call inside WeightedCluster — most
importantly ``R/clustrangefamily.R:27`` (``wcCmpCluster``),
``R/clara_clustrange.R:101`` (``seqclararange``) and ``R/seqnullcqi.R:239`` —
runs **fastcluster's** hierarchical clustering, not ``stats::hclust``.  The two
break equal-distance merges differently, and for ``centroid`` they do not even
agree on the dendrogram heights.  Sequence dissimilarities are integer-valued,
so equal distances are the normal case in this project's domain rather than an
edge case.

This module is a faithful port of fastcluster's *stored matrix* algorithms:

* ``MST_linkage_core``     — single linkage (Rohlf's minimum spanning tree)
* ``NN_chain_core``        — complete, average, mcquitty and both Ward variants
  (Murtagh's nearest-neighbour chain)
* ``generic_linkage``      — centroid and median (Mullner's generic algorithm,
  with the binary min-heap its tie behaviour depends on)

together with fastcluster's dendrogram generation: a *stable* sort by merge
distance plus union-find relabelling for the first two families, and the
algorithm's own order for centroid and median.

The output is a SciPy-format linkage matrix, which is what
``cut_linkage_by_merge_count`` and ``as_clustrange`` already consume.  Merge
*order* is the load-bearing property, because R's ``cutree`` and this project's
``cut_linkage_by_merge_count`` both cut by merge count rather than by height.

Method conventions follow R, not SciPy: ``ward.D``, ``centroid`` and ``median``
consume the dissimilarities as given (R documents them as squared Euclidean
distances), while ``ward.D2`` squares the input and takes the square root of the
resulting heights.  ``members`` carries R's initial cluster masses and is used
only by the methods whose update formula depends on cluster size.
"""

from __future__ import annotations

import numpy as np

_NN_CHAIN_METHODS = frozenset({"complete", "average", "mcquitty", "ward.d", "ward.d2"})
_GENERIC_METHODS = frozenset({"centroid", "median"})
#: Methods whose Lance-Williams update reads cluster masses.
_MASS_METHODS = frozenset({"average", "ward.d", "ward.d2", "centroid"})
#: Every R ``hclust`` method this module reproduces, in R's lower-case spelling.
FASTCLUSTER_METHODS = frozenset({"single"}) | _NN_CHAIN_METHODS | _GENERIC_METHODS


class _BinaryMinHeap:
    """fastcluster's ``binary_min_heap``.

    The generic algorithm picks the next merge with ``argmin()``, so which of
    several equally small entries sits at the root decides the tie.  The sift
    order is therefore part of the contract and is reproduced exactly.
    """

    __slots__ = ("_values", "_size", "_at", "_position")

    def __init__(self, values: np.ndarray, size: int) -> None:
        self._values = values
        self._size = size
        self._at = list(range(size))
        self._position = list(range(size))

    def _height(self, index: int) -> float:
        return self._values[self._at[index]]

    def heapify(self) -> None:
        index = self._size >> 1
        while index > 0:
            index -= 1
            self._sift_down(index)

    def argmin(self) -> int:
        return self._at[0]

    def pop(self) -> None:
        self._size -= 1
        self._at[0] = self._at[self._size]
        self._position[self._at[0]] = 0
        self._sift_down(0)

    def update(self, entry: int, value: float) -> None:
        if value <= self._values[entry]:
            self.update_leq(entry, value)
        else:
            self.update_geq(entry, value)

    def update_leq(self, entry: int, value: float) -> None:
        self._values[entry] = value
        self._sift_up(self._position[entry])

    def update_geq(self, entry: int, value: float) -> None:
        self._values[entry] = value
        self._sift_down(self._position[entry])

    def _sift_up(self, index: int) -> None:
        while index > 0:
            parent = (index - 1) >> 1
            if not self._height(index) < self._height(parent):
                break
            self._swap(index, parent)
            index = parent

    def _sift_down(self, index: int) -> None:
        while True:
            child = 2 * index + 1
            if child >= self._size:
                break
            if self._height(child) >= self._height(index):
                child += 1
                if child >= self._size or self._height(child) >= self._height(index):
                    break
            elif child + 1 < self._size and self._height(child + 1) < self._height(child):
                child += 1
            self._swap(index, child)
            index = child

    def _swap(self, left: int, right: int) -> None:
        self._at[left], self._at[right] = self._at[right], self._at[left]
        self._position[self._at[left]] = left
        self._position[self._at[right]] = right


class _UnionFind:
    """fastcluster's ``union_find``: row indices to output node labels."""

    __slots__ = ("_parent", "_next_parent")

    def __init__(self, size: int) -> None:
        self._parent = [0] * (2 * size - 1)
        self._next_parent = size

    def find(self, node: int) -> int:
        while self._parent[node] != 0:
            node = self._parent[node]
        return node

    def union(self, left: int, right: int) -> None:
        self._parent[left] = self._parent[right] = self._next_parent
        self._next_parent += 1


def _minimum_spanning_tree(work: np.ndarray, n: int) -> list[tuple[int, int, float]]:
    """fastcluster ``MST_linkage_core`` — single linkage."""
    steps: list[tuple[int, int, float]] = []
    reach = np.full(n, np.inf)
    active = np.ones(n, dtype=bool)

    reach[1:] = work[0, 1:]
    nearest = 1 + int(np.argmin(reach[1:]))
    steps.append((0, nearest, float(reach[nearest])))

    for _ in range(1, n - 1):
        previous = nearest
        active[previous] = False
        remaining = np.flatnonzero(active[1:]) + 1
        reach[remaining] = np.minimum(reach[remaining], work[previous, remaining])
        nearest = int(remaining[int(np.argmin(reach[remaining]))])
        steps.append((previous, nearest, float(reach[nearest])))
    return steps


def _nearest_neighbour_chain(
    work: np.ndarray, n: int, method: str, masses: np.ndarray
) -> list[tuple[int, int, float]]:
    """fastcluster ``NN_chain_core`` — complete, average, mcquitty, Ward."""
    steps: list[tuple[int, int, float]] = []
    chain = np.zeros(n, dtype=np.intp)
    tip = 0
    active = np.ones(n, dtype=bool)
    left = right = 0
    shortest = np.inf

    for _ in range(n - 1):
        alive = np.flatnonzero(active)
        if tip <= 3:
            left = int(alive[0])
            chain[0] = left
            tip = 1
            others = alive[1:]
            right = int(others[int(np.argmin(work[left, others]))])
            shortest = float(work[left, right])
        else:
            # fastcluster keeps the head of the chain across merges rather than
            # rebuilding it; the retained prefix is what decides equal-distance
            # merges, so the pop of three is reproduced exactly.
            tip -= 3
            left = int(chain[tip - 1])
            right = int(chain[tip])
            shortest = float(work[left, right])

        while True:
            chain[tip] = right
            candidates = alive[alive != right]
            position = int(np.argmin(work[right, candidates]))
            candidate = float(work[right, candidates[position]])
            # A strict comparison against the running minimum keeps the chain's
            # previous element on a tie, which is how the chain terminates.
            if candidate < shortest:
                shortest = candidate
                left = int(candidates[position])
            right, left = left, int(chain[tip])
            tip += 1
            if right == chain[tip - 2]:
                break

        steps.append((left, right, shortest))
        if left > right:
            left, right = right, left
        left_mass = right_mass = 0.0
        if method in _MASS_METHODS:
            left_mass = float(masses[left])
            right_mass = float(masses[right])
            masses[right] += masses[left]

        active[left] = False
        others = np.flatnonzero(active)
        others = others[others != right]
        if others.size:
            to_left = work[left, others]
            to_right = work[right, others]
            if method == "complete":
                updated = np.maximum(to_left, to_right)
            elif method == "average":
                total = left_mass + right_mass
                updated = (left_mass / total) * to_left + (right_mass / total) * to_right
            elif method == "mcquitty":
                updated = 0.5 * (to_left + to_right)
            else:  # ward.d and ward.d2 share the update formula
                other_mass = masses[others]
                updated = (
                    (other_mass + left_mass) * to_left
                    - other_mass * shortest
                    + (other_mass + right_mass) * to_right
                ) / (left_mass + right_mass + other_mass)
            work[right, others] = updated
            work[others, right] = updated
    return steps


def _generic(
    work: np.ndarray, n: int, method: str, masses: np.ndarray
) -> list[tuple[int, int, float]]:
    """fastcluster ``generic_linkage`` — centroid and median."""
    steps: list[tuple[int, int, float]] = []
    last = n - 1
    nearest = np.zeros(last, dtype=np.intp)
    # ``lower_bound`` is fastcluster's ``mindist``: a lower bound on each row's
    # minimum, refreshed lazily.  The heap owns this array and mutates it.
    lower_bound = np.empty(last, dtype=float)
    label_of_row = np.arange(n, dtype=np.intp)
    active = np.ones(n, dtype=bool)

    for row in range(last):
        tail = work[row, row + 1 :]
        position = int(np.argmin(tail))
        lower_bound[row] = tail[position]
        nearest[row] = row + 1 + position

    heap = _BinaryMinHeap(lower_bound, last)
    heap.heapify()

    for step in range(last):
        alive = np.flatnonzero(active)
        left = heap.argmin()
        while lower_bound[left] < work[left, nearest[left]]:
            # The bound went stale: recompute this row's true minimum and let
            # the heap decide again.
            later = alive[alive > left]
            position = int(np.argmin(work[left, later]))
            nearest[left] = later[position]
            heap.update_geq(left, float(work[left, later[position]]))
            left = heap.argmin()

        heap.pop()
        right = int(nearest[left])
        left_label = int(label_of_row[left])
        right_label = int(label_of_row[right])
        distance = float(lower_bound[left])

        left_mass = right_mass = 0.0
        if method == "centroid":
            left_mass = float(masses[left])
            right_mass = float(masses[right])
            masses[right] += masses[left]

        steps.append((left_label, right_label, distance))

        active[left] = False
        label_of_row[right] = n + step
        alive = np.flatnonzero(active)

        # The two update formulas are kept apart rather than folded into one
        # weighted sum: fastcluster writes centroid as `s*a - stc + t*b` and
        # median as `(a + b)*0.5 - c/4`, and those associations do not round
        # identically.
        share_left = share_right = correction = quarter = 0.0
        if method == "centroid":
            total = left_mass + right_mass
            share_left = left_mass / total
            share_right = right_mass / total
            correction = share_left * share_right * distance
        else:
            quarter = distance * 0.25

        below = alive[alive < right]
        if below.size:
            if method == "centroid":
                updated = (
                    share_left * work[left, below] - correction + share_right * work[right, below]
                )
            else:
                updated = (work[left, below] + work[right, below]) * 0.5 - quarter
            work[right, below] = updated
            work[below, right] = updated
            improved = updated < lower_bound[below]
            for position in np.flatnonzero(improved):
                row = int(below[position])
                heap.update_leq(row, float(updated[position]))
                nearest[row] = right
            # A row that pointed at the absorbed cluster now points at its
            # survivor; only rows before it can, since ``nearest`` looks forward.
            redirect = (~improved) & (below < left) & (nearest[below] == left)
            nearest[below[redirect]] = right

        if right < last:
            above = alive[alive > right]
            if above.size:
                if method == "centroid":
                    updated = (
                        share_left * work[left, above]
                        - correction
                        + share_right * work[right, above]
                    )
                else:
                    updated = (work[left, above] + work[right, above]) * 0.5 - quarter
                work[right, above] = updated
                work[above, right] = updated
                position = int(np.argmin(updated))
                nearest[right] = above[position]
                heap.update(right, float(updated[position]))
    return steps


def _to_linkage(steps: list[tuple[int, int, float]], n: int, *, labelled: bool) -> np.ndarray:
    """fastcluster ``generate_SciPy_dendrogram``.

    ``labelled`` is fastcluster's ``sorted`` template argument: the generic
    algorithm already emits node labels in merge order, while the chain and
    tree algorithms emit row indices that a stable sort by distance and a
    union-find pass turn into labels.
    """
    linkage = np.zeros((n - 1, 4), dtype=float)
    if labelled:
        ordered = steps
        nodes = None
    else:
        order = sorted(range(len(steps)), key=lambda index: (steps[index][2], index))
        ordered = [steps[index] for index in order]
        nodes = _UnionFind(n)

    def size_of(label: int) -> float:
        return 1.0 if label < n else linkage[label - n, 3]

    for row, (first, second, distance) in enumerate(ordered):
        if nodes is None:
            left, right = first, second
        else:
            left = nodes.find(first)
            right = nodes.find(second)
            nodes.union(left, right)
        total = size_of(left) + size_of(right)
        if left > right:
            left, right = right, left
        linkage[row] = (left, right, distance, total)
    return linkage


def linkage_fastcluster(
    diss: np.ndarray,
    *,
    method: str,
    members: np.ndarray | None = None,
) -> np.ndarray:
    """Reproduce ``fastcluster::hclust(as.dist(diss), method, members)``.

    Parameters
    ----------
    diss:
        Square dissimilarity matrix.  Only the values are read; the caller's
        array is never modified.
    method:
        One of ``FASTCLUSTER_METHODS``, spelled as R spells it in lower case
        (``"ward.d"``, ``"ward.d2"``, ``"single"``, ``"complete"``,
        ``"average"``, ``"mcquitty"``, ``"median"``, ``"centroid"``).
    members:
        R's ``members`` argument — the initial mass of each observation.
        ``None`` means unit masses.

    Returns
    -------
    numpy.ndarray
        A SciPy-format ``(n - 1, 4)`` linkage matrix whose row order is the
        merge order R's ``cutree`` would apply.
    """
    if method not in FASTCLUSTER_METHODS:
        raise ValueError(f"unsupported hclust method {method}")
    work = np.array(diss, dtype=float, copy=True, order="C")
    if work.ndim != 2 or work.shape[0] != work.shape[1]:
        raise ValueError("diss must be a square matrix")
    n = int(work.shape[0])
    if n < 2:
        raise ValueError("diss must contain at least two observations")
    # fastcluster checks for NaN before clustering and refuses; R surfaces that
    # as `Error: NaN dissimilarity value.` from both `fastcluster::hclust` and
    # `WeightedCluster::wcCmpCluster`. Reproducing the refusal keeps a matrix
    # the public distance contract now accepts from silently producing a
    # meaningless hierarchy.
    if np.isnan(work).any():
        raise ValueError("NaN dissimilarity value.")
    if method == "ward.d2":
        work *= work
    masses = (
        np.ones(n, dtype=float)
        if members is None
        else np.array(members, dtype=float, copy=True).reshape(-1)
    )
    if masses.size != n:
        raise ValueError("members must have one entry per observation")

    if method == "single":
        steps = _minimum_spanning_tree(work, n)
        labelled = False
    elif method in _NN_CHAIN_METHODS:
        steps = _nearest_neighbour_chain(work, n, method, masses)
        labelled = False
    else:
        steps = _generic(work, n, method, masses)
        labelled = True

    if method == "ward.d2":
        steps = [(left, right, float(np.sqrt(distance))) for left, right, distance in steps]
    return _to_linkage(steps, n, labelled=labelled)

import numpy as np

from weightedcluster import wcKMedoids
from weightedcluster._seqdist import sequences_distance

pytest_plugins = ["tests.r_parity_utils"]


def _canonical_partition(labels: np.ndarray) -> tuple[tuple[int, ...], ...]:
    labels = np.asarray(labels).reshape(-1)
    groups = [tuple(int(i) for i in np.where(labels == value)[0]) for value in np.unique(labels)]
    return tuple(sorted(groups))


def test_parity_kmedoids_labels_up_to_permutation(r_parity_session):
    X = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ],
        dtype=int,
    )
    D = sequences_distance(X, metric="HAM", normalize=True)
    initial_medoids_1b = np.array([1, 3], dtype=int)
    py = wcKMedoids(D, 2, initialclust=initial_medoids_1b, npass=1, random_state=42)
    lab_py = np.asarray(py["clustering"])

    oracle = r_parity_session.wc_kmedoids(
        D,
        k=2,
        weights=None,
        initialclust=initial_medoids_1b,
        method="PAMonce",
        npass=0,
    )
    lab_r = np.asarray(oracle["clustering"])

    assert _canonical_partition(lab_py) == _canonical_partition(lab_r)

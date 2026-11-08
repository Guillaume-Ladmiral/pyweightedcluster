import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from weightedcluster import _wc_core

OWNED_CORE_SYMBOLS = {"weighted_medoids", "cluster_quality"}
FOREIGN_DISTANCE_SYMBOLS = {
    "seq_hamming",
    "seq_hamming_between",
    "seq_om",
    "seq_om_between",
    "seq_lcs",
    "seq_lcs_between",
    "seq_dhd",
    "seq_dhd_between",
}


def test_compiled_core_exports_only_owned_algorithm_symbols():
    exported = {name for name in dir(_wc_core) if not name.startswith("_")}

    assert OWNED_CORE_SYMBOLS <= exported
    assert FOREIGN_DISTANCE_SYMBOLS.isdisjoint(exported)


def test_weighted_medoids():
    # simple square distance matrix
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=float,
    )
    # Hamming on binary vectors, constructed without a foreign core distance.
    Xi = X.astype(int)
    D = np.mean(Xi[:, None, :] != Xi[None, :, :], axis=2)
    labels, medoids, obj = _wc_core.weighted_medoids(D, 2, None, 50, 2, 0)
    assert labels.shape == (4,)
    assert medoids.shape == (2,)
    assert obj >= 0.0


@pytest.mark.parametrize(
    "initial",
    (
        np.array([[0, 2]], dtype=int),
        np.array([0, 0], dtype=int),
        np.array([0, 4], dtype=int),
        np.array([0.5, 2.0], dtype=float),
    ),
)
def test_compiled_medoids_rejects_invalid_initialization(initial):
    D = np.abs(np.arange(4.0)[:, None] - np.arange(4.0)[None, :])

    with pytest.raises(ValueError, match="init_medoids"):
        _wc_core.weighted_medoids(D, 2, init_medoids=initial)


def test_compiled_label_validation_cannot_escape_as_native_memory_error():
    script = textwrap.dedent("""
        import json
        import numpy as np
        from weightedcluster import _wc_core

        D = np.array([[0.0, 1.0], [1.0, 0.0]])
        cases = [
            lambda: _wc_core.cluster_quality(D, np.array([-1, 0]), None),
            lambda: _wc_core.cluster_quality(D, np.array([0.5, 1.0]), None),
            lambda: _wc_core.cluster_quality(D, np.array([0, 2]), None),
        ]
        caught = []
        for case in cases:
            try:
                case()
            except ValueError as exc:
                caught.append(str(exc))
        print(json.dumps(caught))
        """)

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert len(json.loads(completed.stdout)) == 3


def test_compiled_kernels_reject_nonfinite_inputs():
    D = np.array([[0.0, np.inf], [np.inf, 0.0]])
    finite_D = np.array([[0.0, 1.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match="finite"):
        _wc_core.weighted_medoids(D, 2)
    with pytest.raises(ValueError, match="finite"):
        _wc_core.cluster_quality(finite_D, np.array([0, 1]), np.array([1.0, np.nan]))

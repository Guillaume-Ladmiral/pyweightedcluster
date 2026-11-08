from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from weightedcluster._utils import generate_random_seed
from weightedcluster.clustassoc import _clustassoc_plot_data
from weightedcluster.seqnull import _LINKAGE_WARNING_WORK, _warn_for_linkage_workload


@pytest.mark.parametrize(
    ("stat", "expected"),
    [
        ("Unaccounted", "Unreproduced proportion of the association"),
        ("Remaining", "Unexplained association"),
        ("BIC", "BIC"),
        ("numcluster", "Number of clusters"),
    ],
)
def test_clustassoc_plot_labels_all_valid_statistics(stat, expected):
    assoc = {
        "Unaccounted": [1.0, 0.5],
        "Remaining": [4.0, 2.0],
        "BIC": [10.0, 8.0],
        "numcluster": [1, 2],
    }
    assert _clustassoc_plot_data(assoc, stat=stat)["ylab"] == expected


def test_seedsequence_derivation_separates_audit_collision_pairs_and_paths():
    assert generate_random_seed(42, 5) != generate_random_seed(43, 4)
    assert generate_random_seed(42, 1, 4) != generate_random_seed(42, 2, 3)
    assert generate_random_seed(42, 1, 4) == generate_random_seed(42, 1, 4)
    for seed in (
        generate_random_seed(0, 0),
        generate_random_seed(2**32 - 1, 99, 12),
    ):
        assert 0 <= seed < 2**31


def test_large_hierarchical_null_workload_warns_but_small_or_kmedoid_does_not():
    # 2026-09-03 audit WC-090302: since the F-01 reroute the Ward.D
    # reclustering runs through the quadratic nearest-neighbour-chain port,
    # so the workload heuristic and the warning text must say quadratic and
    # scale with R * N**2, not R * N**3.
    boundary_n = 100
    boundary_replicates = _LINKAGE_WARNING_WORK // boundary_n**2
    with pytest.warns(RuntimeWarning, match=r"quadratic.*N=100.*R=100"):
        _warn_for_linkage_workload(
            n_obs=boundary_n,
            replicates=boundary_replicates,
            kmedoid=False,
        )
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        _warn_for_linkage_workload(
            n_obs=boundary_n,
            replicates=boundary_replicates - 1,
            kmedoid=False,
        )
        _warn_for_linkage_workload(
            n_obs=boundary_n,
            replicates=boundary_replicates,
            kmedoid=True,
        )
    assert not recorded

    # The two user documents drifted for a year after the F-01 reroute while
    # only the code changed (WC-090302), so their seqnullcqi sections share
    # this drift detector with the warning text above.
    repo_root = Path(__file__).resolve().parents[1]
    for document in ("docs/API_INDEX.md", "docs/R_MIGRATION_GUIDE.md"):
        text = (repo_root / document).read_text(encoding="utf-8")
        start = text.index("### `seqnullcqi")
        section = text[start : start + 2000]
        assert "quadratic-time" in section, document
        assert "nearest-neighbour-chain" in section, document
        assert "cubic" not in section, document


def test_seqclustname_type_boundary_is_explicit(monkeypatch):
    module = importlib.import_module("weightedcluster.seqclustname")

    monkeypatch.setattr(module, "seqformat", lambda *args, **kwargs: ("A", "B"))
    sequences = np.array([[0, 0], [1, 1]])
    D = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(TypeError, match="seqformat.*list"):
        module.seqclustname(sequences, [1, 2], D)


def test_library_validation_no_longer_depends_on_assert_statements():
    import weightedcluster._boot_controls as boot_controls
    import weightedcluster.seqclustname as seqclustname_module

    assert "assert medoids is not None" not in inspect.getsource(boot_controls)
    assert "assert isinstance(labels, list)" not in inspect.getsource(seqclustname_module)


@pytest.mark.parametrize("optimized", [False, True], ids=["ordinary", "python-O"])
def test_seqclustname_type_boundary_survives_optimized_python(optimized):
    command = [sys.executable]
    if optimized:
        command.append("-O")
    command.extend(
        [
            "-c",
            """
import importlib
import numpy as np

from weightedcluster._boot_controls import _validate_replay_samples

validated = _validate_replay_samples(
    [np.array([0, 2, 1])],
    R=1,
    sample_size=3,
    n_obs=3,
    sampling='medoids',
    strata_idx=None,
    medoids=np.array([0, 2]),
    clustering={'cluster2': np.array([1, 1, 2])},
    kvals=[2],
)
assert np.array_equal(validated[0], [0, 2, 1])

module = importlib.import_module('weightedcluster.seqclustname')
module.seqformat = lambda *args, **kwargs: ('A', 'B')
try:
    module.seqclustname(
        np.array([[0, 0], [1, 1]]),
        [1, 2],
        np.array([[0.0, 1.0], [1.0, 0.0]]),
    )
except Exception as exc:
    print(type(exc).__name__ + ':' + str(exc))
else:
    raise SystemExit('boundary unexpectedly accepted a non-list result')
""",
        ]
    )
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    assert completed.stdout.strip() == (
        "TypeError:traminex.seqformat(..., to='SPS') must return a list"
    )

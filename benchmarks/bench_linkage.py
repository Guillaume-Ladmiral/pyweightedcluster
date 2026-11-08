"""Measure the R-faithful weighted linkage path without changing its semantics.

The benchmark first checks a fixed deterministic equivalence corpus, then
times ``ward.d`` on several Euclidean distance matrices.  Its JSON output is
intended as durable audit evidence, not as a machine-independent speed gate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from weightedcluster.cmpcluster import _linkage_hclust

_FIXTURE_DISTANCE = np.array(
    [
        [0.0, 1.0, 4.0, 5.0],
        [1.0, 0.0, 3.0, 4.0],
        [4.0, 3.0, 0.0, 2.0],
        [5.0, 4.0, 2.0, 0.0],
    ]
)
_FIXTURE_EXPECTED = {
    "unit": np.array([[0.0, 1.0, 1.0, 2.0], [2.0, 3.0, 2.0, 2.0], [4.0, 5.0, 6.5, 4.0]]),
    "weighted": np.array(
        [
            [0.0, 1.0, 1.0, 2.0],
            [2.0, 3.0, 2.0, 2.0],
            [4.0, 5.0, 6.714285714285714, 4.0],
        ]
    ),
}


def verify_equivalence_corpus() -> dict[str, object]:
    """Check fixed unit/weighted Ward.D results and deterministic tie order."""
    cases = {
        "unit": None,
        "weighted": np.array([1.0, 2.0, 1.0, 3.0]),
    }
    checked: list[str] = []
    for name, weights in cases.items():
        first = _linkage_hclust(_FIXTURE_DISTANCE, method="ward.d", members=weights)
        second = _linkage_hclust(_FIXTURE_DISTANCE, method="ward.d", members=weights)
        np.testing.assert_allclose(first, _FIXTURE_EXPECTED[name], rtol=0.0, atol=1e-14)
        np.testing.assert_array_equal(first, second)
        checked.append(name)
    return {"status": "PASS", "cases": checked}


def generate_distance(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic Euclidean matrix and positive nonuniform masses."""
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n, 4))
    delta = points[:, None, :] - points[None, :, :]
    distance = np.sqrt(np.sum(delta * delta, axis=2))
    np.fill_diagonal(distance, 0.0)
    weights = 0.25 + rng.random(n)
    return distance, weights


def benchmark(sizes: list[int], repeats: int, seed: int) -> dict[str, object]:
    """Run the equivalence checks and multi-size timing study."""
    if len(sizes) < 3 or any(size < 4 for size in sizes):
        raise ValueError("sizes must contain at least three integers >= 4")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    equivalence = verify_equivalence_corpus()
    timings: list[dict[str, object]] = []
    for size in sizes:
        distance, weights = generate_distance(size, seed + size)
        samples: list[float] = []
        reference: np.ndarray | None = None
        for _ in range(repeats):
            started = time.perf_counter()
            hierarchy = _linkage_hclust(distance, method="ward.d", members=weights)
            samples.append(time.perf_counter() - started)
            if reference is None:
                reference = hierarchy
            else:
                np.testing.assert_array_equal(hierarchy, reference)
        median = statistics.median(samples)
        timings.append(
            {
                "n": size,
                "median_seconds": median,
                "seconds_per_n_cubed": median / float(size**3),
                "samples_seconds": samples,
            }
        )
    return {
        "schema_version": "1.0",
        "operation": "weightedcluster.cmpcluster._linkage_hclust(method='ward.d')",
        "complexity_model": "O(N^3) time and O(N^2) memory per replicate",
        "equivalence": equivalence,
        "timings": timings,
    }


def _parse_sizes(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=_parse_sizes, default=_parse_sizes("64,96,144,216"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark(args.sizes, args.repeats, args.seed)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

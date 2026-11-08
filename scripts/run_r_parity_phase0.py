from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

REFERENCE_STATS = ("PBC", "HG", "HGSD", "ASW", "ASWw", "CH", "R2", "CHsq", "R2sq", "HC")
STAT_ABSDIFF_TOLERANCE = 1e-8
SILHOUETTE_ABSDIFF_TOLERANCE = 1e-10
CLUSTRANGE_ABSDIFF_TOLERANCE = 1e-10


def _absdiff(a: float, b: float) -> float:
    if np.isfinite(a) and np.isfinite(b):
        return float(abs(a - b))
    if (not np.isfinite(a)) and (not np.isfinite(b)):
        return 0.0
    return float("inf")


def _parity_problems(payload: dict[str, object]) -> list[str]:
    """Return deterministic failures for the bounded Phase-0 contract."""
    problems: list[str] = []
    if payload["wcKMedoids_partition_match"] is not True:
        problems.append("wcKMedoids partition differs from the live R oracle")
    if payload["wcKMedoids_discriminating_partition_match"] is not True:
        problems.append("wcKMedoids discriminating partition differs from the live R oracle")
    discriminating_objective = float(payload["wcKMedoids_discriminating_objective_absdiff"])
    if (
        not np.isfinite(discriminating_objective)
        or discriminating_objective > STAT_ABSDIFF_TOLERANCE
    ):
        problems.append("wcKMedoids discriminating objective exceeds tolerance")
    for family, label in (
        ("wcKMedoids_stats_absdiff", "wcKMedoids statistic"),
        ("wcClusterQuality_stats_absdiff", "wcClusterQuality statistic"),
    ):
        values = payload[family]
        assert isinstance(values, dict)
        for name, difference in values.items():
            value = float(difference)
            if not np.isfinite(value) or value > STAT_ABSDIFF_TOLERANCE:
                problems.append(f"{label} {name} exceeds tolerance")
    silhouette = payload["wcSilhouetteObs_max_absdiff"]
    assert isinstance(silhouette, dict)
    for name, difference in silhouette.items():
        value = float(difference)
        if not np.isfinite(value) or value > SILHOUETTE_ABSDIFF_TOLERANCE:
            problems.append(f"wcSilhouetteObs {name} exceeds tolerance")
    clustrange = float(payload["as_clustrange_max_absdiff"])
    if not np.isfinite(clustrange) or clustrange > CLUSTRANGE_ABSDIFF_TOLERANCE:
        problems.append("as_clustrange statistics exceed tolerance")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase-0 R parity checks.")
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    r_parity_utils = importlib.import_module("tests.r_parity_utils")
    weightedcluster = importlib.import_module("weightedcluster")
    canonical_partition = r_parity_utils.canonical_partition
    create_r_parity_session = r_parity_utils.create_r_parity_session
    exact_oracle_version_problems = r_parity_utils.exact_oracle_version_problems
    as_clustrange = weightedcluster.as_clustrange
    wcClusterQuality = weightedcluster.wcClusterQuality
    wcKMedoids = weightedcluster.wcKMedoids
    wcSilhouetteObs = weightedcluster.wcSilhouetteObs

    session, reason = create_r_parity_session()
    if session is None:
        payload = {"available": False, "passed": False, "reason": reason}
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"R parity unavailable: {reason}")
        return 2
    versions = session.oracle_versions()
    version_problems = exact_oracle_version_problems(versions)
    if version_problems:
        payload = {
            "available": True,
            "passed": False,
            "versions": versions,
            "problems": ["Live R oracle version mismatch: " + "; ".join(version_problems)],
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(payload["problems"][0])
        return 2

    D = np.array(
        [
            [0.0, 1.0, 2.0, 5.0, 6.0, 7.0],
            [1.0, 0.0, 1.0, 4.0, 5.0, 6.0],
            [2.0, 1.0, 0.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 0.0, 1.0, 2.0],
            [6.0, 5.0, 4.0, 1.0, 0.0, 1.0],
            [7.0, 6.0, 5.0, 2.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    weights = np.array([1.0, 2.0, 1.5, 1.0, 0.5, 1.2], dtype=float)
    labels2 = np.array([1, 1, 1, 2, 2, 2], dtype=int)
    labels3 = np.array([1, 1, 2, 2, 3, 3], dtype=int)
    initial_medoids_1b = np.array([1, 5], dtype=int)

    py_km = wcKMedoids(
        D,
        2,
        weights=weights,
        npass=0,
        initialclust=initial_medoids_1b,
        random_state=42,
    )
    r_km = session.wc_kmedoids(
        D,
        k=2,
        weights=weights,
        initialclust=initial_medoids_1b,
        method="PAMonce",
        npass=0,
    )
    partition_match = canonical_partition(py_km["clustering"]) == canonical_partition(
        r_km["clustering"]
    )
    km_stats_diff = {
        key: _absdiff(float(py_km["stats"].get(key, np.nan)), float(r_km["stats"].get(key, np.nan)))
        for key in REFERENCE_STATS
    }

    discriminating_D = np.array(
        [
            [0, 2.5, 0.5, 1, 1.5, 0.5, 1, 0, 0.5],
            [2.5, 0, 1, 1.5, 2, 0, 1, 1.5, 1],
            [0.5, 1, 0, 0, 1, 1.5, 1, 1, 0],
            [1, 1.5, 0, 0, 1, 2, 2, 1.5, 0.5],
            [1.5, 2, 1, 1, 0, 0.5, 0, 3, 3],
            [0.5, 0, 1.5, 2, 0.5, 0, 0.5, 1, 1],
            [1, 1, 1, 2, 0, 0.5, 0, 1.5, 2.5],
            [0, 1.5, 1, 1.5, 3, 1, 1.5, 0, 0.5],
            [0.5, 1, 0, 0.5, 3, 1, 2.5, 0.5, 0],
        ],
        dtype=float,
    )
    discriminating_weights = np.array([2, 1, 3, 0, 1, 4, 2, 1, 4], dtype=float)
    discriminating_start_1b = np.array([1, 6], dtype=int)
    py_discriminating = wcKMedoids(
        discriminating_D,
        2,
        weights=discriminating_weights,
        npass=0,
        initialclust=discriminating_start_1b,
    )
    r_discriminating = session.wc_kmedoids(
        discriminating_D,
        k=2,
        weights=discriminating_weights,
        initialclust=discriminating_start_1b,
        method="PAMonce",
        npass=0,
    )
    discriminating_partition_match = canonical_partition(
        py_discriminating["clustering"]
    ) == canonical_partition(r_discriminating["clustering"])
    discriminating_objective_absdiff = _absdiff(
        py_discriminating["info"]["Sum of Distances"],
        r_discriminating["info"]["Sum of Distances"],
    )

    py_cq = wcClusterQuality(D, labels2, weights=weights)
    r_cq = session.wc_cluster_quality(D, labels2, weights=weights)
    cq_stats_diff = {
        key: _absdiff(float(py_cq["stats"].get(key, np.nan)), float(r_cq["stats"].get(key, np.nan)))
        for key in REFERENCE_STATS
    }

    py_s_asw = np.asarray(wcSilhouetteObs(D, labels2, weights=weights, measure="ASW"), dtype=float)
    py_s_asww = np.asarray(
        wcSilhouetteObs(D, labels2, weights=weights, measure="ASWw"), dtype=float
    )
    r_s_asw = session.wc_silhouette_obs(D, labels2, weights=weights, measure="ASW")
    r_s_asww = session.wc_silhouette_obs(D, labels2, weights=weights, measure="ASWw")
    sil_asw_max = float(np.nanmax(np.abs(py_s_asw - r_s_asw)))
    sil_asww_max = float(np.nanmax(np.abs(py_s_asww - r_s_asww)))

    py_cr = as_clustrange({"cluster2": labels2, "cluster3": labels3}, D, weights=weights)
    r_cr = session.as_clustrange_default(
        D,
        {"cluster2": labels2, "cluster3": labels3},
        weights=weights,
    )
    r_names = list(r_cr["stats_names"])
    py_stats_mat = np.column_stack(
        [np.asarray(py_cr["stats"].get(name, []), dtype=float).reshape(-1) for name in r_names]
    )
    clustrange_max = float(np.nanmax(np.abs(py_stats_mat - r_cr["stats_matrix"])))

    payload = {
        "available": True,
        "versions": versions,
        "wcKMedoids_partition_match": bool(partition_match),
        "wcKMedoids_discriminating_partition_match": bool(discriminating_partition_match),
        "wcKMedoids_discriminating_objective_absdiff": discriminating_objective_absdiff,
        "wcKMedoids_stats_absdiff": km_stats_diff,
        "wcClusterQuality_stats_absdiff": cq_stats_diff,
        "wcSilhouetteObs_max_absdiff": {"ASW": sil_asw_max, "ASWw": sil_asww_max},
        "as_clustrange_max_absdiff": clustrange_max,
    }
    problems = _parity_problems(payload)
    payload["passed"] = not problems
    payload["problems"] = problems
    payload["tolerances"] = {
        "statistics_absolute": STAT_ABSDIFF_TOLERANCE,
        "silhouette_absolute": SILHOUETTE_ABSDIFF_TOLERANCE,
        "clustrange_absolute": CLUSTRANGE_ABSDIFF_TOLERANCE,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("Phase-0 parity snapshot")
        print(f"- wcKMedoids partition match: {payload['wcKMedoids_partition_match']}")
        print(
            "- wcKMedoids discriminating partition match: "
            f"{payload['wcKMedoids_discriminating_partition_match']}"
        )
        print(
            "- wcKMedoids discriminating objective abs diff: "
            f"{payload['wcKMedoids_discriminating_objective_absdiff']:.6g}"
        )
        print(
            "- wcSilhouetteObs max abs diff: "
            f"ASW={payload['wcSilhouetteObs_max_absdiff']['ASW']:.6g}, "
            f"ASWw={payload['wcSilhouetteObs_max_absdiff']['ASWw']:.6g}"
        )
        print(f"- as_clustrange max abs diff: {payload['as_clustrange_max_absdiff']:.6g}")
        print("- wcClusterQuality metric abs diffs:")
        for key in REFERENCE_STATS:
            print(f"  - {key}: {payload['wcClusterQuality_stats_absdiff'][key]:.6g}")
        print(f"- passed: {payload['passed']}")
        for problem in problems:
            print(f"  - failure: {problem}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Capture the exact installed WeightedCluster 2.0 namespace contract."""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

MANDATED_PYTHON = Path("~/miniforge3/envs/datasci/bin/python").expanduser()
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "tests" / "fixtures" / "weightedcluster_2_0_namespace.json"

if Path(sys.executable).resolve() != MANDATED_PYTHON.resolve():
    raise SystemExit(
        "This capture must run with "
        f"{MANDATED_PYTHON}; received {Path(sys.executable).resolve()}"
    )

sys.path.insert(0, str(REPO_ROOT))

from rpy2 import rinterface  # noqa: E402
from rpy2 import robjects as ro  # noqa: E402
from tests.r_parity_utils import (  # noqa: E402
    EXPECTED_TRAMINER_VERSION,
    EXPECTED_WC_VERSION,
    create_r_parity_session,
    exact_oracle_version_problems,
)


def _formal_rows(export_name: str, r) -> list[dict[str, object]]:
    r.assign("wc_capture_export", ro.StrVector([export_name]))
    formals = r("formals(getExportedValue('WeightedCluster', as.character(wc_capture_export)))")
    deparse_default = r(
        "function(pair) { value <- pair[[1]]; "
        "paste(deparse(value, width.cutoff=500L), collapse=' ') }"
    )
    rows: list[dict[str, object]] = []
    for index, name in enumerate(formals.names):
        value = formals[index][0]
        missing = bool(value.rsame(rinterface.MissingArg))
        required = str(name) != "..." and missing
        default = None if missing else str(deparse_default(formals[index])[0])
        rows.append(
            {
                "default": default,
                "name": str(name),
                "required": required,
            }
        )
    return rows


def capture_namespace() -> dict[str, object]:
    session, reason = create_r_parity_session()
    if session is None:
        raise SystemExit(reason or "The live R oracle is unavailable")
    versions = session.oracle_versions()
    problems = exact_oracle_version_problems(versions)
    if problems:
        raise SystemExit("Live R oracle version mismatch: " + "; ".join(problems))

    r = session.r
    export_names = sorted(str(value) for value in r('getNamespaceExports("WeightedCluster")'))
    registrations = r('getNamespaceInfo("WeightedCluster", "S3methods")')
    r.assign("wc_capture_s3", registrations)
    s3_methods = [
        {
            "class": str(class_name),
            "generic": str(generic),
            "implementation": str(implementation),
        }
        for generic, class_name, implementation in zip(
            r("as.character(wc_capture_s3[, 1])"),
            r("as.character(wc_capture_s3[, 2])"),
            r("as.character(wc_capture_s3[, 3])"),
            strict=True,
        )
    ]
    return {
        "exports": [
            {"formals": _formal_rows(export_name, r), "name": export_name}
            for export_name in export_names
        ],
        "packages": {
            "TraMineR": EXPECTED_TRAMINER_VERSION,
            "WeightedCluster": EXPECTED_WC_VERSION,
        },
        "s3_methods": s3_methods,
        "schema_version": 1,
    }


def _render(data: dict[str, object]) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write the captured fixture")
    mode.add_argument("--check", action="store_true", help="verify the committed fixture")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="fixture path (default: tests/fixtures/weightedcluster_2_0_namespace.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output.resolve()
    rendered = _render(capture_namespace())
    if args.write:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8", newline="\n")
        print(f"wrote {output}")
        return 0

    if not output.is_file():
        print(f"missing namespace fixture: {output}", file=sys.stderr)
        return 1
    committed = output.read_text(encoding="utf-8")
    if committed == rendered:
        print(f"namespace fixture is current: {output}")
        return 0
    diff = difflib.unified_diff(
        committed.splitlines(),
        rendered.splitlines(),
        fromfile=f"{output} (committed)",
        tofile=f"{output} (live R)",
        lineterm="",
        n=2,
    )
    print("\n".join(diff), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

# Contributing to weightedcluster (Python)

This repository ships the Python package importable as `weightedcluster`.
The current release line is `0.1.x`; see `CHANGELOG.md` for the version history
(the version is single-sourced from `pyproject.toml`).

Use this guide for active repository work. Historical reports in `archive/` and the vendored
R package materials under `orig_R_pkg/` are reference-only and must not be treated as the live
project contract.

## Development Environment

Ubuntu 24.04 is the canonical and primary development platform. The supported
Python matrix is 3.11 and 3.12. Local development, testing, linting, and
packaging use the environment `.venv`, built on this exact interpreter with
`--system-site-packages` so that `datasci` itself is never changed:

```bash
~/miniforge3/envs/datasci/bin/python
```

The runtime dependency `traminex` is a local sibling project. Install it first,
then install and verify this project from the repository root:

```bash
~/miniforge3/envs/datasci/bin/python -m venv --system-site-packages .venv
PY=.venv/bin/python
$PY -m pip install --no-deps --no-build-isolation -e ../PROJECT_TRAMINEX
$PY -m pip install --no-build-isolation -e ".[dev,numba]"
$PY -m pytest -q
WC_PARITY_STRICT=1 $PY -m pytest -q
$PY -m ruff check .
$PY -m black --check .
$PY -m build
```

Optional local prerequisites:
- Git
- A C++17-capable compiler if you want to build the compiled extension locally
- Quarto if you want to render tutorial notebooks

## Documentation Layout

Active documentation lives here:
- `README.md`
- `docs/API_INDEX.md`
- `docs/ARCHITECTURE.md`
- `docs/WORKFLOW_GUIDE.md`
- `docs/R_MIGRATION_GUIDE.md`
- `docs/vignettes/`

Maintainer-only notes live here:
- `docs/maintainers/`

Reference-only and historical material lives here:
- `docs/legacy/`
- `orig_R_pkg/`
- `archive/`

New tutorials and worked examples belong in `docs/vignettes/`, not in a root-level
`vignettes/` directory.

## Change Expectations

For code changes:
- Read the relevant implementation directly before editing. Do not rely on documentation alone.
- Remove deprecated functions and parameters instead of preserving compatibility shims.
- Prefer fixing incorrect behavior over adding compatibility layers. This project does not carry
  backward-compatibility obligations to unreleased draft states.
- Update tests with every behavior change.
- Update the active docs when public behavior, parameters, or return structures change.

For documentation changes:
- Keep user-facing guidance in the canonical docs under `docs/`.
- Keep maintainer backlog/status notes under `docs/maintainers/`.
- Move superseded audits, milestone summaries, and draft notes into `archive/`.
- Treat vendored R materials as upstream reference, not as Python package documentation.

## Coding and Review Standards

- Follow the existing public API and naming conventions unless you are intentionally removing or
  replacing a deprecated surface.
- Keep exception handling specific. Broad `except Exception` fallbacks are not acceptable unless
  there is a narrow, documented reason.
- Add concise tests for regressions, edge cases, and parity-sensitive behavior.
- Preserve deterministic behavior when random seeds or `random_state` are part of the API.

## Submitting Work

Before opening a pull request:

1. Install the package in editable mode with the mandated interpreter.
2. Run the ordinary and strict live-R suites, lint, formatting, and build checks
   shown above.
3. Update documentation and changelog/release notes if the change affects public behavior.
4. Confirm any archived or historical material you touched is still clearly labeled as
   non-canonical.

Pull requests should explain:
- what changed
- why it changed
- what tests were added or updated
- whether any docs or release notes were updated

## Reporting Issues

- Bug reports: include a minimal reproduction, expected behavior, actual behavior, and traceback
  where applicable.
- Feature requests: explain the use case and the proposed API or workflow impact.
- Security issues: follow `SECURITY.md` and do not post them in public issues.

## License

By contributing, you agree that your contributions are licensed under the repository license.

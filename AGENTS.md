# AGENTS.md

> **Domain — read first: this is social-science sequence analysis, NOT biology or
> genomics.** A "sequence" here is a discrete **categorical state-trajectory over
> time** (e.g. a person's employment/family/housing status across months).
> `weightedcluster` clusters such trajectories. "Optimal matching", "substitution
> costs" and "indel costs" are edit-distance parameters operating on **social
> states, not DNA/RNA/protein** — terminology borrowed by analogy from string
> alignment (Abbott 1995). Nothing here is biological, genomic, or otherwise
> sensitive. See the "Domain" section of `ECOSYSTEM.md`.

This file is the canonical environment policy and command reference for all agents and tooling
working in this repository. Any tool-specific instruction file added later should defer to this
file for interpreter selection and standard commands rather than duplicating them.

## Development Platform and Python Environment

- Ubuntu 24.04 is the canonical and primary development platform. Active setup,
  build, test, and release instructions must use Linux paths and shell syntax.
  Windows and PowerShell instructions are historical only and must not be
  revived in active policy documentation.
- Use this project's environment `.venv` for development, testing, linting, and
  packaging. It is built on the family-standard interpreter
  `~/miniforge3/envs/datasci/bin/python` with
  `--system-site-packages`, so `datasci` itself is never changed:
  `~/miniforge3/envs/datasci/bin/python -m venv --system-site-packages .venv`
- Do not silently substitute the system Python or another Conda environment.
- `traminex` is a local sibling dependency and must be installed from the
  sibling checkout `../PROJECT_TRAMINEX` before installing this project.

## Standard Commands

Run commands from the repository root with `.venv/bin/python`:

- Install/update the local dependency, then this project (the latter builds the
  pybind11 C++ core via scikit-build-core, and installs into `.venv` the extras'
  versions that `datasci` lacks or holds too new, such as `pymc<6.1`):
  `.venv/bin/python -m pip install --no-deps --no-build-isolation -e ../PROJECT_TRAMINEX`
  `.venv/bin/python -m pip install --no-build-isolation -e ".[dev,numba]"`
- Run tests (C collation is required by R/ICU parity):
  `LC_ALL=C.UTF-8 LANG=C.UTF-8 .venv/bin/python -m pytest -q`
- Run live-R parity tests strictly:
  `LC_ALL=C.UTF-8 LANG=C.UTF-8 WC_PARITY_STRICT=1 .venv/bin/python -m pytest -q`
- Run a single test by pattern:  
  `LC_ALL=C.UTF-8 LANG=C.UTF-8 .venv/bin/python -m pytest -q -k "seqclararange"`
- Run a specific test node:  
  `LC_ALL=C.UTF-8 LANG=C.UTF-8 .venv/bin/python -m pytest -q tests/test_seq_based.py::test_seqclararange_smoke`
- Build wheel and sdist (PEP 517):  
  `.venv/bin/python -m build`
- Run lint:  
  `.venv/bin/python -m ruff check .`
- Run format check:  
  `.venv/bin/python -m black --check .`

Optional extras for specific features (the authoritative list is
`[project.optional-dependencies]` in `pyproject.toml`):

- Plotting (`plot`): `matplotlib`, plus `seaborn` for styling and `mplcursors`
  for `plot_clustassoc` tooltips.
- Progress bars (`progress`): `tqdm`.
- Native thread limits (`threads`): `threadpoolctl`.
- Numba backend tier (`numba`): `numba`.

`scikit-learn` is **not** a dependency of this project. The ARI stability metric
is computed natively from a weighted contingency table in
`src/weightedcluster/_seqclara_runtime.py` (`stability_metrics`); no module
imports scikit-learn.

## Scientific Governance

- `docs/maintainers/SCIENTIFIC_GOVERNANCE.md` controls research protocols,
  simulation evidence, diagnostic classification, and scientific phase
  decisions.
- Distinguish scientific invalidity, qualified evidence, procedural defects,
  and informational findings before applying a gate. A historical prompt is
  not current authority after it is completed or explicitly superseded.
- Preserve raw evidence, not obsolete procedural vetoes. Seeds, datasets,
  responses, fits, denominators, attempt histories, and the producing revision
  remain immutable evidence; tests, documentation, orchestration, and decision
  policy must be corrected transparently when defective.
- Do not convert optional framework metadata, warning text, graph availability,
  source-location drift, or lifecycle-specific artifact checks into scientific
  failure without a demonstrated causal path to the estimand or inference.
- Active project status and Bayesian RARCAT authorization are recorded in
  `docs/maintainers/BAYESIAN_RARCAT_B1_GOVERNANCE_CORRECTION.md` and the active
  continuation document, not in historical terminal dispositions.

## Project Family (see ECOSYSTEM.md)

This project is the clustering layer of a four-project family. It **builds on
traminex**; py_seqplot provides the dedicated grammar-of-graphics
visualization layer, and flexscanx is the spatial-scan project:

- `PROJECT_SEQANALYSIS_and_SPATSCAN/PROJECT_TRAMINEX` — `traminex`, the foundation.
- `PROJECT_SEQANALYSIS_and_SPATSCAN/PROJECT_PY_WEIGHTEDCLUSTER` — this repository.
- `PROJECT_SEQANALYSIS_and_SPATSCAN/PROJECT_PY_SEQPLOT` — `py_seqplot`, visualization.
- `PROJECT_SEQANALYSIS_and_SPATSCAN/PROJECT_FLEXSCANX` — `flexscanx`, spatial scans.

pyweightedcluster owns clustering of sequences and cluster quality. It does
**not** own sequence distances/costs — traminex does, with R-golden parity
tests. Before implementing or extending any sequence distance, cost matrix, or
sequence tree here, check traminex first and prefer it; record any deliberate
local copy in `DUPLICATES.md`. `ECOSYSTEM.md` in this folder points to the full
charter (one master copy, in `../PROJECT_TRAMINEX/ECOSYSTEM.md`); `DUPLICATES.md`
holds the convergence plan.

## Supervisor notes

Findings from the family's supervision repository
(`PROJECT_SEQANALYSIS_and_SPATSCAN`, the four projects' parent folder) are recorded, dated and
with their evidence, in `docs/SUPERVISOR_NOTES.md` (kept in the development repository, not published).
Read an open entry before working on the area it names, and record its
resolution in that entry: the date, the commit and the evidence. The supervision
repository writes only that file here; changes to this project's code, tests
and documents are made in a session started in this project (owner decision of
2026-10-26).
Consult the affected entry before working in that area.

# Architecture

**Project:** weightedcluster (Python)
**Versioning:** see `CHANGELOG.md` (version single-sourced from `pyproject.toml`).

## Design Goals

- Preserve R `WeightedCluster` naming conventions where practical.
- Provide Python-native execution with reproducible outputs.
- Report operation-level backend ownership and selection for portability.
- Expose dictionary-based return structures aligned with R-style object fields.

## High-Level Layers

1. Public API layer
- `src/weightedcluster/__init__.py`
- `src/weightedcluster/bayesian.py`
- Curates exported functions and plotting helpers.

2. Algorithm layer
- `kmedoids.py`, `cmpcluster.py`, `_cmpcluster_algorithms.py`,
  `_cmpcluster_controls.py`, `_cmpcluster_result.py`, `quality.py`, `fuzzy.py`,
  `seqclararange.py`, `boot.py`, `_boot_inputs.py`, `_boot_controls.py`,
  `_boot_result.py`, `seqnull.py`, `proptree.py`, `clustassoc.py`

2b. Result schema layer
- `schemas.py`
- Centralizes typed result contracts (`TypedDict`) and table coercion/validation utilities for range-style objects.

3. Distance layer
- `src/weightedcluster/_seqdist.py`
- A thin shim: HAM/OM/LCS/DHD/OMSpell are delegated to the explicit public
  `traminex.distances` submodule boundary (`pairwise` and
  `pairwise_between_sets`). These functions intentionally remain outside
  traminex's root manifest. This package computes no sequence distances itself.

4. Backend layer (clustering/quality kernels only — distances are traminex's)
- The canonical [README backend model](../README.md#backend-model) describes the
  operation-level contract; there is no package-wide compiled/Numba/Python
  selection order.
- Public medoid workflows select compiled then pure Python, never Numba.
- Public cluster quality selects compiled, Numba, then pure Python at call time.
- `wcCmpCluster` routes all eight `hclust` methods, weighted and unweighted,
  through `weightedcluster._fastcluster_linkage`, a Python reproduction of
  **`fastcluster::hclust`** — the implementation
  `orig_R_pkg/WeightedCluster/NAMESPACE` binds `hclust` to and therefore the one
  R's own `wcCmpCluster` runs. SciPy serves only the optional
  coordinate-embedding route, which clusters embedded coordinates rather than a
  precomputed dissimilarity matrix. Hierarchy consumers cut by merge count, so
  valid centroid and median inversions remain usable.
- `wcSilhouetteObs` is Python-only. The compiled, Numba, and pure-Python
  weighted-silhouette kernels are private and do not back that public API.
- `seqnull` selection depends on its model and transition shape: selected
  state-independent and Markov/static user-position paths can use Numba, while
  spell and time-varying user-position paths use Python. It has no compiled
  implementation.
- `_wc_core` exports only the production-consumed `weighted_medoids` and
  `cluster_quality` kernels; it owns no sequence-distance symbol.

5. Modular Bayesian RARCAT layer
- `bayesian.py` owns the lightweight public frozen result identity and lazy
  facade. Base import does not initialize PyMC, ArviZ, xarray, or PyTensor.
- `_bayesian_rarcat_design.py` owns the live formula/design/label helpers
  consumed by the conditional workflow. Patsy is therefore a Bayesian-extra
  runtime dependency, not a base dependency.
- `_bayesian_rarcat_partition.py` learns predictor-independent empirical
  partitions from dissimilarities and reference labels.
- `_bayesian_rarcat_conditional.py` fits one categorical-softmax posterior per
  exact distinct partition and retains draw-wise effects and diagnostics.
- `_bayesian_rarcat_mixture.py` performs deterministic empirical-mass
  integration and within/between/total uncertainty decomposition.
- `_bayesian_rarcat_storage.py` owns the unexported private schema-v1 bundle.
  Its SHA-256 inventory detects corruption but is not authentication.
- `tests/_historical_rarcat_solver.py` holds the test-only historical
  Statsmodels GLM/REML/Fisher proof. `_rarcat_solver.py` is not a production
  module or a distribution artifact.

This is a modular/cut posterior rather than one conventional joint generative
likelihood. B2 cannot see predictors, B3 cannot alter partition masses, and B4
cannot refit either phase. Public `parallel` affects only deterministic B4
post-processing.

## Two Main Workflows

### Distance-first workflow
Use when you already have a distance matrix.

Typical functions:
- `wcKMedoids`
- `wcKMedRange`
- `wcCmpCluster`
- `wcPropertyClustering`
- `wcClusterQuality`

### Sequence-first workflow
Use when you start from sequence arrays `(N, T)`.

Typical functions:
- `seqclararange`
- `seqclararange_family`
- `bootclustrange(object, seqdata, ...)`
- `seqnull`
- `seqnullcqi`

### Modular Bayesian association workflow
Use `bayes_rarcat` when the scientific question is how categorical covariate
associations change across an uncertain fixed-`K` sequence typology. It is not
an R `rarcat` compatibility path. See
[`bayesian_rarcat.qmd`](vignettes/bayesian_rarcat.qmd).

## Data/Label Conventions

- Sequence arrays: integer-coded NumPy arrays.
- Cluster labels exposed to users are 1-based for R compatibility.
  `wcKMedoids` uses R's more specific representation: every value is the
  one-based observation position of that row's medoid.
- Most complex outputs are dictionaries with stable keys (`kvals`, `clustering`, `stats`, etc.).

## Important Object Contracts

### `seqclararange`
- Produces clustering solutions over `kvals`.
- Returns R-style core `stats` plus extended `quality_stats`.
- Fuzzy and PCMdd runs enrich `quality_stats` with fuzzy-only diagnostics (`FPC`, `FPE`, `XBf`) in addition to crisp CQI values.
- Stores per-K run diagnostics in `clara`.

### `bootclustrange`
- Bootstrap validation on a clustering object + sequence matrix.
- Supports `sampling` strategies: `clustering`, `medoids`, `strata`, `random`.
- Returns bootstrap means (`stats`/`meant`), standard errors (`stderr`), and raw
  samples (`boot`) in the R-primary mapping; the full-data baseline and
  Python-only replay diagnostics are explicit `.additions` metadata.
- Keeps public orchestration, distance adaptation, the process worker wrapper,
  task dispatch, and runtime cleanup in `boot.py`; private input coercion,
  sampling/replay controls, and result assembly live in `_boot_inputs.py`,
  `_boot_controls.py`, and `_boot_result.py`, respectively.

### `seqnullcqi`
- Generates null datasets, reclusters each replicate, and compares observed vs null quality.
- Returns R-style null matrices: `stats[stat] -> (R, n_k)`.
- Includes R-named metadata keys (`hclust.method`, `seqdist.args`).

## Module Map

(Completed to cover all 43 production modules by the 2026-09-02 audit,
finding WC-090210.)

- `aggregate.py`: case aggregation utilities
- `_backend_contract.py`: declarative capability registry behind
  `get_backend_info()['capabilities']` — implementations, public dispatch,
  selection timing, and the recorded non-selection decisions
- `bayesian.py`: public lazy Bayesian RARCAT facade and stable frozen result
- `boot.py`: public `bootclustrange` orchestration, Traminex distance adapter,
  process worker wrapper, shared task dispatch, and runtime-context cleanup
- `_boot_inputs.py`: private sequence/clustering coercion and initial validation
- `_boot_controls.py`: private sampling, replay validation, and worker-context preparation
- `_boot_result.py`: private bootstrap result/statistic assembly
- `_boot_runtime.py`: private bootstrap worker context (the init/clear seam
  shared by the serial path and spawned workers) and per-replicate execution
- `_constants.py`: shared tolerances, iteration defaults, and label-base
  constants
- `clustassoc.py`: cluster-covariate association; discrepancy terms delegate
  to `traminex.dissmfacw`, while local BIC adapters reproduce R `lm` and
  factor-only `nnet::multinom` likelihood contracts. Separated multinomial fits
  use the same `abstol` stopping contract with a documented cross-optimizer
  numerical tolerance.
- `cmpcluster.py`: public `wcCmpCluster` orchestration and presentation,
  linkage kernels (including the kernel consumed by `seqnull.py`), PAM
  helpers, and per-method dispatch
- `_cmpcluster_algorithms.py`: private classical-MDS helper and the
  `cluster::diana`-compatible splinter/banner/merge-order port
- `_cmpcluster_controls.py`: private method normalization, aliases, weighted-
  method restrictions, and deterministic replay-start validation
- `_cmpcluster_result.py`: private comparison-family parameter and `allstats`
  result assembly
- `fuzzy.py`: fuzzy/noise/possibilistic clustering variants
- `_fallback.py`: pure-Python medoid and cluster-quality kernels — the
  validated direct kernels with exact range-risk statistics, the
  R-permissive public adapter, and the exact-dyadic objective helpers that
  `kmedoids.py` imports
- `kmedoids.py`: PAM-like weighted k-medoids
- `_metrics.py`: sequence-distance metric-name normalization shared by the
  delegating adapters
- `_numba.py`: optional Numba tier — the JIT cluster-quality kernel and
  `seqnull` generators plus their validated plain-Python wrappers and
  range-risk dispatch
- `_parallel.py`: the shared executor seam (`run_task_batch`):
  spawn-context processes or threads, submission-index result reassembly,
  and optional progress reporting
- `plot.py`: plotting and reporting helpers
- `proptree.py`: WeightedCluster-facing property-tree adapters over
  `traminex.disstree` and `traminex.seqtree`. The adapter preserves the public
  dictionary, cutting, labeling, pruning, and plotting contracts; it contains
  no local split, discrepancy, permutation, or tree-growth mathematics.
- `quality.py`: quality indices and silhouette tools
- `_quality_contract.py`: shared kernel input contracts (distance, label,
  and weight validation, range-risk predicates, relative scaling), the
  ASW-through-`wcSilhouetteObs` bridge the Numba and validated pure tiers
  use, and the observation-silhouette aggregation the permissive kernel's
  range-risk route shares with them
- `range.py`: clustrange assembly helpers
- `seqclararange.py`: CLARA-based sequence clustering
- `_seqclara_runtime.py`: private CLARA iteration runtime (aggregation,
  per-K execution, stability metrics)
- `seqclustname.py`: public `seqclustname` medoid-based cluster naming
- `seqnull.py`: null-model generation and null-CQI workflows
- `schemas.py`: canonical result schemas and validators for clustrange/range/bootstrap outputs
- `seqtree.py`: tree conversion helpers
- `_threads.py`: `limit_threads` context manager for native threadpool and
  environment limits (test- and benchmark-facing; no production caller)
- `_traminex_compatibility.py`: the stdlib-only fail-closed import gate that
  verifies the installed traminex provider against the packaged 79-file
  Python-source commitment before any delegated import
- `_utils.py`: small shared helpers (seed derivation, weighted medoid
  lookup, and the test-facing traminex delegation witness `pairwise_sum`)
- `_validation.py`: distance-matrix and argument validation plus
  merge-count linkage cutting

## Runtime Dependencies

Required runtime:
- `numpy`
- `pandas` (for some tabular outputs)
- `scipy` (hierarchical clustering paths and distance-matrix helpers)

Optional runtime extras:
- `matplotlib` / `seaborn` (plotting)
- `tqdm` (progress bars)
- `numba` (optional acceleration for the operation-specific paths named above)
- `bayesian`: PyMC `>=6.0.1,<6.1.0`, ArviZ `>=1.2.0,<1.3.0`, and xarray
  `>=2026.4.0,<2026.5.0`, plus h5netcdf `>=1.8.1,<1.9.0` for the required
  private schema-v1 NetCDF round-trip and its HDF5 implementation h5py
  `>=3.16.0,<3.17.0`, and Patsy `>=1.0.2` for the live conditional formula
  design boundary. Statsmodels is not a production or Bayesian-extra
  requirement; it is retained only in `dev` for executable historical proof.
  Bambi/formulae are development-only construction cross-checks; PyTensor is
  transitive through PyMC.

The accepted local Phase 4 run used NumPy 2.3.5 with Numba 0.65.1. Those
versions are evidence that the optional extra was available on that host, not a
new runtime ceiling or a requirement to install Numba.

## Performance Notes

- `seqclararange` now reuses `wcClusterQuality` outputs to avoid duplicate internal-index recomputation in final quality assembly.
- Fallback `wcClusterQuality` computes `HC` from pairwise moments directly, avoiding temporary pair-array allocation.
- The committed Phase 4 benchmark is local Ubuntu x86_64 evidence only, not hosted-wheel or aarch64 validation.
  It reports medians/IQRs and cold/warm
  separation for one host without a speed guarantee or CI threshold; see the
  maintainer report (`docs/maintainers/PHASE4_BACKEND_REPORT.md`).

## Documentation Boundaries

Canonical documentation lives in:
- `README.md`
- `docs/README.md`
- `docs/API_INDEX.md`
- `docs/WORKFLOW_GUIDE.md`
- `docs/R_MIGRATION_GUIDE.md`
- `docs/vignettes/`

Maintainer-only notes live in:
- `docs/maintainers/`

Historical reports and milestone artifacts live under:
- `archive/audits/`
- `archive/history/`

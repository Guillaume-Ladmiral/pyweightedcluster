# weightedcluster (Python)

Python port of key APIs from the R package `WeightedCluster` (Studer), with R-oriented naming/return structures where practical.

> **Project family.** pyweightedcluster is the clustering layer of a
> **four-project family**: traminex provides sequence-analysis foundations,
> pyweightedcluster owns sequence clustering and cluster quality, py_seqplot
> provides grammar-of-graphics sequence visualization, and flexscanx owns
> spatial scans. pyweightedcluster builds on traminex for sequence distances,
> costs, discrepancy analysis, and faithful property-tree growth. See
> [ECOSYSTEM.md](ECOSYSTEM.md) for the charter, [DUPLICATES.md](DUPLICATES.md)
> for the duplicate inventory, and [MIGRATION_PLAN.md](MIGRATION_PLAN.md) for the
> convergence plan. Any AI assistant should read [AGENTS.md](AGENTS.md) first.

Release status:
- Current release: `0.1.1` — see [CHANGELOG.md](CHANGELOG.md) for the full version history
- The version is single-sourced from `pyproject.toml`; `weightedcluster.__version__` reads it
- Version strings like `0.2.x`, `0.3.x`, `0.4.x` in archived reports are disowned draft artifacts

Parity target status:

- WeightedCluster 2.0 is the canonical written and live target, with TraMineR
  2.2-14 as the exact companion oracle (moved from 2.2-13 on 2026-09-21).
- The canonical readable source is `orig_R_pkg/WeightedCluster`; the exact
  artifact is `orig_R_pkg/WeightedCluster_2.0.tar.gz` from
  `https://cran.r-project.org/src/contrib/WeightedCluster_2.0.tar.gz`, SHA-256
  `6a03fbbc090ac92498a410faab1aeb88d686ff7a81dcfdcb97edacda44f393cc`.
- WeightedCluster 1.8-1 remains historical evidence only: readable tree
  `orig_R_pkg/WeightedCluster-1.8-1` and exact artifact
  `docs/legacy/WeightedCluster_1.8-1.tar.gz`, SHA-256
  `daf9987cee5ac9e706a39e72f54a64e49e9fe91b0bc3cfb5e5a94d5076b2df5b`.
- Phase 2 is complete. For Phase 3, exact R `rarcat` parity has been replaced
  by an independently validated modular Bayesian redesign. The public
  `bayes_rarcat` facade retains predictor-independent partition uncertainty,
  one direct-PyMC categorical posterior per distinct partition, draw-wise
  continuous/categorical effects, exact empirical-mass integration, and
  within/between/total posterior variance in a frozen `BayesRARCATResult`.
  Scientific decisions follow the
  scientific-governance policy (`docs/maintainers/SCIENTIFIC_GOVERNANCE.md`),
  and the B1 evidence decision is recorded in the
  B1 governance correction (`docs/maintainers/BAYESIAN_RARCAT_B1_GOVERNANCE_CORRECTION.md`).
  The enforced `n <= 12` envelope has no valid in-envelope calibration
  evidence; the outcome-selected small-n study cannot support recovery,
  coverage, direction, or magnitude guidance. See the
  small-n evidence correction (`docs/maintainers/BAYESIAN_RARCAT_SMALL_N_EVIDENCE_CORRECTION_2026_08_27.md`).
  The executable R-parity ledger remains 18 strong, 0 partial,
  0 unverified, and 1 missing export. Complete package parity is not complete:
  R `rarcat` remains missing and `bayes_rarcat` is intentionally a different
  estimator, not a compatibility alias.
  **B5 complete; public `bayes_rarcat` available; R `rarcat` remains missing.**
  See the
  Bayesian RARCAT design (`docs/maintainers/BAYESIAN_RARCAT_REDESIGN.md`) and
  [migration vignette](docs/vignettes/bayesian_rarcat.qmd).
- The 2026-08-21 whole-codebase audit corrected four material parity claims:
  weighted CLARA now separates sampling probabilities from analysis weights;
  bootstrap range statistics replay the installed unguarded native R kernel;
  zero-denominator and zero-mass silhouette results retain R's non-finite and
  sentinel semantics; and the directly captured installed `ari.boxplot` first-K
  defect is deliberately corrected to per-K selection in Python. Export grades
  remain 18 strong, 0 partial, 0 unverified, and 1 missing. Presentation
  evidence is counted separately: 22 strong, 3 partial, 0 unverified, and 3
  missing S3 methods.

## What Is Implemented

### Core clustering
- `wcKMedoids`
- `wcKMedRange`
- `wcClusterQuality`
- `wcSilhouetteObs`
- `wcAggregateCases`
- `as_clustrange`

`wcKMedoids(method="PAM")` and `wcKMedRange(method="PAM")` implement
WeightedCluster 2.0's PAM algorithm for square dissimilarity matrices. As in
the R source, PAM remains unavailable for condensed `dist` input; `KMedoids`
and `PAMonce` support both representations.

### Sequence workflows
- `seqclararange`
- `seqclararange_family`
- `bootclustrange`
- `seqnull`
- `seqnullcqi`
- `seqclustname`

The five WeightedCluster 2.0 criteria accepted by `seqclararange` are
`distance`, `DB`, `XB`, `PBM`, and `AMS`; Python-only criteria add `PBC`, `HG`,
`HGSD`, `ASW`, `ASWw`, `CH`, `R2`, `CHsq`, `R2sq`, and `HC`. Selection
minimizes distance, DB, XB, and HC and maximizes PBM, AMS, PBC, HG, HGSD, ASW,
ASWw, CH, R2, CHsq, and R2sq. HC therefore follows the minimizing semantics of
`clustrange` and `clustrangefamily`.

Normalized aggregate weights govern CLARA sampling, while the criteria use the
original aggregate weights as analysis weights. This is a deliberate correction
to WeightedCluster 2.0's reuse of sampling probabilities inside scale-dependent
criteria.

**This applies to every input, not only weighted data.** R computes
`ac$probs <- ac$aggWeights / n` where `n` is the original case count
(`R/clara_clustrange.R:42,47`), and for unweighted input the aggregate weights
are duplicate counts that sum to `n`, so the rescaling factor is `n` there too.
Relative to R, this package reports `Avg dist`, `XB` and crisp `AMS` larger by a
factor of `n`, `PBM` smaller by a factor of `n^2`, and `DB` and fuzzy `AMS`
unchanged. Iteration *selection* is unaffected, because the transform is
monotone. Earlier wording scoped this to weighted data, which was wrong in the
direction that matters: a reader with unweighted data concluded the criteria
matched R's scale when they did not (finding F-08 of the 2026-08-24 audit).

### Method comparison and property-based workflows
- `wcCmpCluster`
- `wcFuzzyRange`
- `wcPropertyClustering`
- `seqpropclust`
- `clustassoc`

`wcCmpCluster` requires `2 <= maxcluster <= N` and validates that complete K
range before selecting DIANA, hierarchical, or PAM machinery. For
`beta.flexible`, every representable distance difference is ordered exactly;
only exactly equal candidates use deterministic pair ordering. The optional
`embed_then_ward=True` classical-MDS approximation uses a scale-relative
positive-eigenvalue threshold, so a common positive distance scale does not
collapse a nondegenerate embedding, even when squaring the physical units would
underflow or overflow. The real public Ward route normalizes coordinates for
linkage and restores its heights. `wcFuzzyRange` forwards the original
validated finite observation-weight vector unchanged to PAM, FCMdd, medoid,
XBf, silhouette, and AMS stages. Each stage uses local exact or relative
accumulators when binary64 products would lose range; no shared normalized
weight vector is materialized. Membership rows retain exact numerator/row-total
ratios through range-sensitive fuzzy powers. FPE uses every exact positive membership, and an
exact-zero entropy term contributes zero without evaluating `log(0)`. The
compiled, Numba, and pure `wcClusterQuality` kernels retain raw distance rank
keys and local positives, routing only squared-moment or pair-mass range risks
through exact/relative accumulators. This covers subnormal squares, finite
reduction overflow, high/low and fourth-order pair-mass aggregates, mixed
weight/distance exponent cancellation, and representable absolute CH/CHsq
magnitudes. Python-appended DB/XB are scale invariant while PBM retains
its documented distance-squared units.
Dynamic-range-safe DB/XB/XBf and public PAM/PAMonce/KMedoids arithmetic retains local positive
distance and fuzzy-coefficient exponents through aggregate and partial-subnormal
boundaries. PAM BUILD, swap/update, and run ranking use exact comparisons when
finite objectives leave binary64 range, and PBM makes
one final IEEE range decision after all factors are combined. SeqCLARA
criteria likewise keep membership/confidence and raw-weight factors
through DB/AMS, and retain crisp/fuzzy distance-functional components until
XB/PBM cancel any common subnormal unit. Exported
`wcSilhouetteObs` preserves its exact ordinary operation path; only range-risk
inputs use exact dyadic distance-weight products, signed sums, and a single
correctly rounded final ratio. ASW
keeps its absolute weighted-size correction and below-one guard; ASWw stays
common-weight-scale invariant. This deliberately
returns the representable mathematical ASW/ASWw where installed R 2.0 inherits
a wrong large-distance result. For
`wcPropertyClustering` and `seqpropclust`, the keyword-only execution controls
are `perm_n_jobs=None`, its alias `n_jobs=None`, and `progressbar=False`.
`None` means one serial worker. Supplying non-`None` values for both aliases
conflicts; a worker count other than `1` or `progressbar=True` raises
`NotImplementedError`. Property-tree parallelism and progress reporting are not
currently available.

### Modular Bayesian RARCAT

- `bayes_rarcat`
- `BayesRARCATResult`

This optional workflow uses a fixed four-chain PyMC profile and requires
explicit partition/conditional root seeds and effect controls. Its supported
initial envelope and interpretation are documented in the
[Bayesian RARCAT vignette](docs/vignettes/bayesian_rarcat.qmd).
The envelope is enforced for runtime feasibility, but no valid calibration
evidence exists inside it; see
`docs/maintainers/BAYESIAN_RARCAT_SMALL_N_EVIDENCE_CORRECTION_2026_08_27.md`.
Patsy is installed through the `bayesian` extra; Statsmodels is not a
production dependency. The supported Bayesian envelope includes `n <= 12`.

### Tree helpers and plotting
- `as_seqtree`
- `dtcut`, `dtlabels`, `dtprune`
- plot helpers exported from `weightedcluster`

For `fuzzyseqplot(type="i")`, each finite fuzzy case weight controls the
vertical row height, so unequal membership/case weights remain visible instead
of being drawn as equal rows. Requested weighted rendering fails before row
selection when the complete sidecar is nonfinite or negative. The existing
index keyword `weighted=False` explicitly selects and draws unit-weight rows;
`members_weighted=False` is different and retains R's membership-only weights.
Raw weights determine geometry whenever their cumulative edges are
representable. Exceptionally tiny positive weights that would collapse at the
`-0.5` origin are divided by one common display scale, preserving proportional
geometry within the representable axis range; otherwise rendering fails with a
stable error instead of erasing a positive row.
Distribution and other non-index plot types continue through public traminex
rendering.

## Install a release

pyweightedcluster is not on PyPI, and neither is traminex, which it needs.
Releases are published on GitHub with their packages and a `SHA256SUMS` file:
[pyweightedcluster](https://github.com/Guillaume-Ladmiral/pyweightedcluster/releases)
and [traminex](https://github.com/Guillaume-Ladmiral/traminex/releases). The
compiled wheel is built for Linux x86-64 and CPython 3.12. Install both packages
in one command, so that pip takes traminex from its release and never looks for
that name on PyPI; pip takes the other dependencies from PyPI:

```bash
# docs-audit: shell-setup reason="install the published release wheels"
python -m pip install \
  https://github.com/Guillaume-Ladmiral/traminex/releases/download/v0.2.3/traminex-0.2.3-py3-none-any.whl \
  https://github.com/Guillaume-Ladmiral/pyweightedcluster/releases/download/v0.1.1/pyweightedcluster-0.1.1-cp312-cp312-linux_x86_64.whl
```

This release works with traminex 0.2.3: importing `weightedcluster` checks that
the installed traminex Python files are exactly those of that release, and stops
otherwise (see the next section). On another platform or Python version, install
the source package `pyweightedcluster-0.1.1.tar.gz` from the same release instead
of the wheel; pip then compiles the C++ extension, which needs a C++ compiler.
The ICU library that traminex requires is described in traminex's README.

## Installation

Ubuntu 24.04 is the canonical development platform. The package requires Python 3.11 or newer.
The CI and wheel matrices currently test Python 3.11 and 3.12. The package depends on the local
sibling `traminex`, so install that project first:

```bash
# docs-audit: shell-setup reason="controlled editable-install commands"
~/miniforge3/envs/datasci/bin/python -m venv --system-site-packages .venv
PY=.venv/bin/python
$PY -m pip install --no-deps --no-build-isolation -e ../PROJECT_TRAMINEX
$PY -m pip install --no-build-isolation -e .
```

The metadata requirement `traminex>=0.2.0` is a necessary version floor but is
insufficient to identify the delegated behavior: traminex source has changed
while its version remained constant. Every `weightedcluster` import first verifies
the installed provider against the packaged 83-file Python-source commitment
for reviewed revision `2cbb3472736d0ccce32a291241983a12b8921122`, traminex 0.2.3 (re-pinned
2026-11-07 from `6e76440b…` and on 2026-11-08 to 0.2.1, 0.2.2, then 0.2.3, through the reviewed procedure). Missing,
extra, changed, or symlinked Python sources fail closed before delegated symbols
load. Maintainers must use the reviewed manifest update procedure in
`docs/maintainers/README.md` when approving a new sibling revision.
Verification is a point-in-time startup check: it detects source replacement
overlapping the check using file identities and a final reinventory, then caches
success for the process. It assumes the installed provider tree remains stable
afterward. Concurrent mutation after the final check is outside the guarantee;
restart the process to verify the tree again.

For the Bayesian RARCAT workflow, install the bounded optional extra after
traminex. Its PyMC stack requires the ICU runtime/development libraries used by
PyICU (`libicu-dev` on Ubuntu 24.04), so install that system package first:

```bash
# docs-audit: shell-setup reason="local Bayesian optional dependency install"
sudo apt-get update
sudo apt-get install -y libicu-dev
$PY -m pip install -e ".[bayesian]"
```

On other Linux distributions or macOS, install the ICU headers and runtime
with the platform package manager before installing the extra. `pip` cannot
install this native prerequisite; a missing ICU/PyICU runtime is reported at
the Bayesian call boundary.

The `strict` extra (`scikit-sparse`) is **test-time only**. No module under
`src/weightedcluster` imports it; its only consumers are the CHOLMOD-based
REML prototype helper in `tests/rarcat_reml_architecture.py` and that
helper's test. Installing it does not change any public behaviour.

Build wheel/sdist:

```bash
# docs-audit: shell-setup reason="controlled local PEP 517 build commands"
$PY -m pip install -U build
$PY -m build
```

Run the test suite under the C collation used by the live-R/ICU parity
contracts:

```bash
# docs-audit: shell-setup reason="canonical locale-pinned test command"
LC_ALL=C.UTF-8 LANG=C.UTF-8 $PY -m pytest -q
```

## Backend Model

There is no global backend ladder. Backend ownership and public selection are
defined per operation; importability of one module does not imply that every
public API selects it.

| Operation | Public selection | Contract |
|---|---|---|
| Weighted medoids | compiled -> pure Python | `wcKMedoids` and `wcKMedRange` use the import-time compiled-then-pure binding only for PAMonce with non-negative ordinary-range weights; the KMedoids and PAM methods and negative-weight PAMonce run module-local searches, and range-risk inputs select the exact pure kernel at call time. Fuzzy range clustering selects the same binding at call time, and CLARA's crisp branch reaches it through `wcKMedoids`. No Numba medoid kernel exists. |
| Cluster quality | compiled -> Numba -> pure Python | `wcClusterQuality` selects a callable implementation at call time. All three implement the ten shared R quality statistics; the publicly selected pure implementation is the R-permissive kernel, and the validated pure kernel is the Numba tier's range-risk escape target. On finite non-negative inputs whose distances or weights leave the binary64 range, the R-permissive kernel takes the same exact statistics and stable silhouette route as the other tiers, with one recorded exception: when every cluster's weighted size is below one it keeps R's unguarded native ASW instead of the wrapper-level NA that the compiled, Numba and validated kernels embed, because R's `clustrangeboot` reaches the native kernel directly (`clusterquality.R:166`) and this kernel replays that bootstrap path. The divergence is private: every public route applies the below-one guard, so `wcClusterQuality` returns NaN on all tiers. See finding F-19 in `docs/maintainers/TECHNICAL_DEBT.md`. It keeps R's float arithmetic on ordinary inputs and throughout the R-permissive domain (negative or non-finite weights, non-finite distances). |
| Observation silhouettes (`wcSilhouetteObs`) | Python only | `wcSilhouetteObs` is the single observation-level implementation: the compiled quality kernel computes its silhouettes inline, while the Numba and pure quality tiers obtain ASW/ASWw through `wcSilhouetteObs`. No private weighted-silhouette kernels exist; their removal is enforced by `tests/test_audit_20260821_runtime_ownership.py` (2026-09-02 audit, finding WC-090205 corrected this table). |
| Hierarchical comparison | one route for all eight methods | All eight `hclust` methods, weighted and unweighted, go through `weightedcluster._fastcluster_linkage`, which reproduces **`fastcluster::hclust`** — the implementation WeightedCluster's own `NAMESPACE` binds `hclust` to, and therefore the reference `wcCmpCluster` must match. No optional package is needed. Flexible-beta orders distinct distances exactly and applies deterministic ordering only to exact ties. SciPy serves only the optional coordinate-embedding route, whose classical-MDS cutoff is relative to the largest positive eigenvalue. Merge-count cutting preserves valid centroid and median inversions. |
| `seqnull` | model- and shape-conditional | State-independent and supported static Markov/user-position paths may select Numba; spell models and time-varying user-position models use Python. There is no compiled `seqnull` implementation. |
| Sequence distances | delegated to traminex | Low-level adapters call `traminex.distances.pairwise`/`pairwise_between_sets`; bootstrap and null workflows call public `traminex.seqdist`. Local validation rejects non-finite HAM/DHD costs only when an observed state pair can reach them, before provider dispatch. `_wc_core` exports no sequence-distance symbols. |

NumPy is a required runtime dependency. Numba is an optional extra and is
reported as unavailable when it is absent or cannot import; no NumPy downgrade
is required for the base package. The pure-Python implementations remain
available. `get_backend_info()` retains its three legacy top-level availability
strings for compatibility and adds the authoritative, JSON-safe
`capabilities` mapping with symbols, ownership, callability, selection timing,
and real public dispatch.

Check backend availability:

```python
# docs-audit: smoke id=readme-backend-info
from weightedcluster import get_backend_info

backend_info = get_backend_info()
print(backend_info["compiled_core"])
print(backend_info["numba"])
print(backend_info["capabilities"]["weighted_medoids"]["public_dispatch"])
print(backend_info["capabilities"]["sequence_distances"]["owner"])
```

The committed Phase 4 backend report (`docs/maintainers/PHASE4_BACKEND_REPORT.md`)
and its raw JSON are local Ubuntu x86_64 evidence only, not hosted-wheel or aarch64 validation.
They record direct-kernel medians and IQRs, correctness,
cold Numba startup, thread limits, and provenance for one host. They establish
no speed guarantee and are not a CI timing gate.

## Quick Example

```python
# docs-audit: smoke id=readme-sequence-bootstrap-smoke
import numpy as np
from weightedcluster import seqclararange, bootclustrange

X = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 0],
], dtype=int)

cr = seqclararange(
    X,
    R=5,
    sample_size=4,
    kvals=[2, 3],
    seqdist_args={"method": "HAM"},
    criteria=("distance",),
    random_state=42,
)

bc = bootclustrange(
    cr,
    X,
    R=20,
    sample_size=4,
    sampling="random",
    random_state=42,
)

print(bc["meant"].keys())
print(bc["stderr"].keys())
```

## Known Limitations

- `wcCmpCluster(maxcluster=..., method='ward')` selects exact `ward.D2` semantics
  and returns the canonical `ward.D2` key. Explicit names, aliases, and
  `method="all"` share the same stable, deduplicated presentation names.
  Precomputed weighted and unweighted input uses the R-compatible local update;
  input uniqueness cannot exclude intermediate Ward ties. Set
  `embed_then_ward=True` only for the optional classical-MDS approximation;
  its positive-spectrum cutoff is scale-relative. Flexible-beta uses exact
  distance comparison and deterministic ordering only for exact ties.
- `wcCmpCluster(maxcluster=..., method='diana')` ports `cluster::diana`'s
  distance-based splinter, banner, and merge-order rules; it is not the older
  recursive k-medoids approximation. `maxcluster=N` is valid, while
  `maxcluster>N` is rejected before any method dispatch so result and parameter
  K sets cannot diverge.
- Plotting parity covers semantic plot-driving arrays and options plus the
  supported presentation contracts recorded in the parity ledger. It does not
  claim pixel-identical rendering, R console padding, or interactive base-R
  device behavior. The clustrange, clustrangefamily, and bootclustrange plot
  rows are graded partial because their payload arrays are reconstructed from
  installed source/result semantics and corroborated by direct installed-S3
  render/error calls rather than captured directly. For `seqclararange` ARI
  boxplots, Python deliberately uses
  each K's selected iteration column; WeightedCluster 2.0 accidentally reuses
  the first K's column for every K. Matplotlib remains a lazy optional
  dependency, loaded only by rendering paths. `plot_clustassoc(tooltip=True)`
  uses optional `mplcursors`; if it is absent, the plot is returned with a
  `RuntimeWarning` that hover tooltips are disabled.

## Documentation Map

This section is the authoritative index of all repository documentation.

Current documentation:

- `docs/README.md` (documentation index for the `docs/` tree)
- `docs/API_INDEX.md` (canonical API map)
- `docs/ARCHITECTURE.md` (design and module layout)
- `docs/WORKFLOW_GUIDE.md` (how to choose workflows)
- `docs/R_MIGRATION_GUIDE.md` (R -> Python mapping)
- `docs/vignettes/README.md` (worked examples and tutorial index)
- `docs/vignettes/bayesian_rarcat.qmd` (Bayesian RARCAT migration and limits)
- `docs/legacy/README.md` (reference-only legacy artifacts from the upstream R package)
- `docs/maintainers/README.md` (maintainer-only notes). The maintainers' records under `docs/maintainers/`,
  the Bayesian B1 validation evidence and the tests that read them are kept in the development
  repository; they are not part of the published package.

Release and change records:

- `CHANGELOG.md` (full version history — the canonical change record)
- `RELEASE_NOTES_v0.0.1.md` (release notes for the initial `0.0.1` release)
- `dist/` (build artifacts, untracked)

Contributor and agent instructions:

- `AGENTS.md` (canonical interpreter policy and standard commands for all tooling/agents)
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`

Reference and historical material:

- `orig_R_pkg/README.md` (canonical WeightedCluster 2.0 source plus the
  preserved historical WeightedCluster 1.8-1 reference)
- `archive/README.md` (index of historical/non-authoritative reports, audits, and disowned drafts)

Audit aids:

- `OPEN_QUESTIONS_FOR_AUDIT.md` (dated audit questions, current resolutions,
  and preserved historical observations)

The graphify knowledge-graph output, a non-authoritative historical snapshot
generated on 2026-08-10, was removed on 2026-10-26 (family supervision plan,
owner decisions D2 and D3); it remains in git history.

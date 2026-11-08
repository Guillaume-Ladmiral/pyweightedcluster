# R to Python Migration Guide

Guide for users familiar with the R package `WeightedCluster` who are using the Python port.

**Versioning:** see `CHANGELOG.md` (version single-sourced from `pyproject.toml`).

## Compatibility Principle

The Python port keeps R-oriented naming/structures where practical so that the original R manual remains a strong reference.

pyweightedcluster delegates sequence foundations to traminex. The dependency
floor `traminex>=0.2.0` is insufficient to reproduce those semantics on its own
because behavior-bearing source changed without a version bump. Importing
`weightedcluster` therefore checks the provider against a packaged, reviewed
Python-source commitment before delegated symbols load. Missing, extra,
changed, or symlinked provider source fails closed. Maintainers approve a new
provider only through the reviewed manifest update procedure in
`docs/maintainers/README.md`; users must not bypass the check to force an
unreviewed same-version checkout.

The check is point-in-time. It detects overlapping pathname/source replacement
with before/after file identities and a final reinventory, caches only a
successful startup result, and assumes the installed provider remains stable
afterward. Concurrent or post-check mutation is outside the guarantee; restart
the process for a fresh verification.

## Main Differences

- R lists -> Python dictionaries
- R matrices/data.frames -> NumPy arrays / pandas DataFrames
- Dot-separated names in R (`as.clustrange`) -> snake case in Python (`as_clustrange`)
- Cluster labels are 1-based at API boundaries for R compatibility

## Function Mapping

### Directly aligned names
- `wcKMedoids`
- `wcKMedRange`
- `wcClusterQuality`
- `wcSilhouetteObs`
- `wcCmpCluster`
- `wcFuzzyRange`
- `wcPropertyClustering`
- `wcAggregateCases`
- `seqclararange`
- `bootclustrange`
- `seqnull`
- `seqnullcqi`
- `seqclustname`
- `fuzzyseqplot`

### Dot-to-underscore names
- `as.clustrange` (R) -> `as_clustrange` (Python)
- `as.seqtree` (R) -> `as_seqtree` (Python)

## WeightedCluster 2.0 Phase 1 translations

- `wcAggregateCases` preserves R's aggregation-key order. Its `aggIndex` and
  `disaggIndex` values are Python zero-based positions, explicitly equal to the
  corresponding one-based R values minus one.
- `seqclustname(seqdata, group, diss, weighted=True, perc=False)` requires the
  dissimilarity matrix and returns a pandas categorical so R factor levels and
  codes remain available.
- `as_seqtree(object, seqdata, diss, weighted=True, ncluster=None)` accepts a
  SciPy linkage matrix for both R `hclust` and `twins`/`agnes` representations,
  or an ordered pandas `Split2` ... `SplitK` predictor frame for the R default
  method. Node members, medoids, and split-variable indices are zero-based.
- `fuzzyseqplot(seqdata, group=None, membership_threashold=0, type="i",
  members_weighted=True, memb_exp=1, ...)` intentionally retains the upstream
  misspelling `membership.threashold`. A NumPy/pandas membership matrix or a
  mapping/object with a `membership` value is the Python fanny-equivalent
  input. Memberships are powered before inclusive thresholding, and semantic
  panels use public traminex plot primitives. Fuzzy index cells alone use a
  narrow adapter so finite case/membership weights produce R-direction
  variable row heights; sorting, sampling, labels, palettes, ticks, and legends
  remain traminex facilities. R's `main` is retained, and `with.legend` maps to
  `with_legend`. `members_weighted=False` retains membership-only R weights;
  the distinct index keyword `weighted=False` requests unit-weight selection
  and geometry. Requested weighted rendering rejects a nonfinite or negative
  complete sidecar before full, systematic, or random selection, matching R's
  fail-closed outcome rather than silently sampling uniformly. As in
  WeightedCluster 2.0, finite memberships are not clamped to `[0, 1]`;
  nonfinite powered weights and an empty retained set remain visible in the
  semantic preparation contract. Matplotlib is lazy and available through the
  `.[plot]` extra. Raw index heights remain unscaled unless an exceptionally
  tiny positive vector would collapse at the `-0.5` display origin; then one
  common display scale preserves proportional geometry within the representable
  axis range. Wider dynamic ranges and unrepresentable per-panel cumulative
  totals fail before figure creation instead of erasing a positive row.

## WeightedCluster 2.0 Phase 2 translations

Phase 2 is complete. All 18 inherited exports have strong evidence. Phase 3
now follows the approved
Bayesian RARCAT redesign (`docs/maintainers/BAYESIAN_RARCAT_REDESIGN.md`), not exact
R `rarcat` parity. Complete package parity is not complete: `rarcat` remains
missing. Public `bayes_rarcat` is available as an intentional Bayesian
replacement, not an R-parity mapping. Its bounded worked example is
[the Bayesian RARCAT vignette](vignettes/bayesian_rarcat.qmd).

- `wcKMedoids` accepts R's `method` and `debuglevel` options, including
  case-insensitive partial names and numeric method codes. Matching the
  installed 2.0 source, `PAM` runs for square matrices and raises the upstream
  not-implemented error only for condensed `dist` input. Its `clustering`
  vector contains one-based medoid observation positions, `info` preserves the
  exact three R names, and `print_kmedoids` provides the semantic
  `print.kmedoids` text. Python additionally exposes `random_state` and an
  explicit zero-based `replay_starts` plan for deterministic stochastic tests.
- `wcKMedRange`, `as_clustrange`, and `wcCmpCluster` preserve R range/family
  fields and rankings. Python exposes zero-based replay plans for stochastic
  starts and bootstrap samples; it does not equate independent R and NumPy
  seeds.
- `wcClusterQuality` preserves the leading ten R statistics and appends the
  documented Python indices. `wcSilhouetteObs` returns NumPy arrays rather than
  R vector/data-frame subsetting objects.
- `wcPropertyClustering`, `seqpropclust`, and `dtcut` preserve R-facing tree,
  property, pruning, and label semantics while delegating owned sequence and
  discrepancy mathematics to public traminex APIs. For `seqpropclust`,
  `properties`, `pmin_support`, `max_k`, and `with_missing` govern extraction.
  The former Python-only `feature_include_transitions` and
  `feature_ngrams_max` keywords were removed because they never affected
  runtime behavior; use `properties` and `max_k` instead.
- `seqclararange`, `seqnull`, `seqnullcqi`, and `bootclustrange` expose explicit
  replay inputs for exact cross-runtime comparison. Python stabilizes only the
  source-visible failure boundaries recorded in the parity ledger.
- The five WeightedCluster 2.0 criteria accepted by `seqclararange` are
  `distance`, `DB`, `XB`, `PBM`, and `AMS`; Python-only criteria add `PBC`,
  `HG`, `HGSD`, `ASW`, `ASWw`, `CH`, `R2`, `CHsq`, `R2sq`, and `HC`.
  Selection minimizes distance, DB, XB, and HC and maximizes PBM, AMS, PBC, HG,
  HGSD, ASW, ASWw, CH, R2, CHsq, and R2sq. HC therefore follows the minimizing
  semantics of `clustrange` and `clustrangefamily`.
- For `seqnullcqi`, `hclust_method` is retained in result metadata. When
  `kmedoid=False`, null-distance reclustering is fixed to Ward.D for
  WeightedCluster 2.0 compatibility; `kmedoid=True` selects the medoid path
  instead.
- `clustassoc(clustrange, diss, covar, *, weights=None)` returns the exact
  `Unaccounted`, `Remaining`, `BIC`, and `numcluster` table through a typed
  dict-compatible result. Regression diagnostics and `BICgain` remain
  accessible off-table through `.additions`. `plot_clustassoc` accepts the R
  plot types and preserves
  WeightedCluster 2.0's legacy blank y-label lookup for the documented
  `Unaccounted` and `Remaining` names.

## RARCAT migration status

R `rarcat()` has no compatibility-equivalent Python function. Do not translate
it to an existing regression helper or the public Bayesian function while
claiming parity. `tests/_historical_rarcat_solver.py` is retired executable
test evidence for the abandoned exact-R solver path, not a production helper
or parity implementation.

Python `bayes_rarcat` is a deliberately different estimator. It uses
Bayesian-bootstrap weighted-k-medoids partitions, reference-label alignment,
conditional Bayesian categorical regression, draw-wise effects, and exact
empirical-mass posterior integration. It does not reproduce R's independent
one-vs-rest GLMs, `margins` standard errors, crossed-`lme4` pooling, result
fields, or S3 presentation. Its categorical probabilities sum to one and its
cluster effects for a common contrast sum to zero, unlike R's independent
binary fits.

Install the ICU development/runtime package first (`libicu-dev` on Ubuntu
24.04), then local traminex and `pyweightedcluster[bayesian]`. Callers
must provide separate partition and conditional root seeds, an ordered
categorical response whose category order matches first observation order,
exact index-aligned dissimilarities, continuous-predictor classification, and
counterfactual steps. The result reports posterior intervals/probabilities and
within/between/total uncertainty; these are not p-values or standard errors.
The full migration, diagnostics, fixed sampler profile, supported envelope,
and limitations are in
[the Bayesian RARCAT vignette](vignettes/bayesian_rarcat.qmd).

## Data Expectations

- Sequence data should be integer-coded arrays shaped `(N, T)`.

### The distance-matrix contract

`wcClusterQuality`, `wcSilhouetteObs`, `as_clustrange` and `wcKMedoids` share
one contract. Earlier wording here said only that distance matrices "should be"
square symmetric arrays, which is advice rather than a contract and described
none of the four behaviours; that gap is finding F-07 of the 2026-08-24
whole-codebase audit.

1. **Shape.** An `(N, N)` square array or a condensed distance vector. Anything
   else raises `ValueError`.
2. **Only the lower triangle is data**, exactly as R's `as.dist` reads a matrix.
   A square input is reduced to it: the upper triangle and the diagonal are
   ignored, not validated. When the ignored part disagrees with the lower
   triangle beyond tolerance, Python emits a `UserWarning` saying so. **R emits
   no such warning** — this is a deliberate addition, because silently computing
   on half of a caller's matrix is the failure it exists to make visible.

   **If you pass R a bare matrix rather than a `dist`, R does not do this, and
   Python deliberately does not follow it.** R is not self-consistent here.
   Measured on R 4.6.1 with WeightedCluster 2.0: given a *non-symmetric* square
   matrix, R's silhouette loop runs over every `j != i` and so reads **both**
   triangles, while its ordered-pair statistics iterate `j > i` and read one.
   The result is that R's own answer changes depending on whether you hand it
   `D` or `as.dist(D)`:

   | R call on a non-symmetric matrix | differs from the same call on `as.dist(D)` |
   |---|---|
   | `wcClusterQuality` | `ASW` and `ASWw` only; the other eight statistics agree |
   | `as.clustrange` | `ASW` and `ASWw` only |
   | `wcSilhouetteObs` | the whole vector |
   | `wcKMedoids` | the partition |

   Python reproduces R's `as.dist` reading in every one of those cases. So a
   script that passed a **symmetric** matrix, or any `dist` object — including
   everything `TraMineR::seqdist` produces — ports with identical numbers. A
   script that passed a **non-symmetric matrix** to `wcSilhouetteObs` or
   `wcKMedoids`, or read `ASW`/`ASWw` from `wcClusterQuality` on one, will get
   R's `as.dist` answer from Python rather than R's matrix answer, and a
   `UserWarning` telling you which half was used. This divergence is deliberate:
   it is the only reading under which the four entry points agree with each
   other. It is pinned by
   `tests/test_r_parity_distance_matrix_contract.py::test_python_follows_rs_dist_reading_where_rs_matrix_path_disagrees`.
3. **Non-finite and negative entries are accepted** and propagate R's behaviour.
   `wcClusterQuality` returns the same values R returns, including `NaN` where R
   produces `NaN`.
4. **One deliberate narrowing.** `wcKMedoids`'s compiled medoid search requires
   finite, non-negative distances and raises otherwise, where R's PAM returns a
   partition. The refusal is the Python kernels' limitation, stated here rather
   than left to be discovered.

Before this contract existed the four entry points disagreed with each other:
three raised on input R computes from, and `wcSilhouetteObs` accepted it and
returned different numbers from R with no error and no warning (finding F-06).

## Example: R-style Sequence Workflow in Python

```python
# docs-audit: smoke id=r-migration-workflow-smoke
import numpy as np
from weightedcluster import seqclararange, bootclustrange, seqnullcqi

X = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 0],
], dtype=int)

cr = seqclararange(
    X,
    R=10,
    sample_size=4,
    kvals=[2, 3],
    seqdist_args={"method": "HAM"},
    criteria=("distance",),
    random_state=42,
)

bc = bootclustrange(cr, X, R=20, sample_size=4, sampling="random", random_state=42)

nc = seqnullcqi(X, cr, R=20, model="combined", seqdist_args={"method": "HAM"}, random_state=42)
```

For tiny toy datasets like this 4-row example, use `sampling="random"` with
`bootclustrange()`. Cluster-stratified sampling (`sampling="clustering"`) is intended for
larger samples where each stratum can contribute enough observations.

`bootclustrange()` also always recomputes the baseline and bootstrap distances from `seqdata`;
it does not consume a hidden precomputed `object["diss"]` payload.

## Key Output Conventions

### `seqclararange`
- `kvals`
- `clustering` with keys like `cluster2`, `cluster3`
- `stats` (R-style core columns)
- `quality_stats` (extended diagnostics; fuzzy and PCMdd runs also expose `FPC`, `FPE`, and `XBf`)
- `clara` (per-K details)

### `bootclustrange`
- `stats` and `meant` are bootstrap means
- `stderr` is bootstrap standard error
- `result.additions["baseline_stats"]` stores full-data quality stats; Python-only
  controls and replay diagnostics use the same explicit metadata accessor

### `seqnullcqi`
- `stats[stat]` is an `(R, n_k)` null-distribution matrix
- R-style metadata keys included: `hclust.method`, `seqdist.args`
- `hclust_method` is result metadata, not a linkage selector: with
  `kmedoid=False`, reclustering is fixed to Ward.D for WeightedCluster 2.0
  compatibility. It runs through the quadratic-time nearest-neighbour-chain
  port of `fastcluster::hclust` for every bootstrap replicate and warns for
  large `N² × R` workloads. `kmedoid=True` selects a scientifically different
  medoid path; it is not a performance-equivalent replacement.

## Implementation Notes vs R Package

These are implementation differences to keep in mind:
- Sequence-distance mathematics remains delegated to traminex through its
  low-level pairwise APIs and high-level `seqdist` API. HAM/DHD substitution
  costs are checked at this adapter boundary: non-finite entries reachable by
  observed state pairs fail before distance/clustering work, while non-finite
  entries for unobserved states remain permitted.
- `wcPropertyClustering` and the tree stage of `seqpropclust` consume
  traminex's TraMineR-compatible `disstree`/`seqtree` implementations. Their
  pyweightedcluster adapter preserves the WeightedCluster-facing structure,
  split schedule, cutting, labels, and pruning without duplicating tree-growth
  mathematics. All nine sequence-property families are exact live-R contracts
  and delegate sequence statistics and event mining to public traminex APIs.
  The legacy random-sampling tree mode composes public tree and discrepancy
  APIs because traminex's public tree signature does not expose that
  WeightedCluster option.
  Both Python entry points expose keyword-only `perm_n_jobs=None`,
  `n_jobs=None`, and `progressbar=False`. The two job names are aliases: a
  non-`None` value for both conflicts, while either `None` resolves to one
  serial worker. Only worker count `1` is available; any other integer and
  `progressbar=True` raise `NotImplementedError` before property extraction or
  tree growth. Invalid types fail eagerly. Parallel permutation and progress
  reporting are not silently ignored.
- **`clustassoc` gives NaN where a clustering is collinear with the covariate;
  WeightedCluster gives a finite value.** The covariate term then adds no rank
  to the design, and R's `dissmfacw` returns a finite value of an arbitrary QR
  direction, which R's `clustassoc` reports (0.0357 and 0.03125 on the test's
  six-object example). traminex's `dissmfacw` gives NaN for a term that adds no
  rank (`3a3a5149`, a `DIVERGENCE` row of its ledger), and this port reports it:
  the number is an artefact, and NaN says so. A deliberate departure, chosen on
  2026-11-07 (option (a) of the family's open item); other rows are R's.
- `clustassoc` matches WeightedCluster 2.0 for weighted/unweighted numeric and
  categorical covariates: discrepancy terms come from `traminex.dissmfacw`, and
  BIC follows R's `lm` and factor-only `nnet::multinom` contracts. Categorical
  BIC is the converged likelihood: exact in closed form, and where a cluster has
  an empty covariate category the BFGS fit reaches the same limit (within 1.4e-11
  on 19 random cases). R's `multinom` stops earlier, at its default `reltol`, so
  its BIC is higher by up to 3.2e-6 without an empty category and 2.4e-3 with one
  (60 random cases, 2026-11-07; with `reltol = 1e-15`, 4.5e-13 and 2.7e-6); it is
  never lower. This corrects the earlier `2e-4` and `1e-7` bounds, which held for
  the fixtures only (2026-09-21 targeted audit, L-5); a test pins the spread on a
  random domain. A fully separated fit stops when the objective crosses nnet's
  `abstol=1e-4`, and SciPy and nnet can cross it on different BFGS steps, so that
  stopping value is pinned within `2e-4`.
- `wcCmpCluster(maxcluster=..., method="ward")` selects exact `ward.D2`
  semantics and returns the canonical `ward.D2` key. Explicit spellings,
  aliases, and `method="all"` share stable, deduplicated presentation names.
  All eight `hclust` methods, weighted and unweighted, reproduce
  **`fastcluster::hclust`** — the implementation WeightedCluster's own
  `NAMESPACE` binds `hclust` to — including its equal-distance merge rule, so no
  optional package is needed and no route depends on one being installed. Set
  `embed_then_ward=True` only for the optional classical-MDS approximation.
- **`seqclararange`'s R-supported fuzzy/noise branches omit R's second `fanny`
  start.** The installed R public method list is exactly `"crisp"`, `"fuzzy"`,
  `"representativeness"`, and `"noise"`; its second-fanny branch is guarded by
  `method %in% c("fuzzy", "noise")` (`R/clara_clustrange.R:15,107-118`). Both
  languages seed those branches from the same Ward hierarchy on the subsample
  — `fastcluster::hclust(as.dist(diss), method="ward.D",
  members=aggWeights)`, cut at `k` — and Python reproduces R's starting
  membership exactly. R then obtains a second membership from
  `cluster::fanny(diss, k, diss=TRUE, memb.exp=m, iniMem.p=memb)` and runs
  FCMdd (`"fuzzy"`) or NCdd (`"noise"`) from it, keeping the lower functional.
  By default Python runs only the Ward-seeded candidate because this project
  does not implement the `cluster` package's `fanny` algorithm.
  **Consequence:** only
  for the R-supported `"fuzzy"` and `"noise"` branches, Python's per-iteration
  functional is greater than or equal to R's, and selected medoids can differ
  where R's fanny-seeded candidate wins. Supplying
  `replay_initial_memberships` reproduces R's selection for those two branches.
  R's `fanny` also refuses `k` outside `{1, …, n/2 − 1}`, so those R branches
  can fail on subsamples Python accepts.

  `"hncdd"` (HNCdd) and `"pcmdd"` (PCMdd) are Python extensions; the installed
  R `seqclararange` method matcher does not expose either method, so there is no
  public-R second-fanny comparison and no fanny-derived functional bound for
  them. Their evidence is separate: their update equations track the vendored
  `wfcmdd` HNCdd/PCMdd branches (for HNCdd including its `m <- 1`, so the caller's
  `m` is accepted and not used; 2026-09-21 targeted audit M-3, fixed 2026-11-07, and
  compared with live `wfcmdd` from fuzzy initial memberships), and Python regression tests pin zero-distance,
  eta, objective, backend, and deterministic execution behavior without
  claiming public `seqclararange` live-R parity.
- **`wcCmpCluster` accepts `method="ward.D"` and `"ward.D2"`; R rejects them.**
  `R/clustrangefamily.R:5` lowercases `method` before testing membership of
  `hclustmethods`, whose entries keep a capital `D`, so an explicit
  `method="ward.D"` matches nothing, `hc` is never assigned, and R fails with
  `object 'hc' not found`. `method="all"` works, because it replaces `method`
  with the original-case vector after the `tolower` call. Python accepts both
  spellings and returns the result R would have produced. This is a deliberate
  correction of an upstream defect, in the same spirit as the `ari.boxplot`
  correction above; a script that fails in R will succeed here. Finding F-23 of
  the 2026-08-24 whole-codebase audit.
- **`wcKMedoids` rejects duplicated starting medoids; R does not terminate.**
  With `initialclust=c(1L,1L)` on a 4x4 matrix, R produced no output and was
  killed by a 90-second timeout, while the same call with distinct starts
  `c(1L,3L)` returned instantly. Python raises
  `ValueError: starting medoids must be unique`. The divergence is a safety
  improvement over an upstream non-termination defect rather than a behavioural
  choice, and it fires only when `np.unique(medoids).size != medoids.size`.
  Finding F-24. **Confirm this in a bounded subprocess if you reproduce it: the
  R call does not return.**
- `wcCmpCluster(maxcluster=..., method="diana")` ports
  `cluster::diana`'s native distance-based splinter, banner, and merge-order
  rules rather than using the older recursive k-medoids approximation.
- Plotting parity covers semantic plot-driving arrays and options plus supported
  presentation contracts. Named safe translations include the scalar/RHC
  `clustrange` paths and grouped-method vector-stat `clustrangefamily` path
  where installed WeightedCluster 2.0 fails. Python's `seqclararange` ARI
  boxplot also deliberately uses each K's selected iteration; installed R
  reuses the first K's column for every K. This does not claim pixel-identical
  rendering, R console padding, or interactive base-R device behavior.
  Matplotlib remains a lazy optional dependency and is available through `.[plot]`.

## Practical Recommendation

Use the R manual as conceptual reference, and use this package's docstrings/API docs for exact Python call signatures and return structures.

## Related Docs

- `docs/API_INDEX.md`
- `docs/WORKFLOW_GUIDE.md`
- `docs/ARCHITECTURE.md`
- `docs/vignettes/README.md`
- `docs/legacy/WeightedCluster.pdf` (original R reference manual)

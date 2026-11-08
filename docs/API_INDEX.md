# API Function Index

Canonical API index for `weightedcluster`.

This file reflects the actual public exports in `src/weightedcluster/__init__.py`.

WeightedCluster 2.0 Phase 2 is complete. All 18 inherited exports have strong
evidence. Phase 3 now follows the approved
Bayesian RARCAT redesign (`docs/maintainers/BAYESIAN_RARCAT_REDESIGN.md`), not exact
R `rarcat` parity. Complete package parity is not complete: `rarcat` remains
missing. The public `bayes_rarcat` replacement is available, but it is an
intentional Bayesian divergence rather than an inherited R-parity export. See
the [migration vignette](vignettes/bayesian_rarcat.qmd).

## Delegated Provider Compatibility

The runtime requirement `traminex>=0.2.0` is insufficient by itself because
different behavior-bearing source trees have reported the same version.
Before binding any delegated provider symbol, package initialization verifies
the installed traminex version, symlink-free exact Python-file inventory, and
aggregate source digest against the packaged commitment. This internal guard is
also called explicitly by editable-install and wheel smoke workflows. A new
provider revision is accepted only through the reviewed manifest update
procedure documented in `docs/maintainers/README.md`.

This is a point-in-time startup contract, not a filesystem immutability claim.
The verifier compares open-file identities before and after reads, checks each
pathname again, and performs a final exact reinventory. It then caches success
for that process and assumes the installed provider tree stays stable.
Concurrent or later mutation after the final check requires a process restart
and fresh verification.

## Core Clustering

- `wcKMedoids(diss, k, weights=None, npass=1, initialclust=None, method="PAMonce", cluster_only=False, debuglevel=0, *, random_state=None, replay_starts=None)`
- `wcKMedRange(diss, kvals, weights=None, R=1, samplesize=None, npass=1, initialclust=None, method="PAMonce", debuglevel=0, *, random_state=None, replay_starts=None, replay_samples=None)`
- `wcCmpCluster(diss, *, weights=None, maxcluster, method="all", pam_combine=True, embed_then_ward=False, random_state=None, replay_starts=None)`

`wcCmpCluster` exposes one canonical method-key contract: aliases and explicit
case variants normalize to the same stable, deduplicated names used by
`method="all"` (including `ward.D` and `ward.D2`). Its canonical K range is
`2..maxcluster`; `maxcluster=N` is valid and `maxcluster>N` fails before any
DIANA, hierarchy, or PAM method dispatch. Flexible-beta compares every
representably distinct candidate distance exactly and uses deterministic pair
ordering only for exact ties. The optional `embed_then_ward=True` approximation
uses a scale-relative classical-MDS positive-eigenvalue threshold.
- `as_clustrange(object, diss, weights=None, R=1, samplesize=None, ncluster=20, *, labels=True, random_state=None, replay_samples=None)`
- `as_seqtree(object, seqdata, diss, weighted=True, ncluster=None)`

## Sequence Workflows

- `seqclararange(seqdata, ...)`
- `seqclararange_family(seqdata, ...)`
- `bootclustrange(object, seqdata, ...)`
- `seqnull(seqdata, ...)`
- `seqnullcqi(seqdata, clustrange, ...)`
- `seqclustname(seqdata, group, diss, weighted=True, perc=False)`

Shared execution controls:
- `seqclararange`, `seqclararange_family`, `bootclustrange`, and `seqnullcqi` share the same `parallel=True` and optional `n_jobs=` contract.
- In executable `.py` files, place calls with `parallel=True` inside an
  `if __name__ == "__main__":` guard so process-spawn platforms do not
  recursively execute the calling module. Notebook cells may call directly.

The five WeightedCluster 2.0 criteria accepted by `seqclararange` are
`distance`, `DB`, `XB`, `PBM`, and `AMS`; Python-only criteria add `PBC`, `HG`,
`HGSD`, `ASW`, `ASWw`, `CH`, `R2`, `CHsq`, `R2sq`, and `HC`. Selection
minimizes distance, DB, XB, and HC and maximizes PBM, AMS, PBC, HG, HGSD, ASW,
ASWw, CH, R2, CHsq, and R2sq. HC therefore follows the minimizing semantics of
`clustrange` and `clustrangefamily`.

For `seqnullcqi`, `hclust_method` is retained in result metadata. When
`kmedoid=False`, null-distance reclustering is fixed to Ward.D for
WeightedCluster 2.0 compatibility; `kmedoid=True` selects the medoid path
instead.

## Property-Based Workflows

- `wcPropertyClustering(diss, properties, maxcluster=None, *, perm_n_jobs=None, n_jobs=None, progressbar=False, **kwargs)`
- `seqpropclust(seqdata, diss, properties=..., ..., *, random_state=None, perm_n_jobs=None, n_jobs=None, progressbar=False, **kwargs)`
- `dtcut(st, k, labels=True)`
- `dtlabels(tree)`
- `dtprune(st, k, ...)`

The tree stage delegates to the faithful public `traminex.disstree` /
`traminex.seqtree` implementations. pyweightedcluster preserves its R-oriented
mapping and helper APIs but does not implement split selection or tree growth.
All nine `seqpropclust(..., prop_only=True)` property families are live-R
pinned: state, duration, spell age/duration, transition/pattern counts,
age-first transition/pattern, and complexity. Extraction delegates to public
traminex sequence-statistic and event-mining APIs. Tree contracts cover
attached and unit weights, R's documented permutation modes, size/depth
controls, pruning, exact node/index translation, and safe translations of
demonstrated R boundary failures.

`seqpropclust` extraction is governed by `properties`, `pmin_support`, `max_k`,
and `with_missing`. The former Python-only `feature_include_transitions` and
`feature_ngrams_max` parameters were removed because they never affected
runtime behavior; unknown keywords now fail explicitly.

For both property-tree entry points, `perm_n_jobs` and `n_jobs` are aliases and
default to `None`, which resolves to one serial worker. Non-`None` values for
both aliases conflict. Only `1` is supported; another integer raises
`NotImplementedError`. `progressbar=False` is the only available progress mode,
and `progressbar=True` raises `NotImplementedError`. Invalid types fail before
property extraction or delegated tree growth.

## Quality and Indices

- `wcClusterQuality(diss, clustering, ...)`
- `wcSilhouetteObs(diss, clustering, ...)`
- `compute_internal_indices(diss, labels)`
- `db_index(diss, labels)`
- `dunn_index(diss, labels)`
- `xb_index(diss, labels)`
- `pbm_index(diss, labels)`
- `wcFuzzyQuality(diss, membership, ...)`
- `wcFuzzyRange(D, kvals, ...)`
- `crispness(membership, ...)`

The medoid PBM value uses the published squared ratio
`((E1 / EK) * (Dk / k)) ** 2`; every exact positive `EK` is used directly and
zero/nonfinite dispersion is undefined. DB and XB retain raw local finite
distances until their linear or squared ratios form, while PBM restores its documented distance-squared units only
at the final IEEE boundary. The compiled, Numba, and pure ten-statistic quality
kernels preserve raw rank keys/local positives and use exact or relative
accumulators only when squared moments, reductions, or pair masses would lose
range, including fourth-order and mixed-exponent cancellation. Representable
CH/CHsq magnitudes retain their absolute case-mass
semantics. Normalized `crispness` at one cluster is undefined (`nan`),
matching installed R. Davies–Bouldin averages over every cluster, including
valid zero contributions. The original validated finite fuzzy case-weight
vector reaches every downstream stage unchanged; stage-local exact/relative
accumulators preserve its conceptual ratios without materializing one shared
normalized vector. Fuzzy partition entropy uses every exact positive normalized membership,
while exact zero contributes zero without evaluating its logarithm; fuzzy
medoid, XBf, silhouette, and AMS calculations are invariant to finite common
distance scaling. Crisp DB/XB, fuzzy medoids/XBf, PBM, and public
PAM/PAMonce/KMedoids objectives and BUILD selection retain local positive
distance, weight, or membership-power exponents until their comparisons,
aggregate reductions, or ratios form, including at minimum-positive binary64;
PAM swap/update and multi-start ranking use the same exact-objective predicate.
SeqCLARA criteria keep membership/confidence and raw-weight factors through
DB/AMS and keep crisp/fuzzy distance-functional components through final
XB/PBM cancellation.
Only mathematically exact-zero coefficients are
excluded. `wcSilhouetteObs` preserves the exact ordinary operation path and
uses exact dyadic distance-weight products, signed aggregation, and one final
ratio only when raw accumulation can lose range. This deliberately improves installed R 2.0's
wrong large-distance result. ASW retains its absolute weighted-size correction
and below-one guard, ASWw is invariant to a finite common weight scale, and the
lower-triangle contract is unchanged.
In `seqclararange`, the R-compatible
`stats["DB"]` remains the sequence-CLARA medoid criterion; the published
partition-level index is exposed separately as `quality_stats["DB"]`.

## Aggregation and Association

- `wcAggregateCases(x, weights=None, *, weighted=True)`
- `clustassoc(clustrange, diss, covar, *, weights=None)`

`clustassoc()` returns a typed, dict-compatible `ClustassocResult`. Iteration
and labeled table access expose the exact R columns `Unaccounted`, `Remaining`,
`BIC`, and `numcluster`; `r_class` and `row_names` retain R metadata. BIC audit
diagnostics and `BICgain` remain available through `.additions` without
changing the exact four-column table. Typed `copy()` preserves a separate copy
of that metadata, while `dict()` and JSON serialize only the R-primary mapping.
`plot_clustassoc` intentionally repairs WeightedCluster
2.0's empty y-axis labels for `Unaccounted` and `Remaining`; all four accepted
statistics now have explicit semantic labels.

## RARCAT status

There is no public compatibility `rarcat`. The public Bayesian estimator is
`weightedcluster.bayes_rarcat`:

```python
# docs-audit: display-only reason="exact public API signature"
bayes_rarcat(
    dissimilarity,
    *,
    formula: str,
    data,
    B: int,
    npass: int,
    partition_root_seed: int,
    conditional_root_seed: int,
    continuous_predictors,
    counterfactual_steps,
    user_prior=None,
    parallel: bool = False,
    n_jobs: int | None = None,
) -> BayesRARCATResult
```

`weightedcluster.BayesRARCATResult` and
`weightedcluster.bayesian.BayesRARCATResult` are the same frozen type. The
facade verifies the Bayesian runtime extra, takes observation identity only
from `data.index`, takes reference labels only from the single untransformed
ordered-categorical formula response, then calls the predictor-independent
partition posterior, conditional categorical posteriors, and exact mixture
integration once each. `parallel` affects only deterministic B4 extraction.
It must be a boolean; `n_jobs` must be `None` or a positive, non-boolean
integer. Invalid execution controls fail before partition construction and
conditional PyMC fitting, and direct private B4 integration enforces the same
shared contract.

The result retains `status`, `failures`, warnings, the complete partition and
conditional posteriors, dimensions/seeds/design/prior/software provenance,
empirical `masses`, weighted target supports, `coefficient_summaries`,
`effect_summaries`, deterministic `allocation`, and private schema name/version.
Its arrays are read-only. A failed conditional component remains at its exact
index and mass and returns a typed blocked result rather than a renormalized
mixture.

Private B4 directory persistence publishes complete bundles with Linux
`renameat2(RENAME_NOREPLACE)` on one filesystem, validates the published
bundle, and refuses a destination that appears in the final publication
window. A missing primitive and explicit unsupported-operation errnos fail
closed with `NotImplementedError`; operation-specific errors such as an invalid
rename topology remain `OSError`. No broader portable atomic no-replace
guarantee is claimed. After a failure, failure handling performs no pathname
mutation. If the recorded directory device/inode is observed at a
writer-managed pathname, that existing path is noted on the original exception
with an explicit concurrent-identity caveat. An absent, mismatched, or rebound
entry is left untouched. Production does not rename, call `unlink`, or call
`rmdir` on failure artifacts. This
fail-closed residue is intentional because Linux has no supported
unlink-by-open-handle operation for directory trees, so deletion cannot be made
safe against arbitrary same-UID pathname rebinding. See
`docs/maintainers/BAYESIAN_RARCAT_STORAGE_PUBLICATION_DESIGN_2026_08_27.md`.

The supported public envelope is `2 <= n <= 12`, `2 <= K <= 3` with `K <= n`,
`1 <= B <= 128`, `1 <= U <= 16`, `U / B <= 0.25`, and
`1 <= design_width <= 4`. Inputs outside it fail before sampling. The
envelope is a runtime-feasibility contract, not a calibrated operating regime:
no valid calibration evidence exists inside it. The estimator algebra was
checked separately, but the preserved outcome-selected small-n study cannot
support unconditional recovery, coverage, direction, or magnitude claims. See
`docs/maintainers/BAYESIAN_RARCAT_SMALL_N_EVIDENCE_CORRECTION_2026_08_27.md`.
The estimator is intentionally not a drop-in port of R's one-vs-rest GLM,
`margins`, bootstrap, and crossed-`lme4` pooling workflow. It is a modular/cut
posterior whose categorical probabilities sum to one and whose cluster effects
for a common contrast sum to zero. Installation, response/category contracts,
fixed sampler profile, diagnostics, uncertainty decomposition, supported
envelope, and limitations are documented in
[the Bayesian RARCAT vignette](vignettes/bayesian_rarcat.qmd).

## Plotting Exports

- `plot_clustrange(range_obj, ...)`
- `clustrange_plot_data(range_obj, ...)`
- `print_clustrange(object, *, digits=2, bootstat=('t0', 'mean', 'stderr'))`
- `summary_clustrange(object, *, max_rank=1)`
- `print_clustrangefamily(object, *, max_rank=1)`
- `summary_clustrangefamily(object, *, max_rank=1)`
- `print_kmedoids(result, *, digits=7)`
- `plot_bootclustrange(boot_obj, ...)`
- `plot_seqclararange_stability(sc_obj, ...)`
- `plot_seqnullcqi(bcq, ...)`
- `print_seqnullcqi(bcq, ...)`
- `plot_clustrangefamily(fam_obj, ...)`
- `clustrangefamily_plot_data(fam_obj, ...)`
- `plot_clustassoc(assoc, *, stat=("Unaccounted", "Remaining", "BIC"), type="b", ax=None, tooltip=True, kind=None, **plot_kwargs)`
- `plot_proptree_labels(tree, ...)`
- `plot_proptree_tree(tree, ...)`
- `plot_proptree_panel(tree, ...)`
- `proptree_plot_data(tree)`
- `fuzzyseqplot(seqdata, group=None, membership_threashold=0, type="i", members_weighted=True, memb_exp=1, ...)`

`fuzzyseqplot` accepts R's `main` plot title and maps `with.legend` to the
Python keyword `with_legend`; duplicate or unsupported plot keywords raise.
For index types `i`/`I`, finite fuzzy weights determine variable row heights
after traminex performs sorting and optional systematic/random selection.
Requested weighted index rendering validates the complete weight sidecar as
finite and nonnegative before selection. The index keyword `weighted=False`
selects and draws unit-weight rows, including when a powered membership is
nonfinite; it is distinct from `members_weighted=False`, which keeps R's
membership-only weights while ignoring attached case weights.
Raw row-height scale is retained unless floating-point addition at the `-0.5`
origin would erase a positive cell; in that exceptional case one common
display normalization preserves proportional geometry within the representable
axis range. A wider dynamic range or nonrepresentable cumulative total fails
before a figure is created rather than erasing a positive row.
Plotting parity covers semantic plot-driving arrays and options plus supported
presentation contracts. It includes named safe translations where installed
WeightedCluster 2.0 fails, such as scalar/RHC `clustrange` paths and the
grouped-method vector-stat `clustrangefamily` path. It does not claim
pixel-identical rendering, R console padding, or interactive base-R device
behavior. Matplotlib remains a lazy optional dependency: base imports do not
require it, and rendering is available through `.[plot]`.

## Metadata / Runtime

- `get_backend_info()`
- `__version__`

`get_backend_info()['capabilities']` provides JSON-safe, operation-level capability diagnostics:
owner/provider, exact implementation symbols and their
callability, public dispatch order, and selection timing. The legacy
`compiled_core`, `numba`, and `pure_python` strings report module availability
only and must not be interpreted as a global dispatch order. See the canonical
[README backend model](../README.md#backend-model) for the public routing table.

## Common Return Structures

Canonical typed contracts and coercion/validation helpers for these payloads are defined in
`src/weightedcluster/schemas.py`.

### `seqclararange()`

```python
# docs-audit: display-only reason="typed result schema pseudocode"
{
  "kvals": list[int],
  "clustering": {"clusterK": np.ndarray},
  "stats": dict,          # R-style core table columns
  "quality_stats": dict,  # extended quality indices; fuzzy/PCMdd runs also add FPC/FPE/XBf
  "clara": dict           # per-K run details
}
```

### `bootclustrange()`

```python
# docs-audit: display-only reason="typed result schema pseudocode"
{
  "kvals": list[int],
  "clustering": dict,
  "stats": dict,          # bootstrap means (same object as "meant")
  "meant": dict,          # bootstrap means by stat and K
  "stderr": dict,         # bootstrap standard errors by stat and K
  "boot": dict,           # boot[k][stat] -> list[float]
}
# Python-only diagnostics and controls are explicit off-table metadata:
result.additions["baseline_stats"]
result.additions["sampling"]
```

Input contract: `seqdata` is required, and `bootclustrange` recomputes baseline and bootstrap distances from `seqdata` rather than consuming `object["diss"]`. Matching WeightedCluster 2.0, weights attached to an `STSList` are retained as result metadata but do not enter the baseline or replicate cluster-quality calculations.

### `wcKMedoids()`

```python
# docs-audit: display-only reason="typed result schema pseudocode"
{
  "clustering": np.ndarray,   # one-based medoid observation position per row
  "info": {                   # exact R names/order
      "Sum of Distances": float,
      "N. Found": float,
      "Method": float,
  },
  "stats": dict[str, float],  # scalar quality stats
  "ASW": dict[int, float],    # keyed by one-based medoid position
  "ASWw": dict[int, float],   # keyed by one-based medoid position
}
```

`method` follows R's case-insensitive partial matching and numeric codes 1:3.
WeightedCluster 2.0 implements `PAM` for square matrices but raises its
upstream not-implemented error for condensed `dist` input; Python preserves
that representation-specific contract. `print_kmedoids()` translates
`print.kmedoids` semantically without reproducing R console column padding.

### `seqnullcqi()`

`hclust_method` is retained in result metadata; it does not select a different
hierarchical linkage. The non-k-medoids path is fixed to Ward.D for
WeightedCluster 2.0 compatibility, while `kmedoid=True` selects the medoid
path. That R-faithful reclustering runs through the quadratic-time
nearest-neighbour-chain port of `fastcluster::hclust` for every bootstrap
replicate; its Python-loop overhead still makes large workloads slow. Large
`N² × R` workloads emit a `RuntimeWarning`; `kmedoid=True` is a scientifically
different clustering method, not a performance-equivalent switch.

```python
# docs-audit: display-only reason="typed result schema pseudocode"
{
  "seqdata": np.ndarray,     # stacked null datasets, shape (R*N, T)
  "stats": dict,             # stats[stat] -> np.ndarray of shape (R, n_k)
  "clustrange": dict,        # includes observed stats
  "R": int,
  "kmedoid": bool,
  "hclust.method": str,
  "seqdist.args": dict,
  "boot": dict,              # convenience: boot[k][stat] -> list
  "nullmodel": dict
}
```

## Notes

- Cluster labels are 1-based at API boundaries for R-style compatibility.
  `wcKMedoids["clustering"]` more specifically contains the one-based
  observation position of each row's medoid, matching R.
- Some helper functions used in docs/examples are internal (not exported), e.g.:
  - `weightedcluster._seqdist.sequences_distance`
  - `weightedcluster._threads.limit_threads`

### Four package attributes are functions, not modules

`weightedcluster/__init__.py` imports four functions whose names match the
modules they live in, so after `import weightedcluster` the package attribute is
the **function**:

| Attribute | Resolves to | The module is still importable as |
|---|---|---|
| `weightedcluster.clustassoc` | the `clustassoc` function | `importlib.import_module("weightedcluster.clustassoc")` |
| `weightedcluster.seqclararange` | the `seqclararange` function | `importlib.import_module("weightedcluster.seqclararange")` |
| `weightedcluster.seqclustname` | the `seqclustname` function | `importlib.import_module("weightedcluster.seqclustname")` |
| `weightedcluster.seqnull` | the `seqnull` function | `importlib.import_module("weightedcluster.seqnull")` |

The public names are kept because they mirror R's, where `seqnull` and
`clustassoc` are functions. The consequence to know is that
`weightedcluster.seqnull.seqnullcqi` raises `AttributeError` after a plain
`import weightedcluster`, while `weightedcluster.boot.bootclustrange` works,
because `boot` is not shadowed. Use `from weightedcluster import seqnullcqi`, or
`importlib.import_module`, to reach a shadowed module.

Finding F-17 of the 2026-08-24 whole-codebase audit. The count is **four**,
measured over all 40 submodules; an earlier five-name sample reported three and
missed `seqclustname`.

**Versioning:** see `CHANGELOG.md` (version single-sourced from `pyproject.toml`).

# Workflow Guide

How to choose the right workflow in `weightedcluster`.

## Start Here

1. Do you already have a distance matrix?
- Yes: use the distance-first workflow.
- No: use the sequence-first workflow.

2. Do you need null-model validation?
- Yes: use `seqnullcqi`.

3. Do you need bootstrap uncertainty around an existing clustering object?
- Yes: use `bootclustrange`.

## Distance-First Workflow

Use when `diss` is already computed.

### Typical path

1. `wcKMedRange(diss, kvals, ...)`
2. `wcClusterQuality(diss, clustering, ...)`
3. `wcCmpCluster(diss, maxcluster=..., ...)` if method comparison is needed

### Example

```python
# docs-audit: smoke id=workflow-distance-first-smoke
import numpy as np
from weightedcluster import wcKMedRange

D = np.array([
    [0.0, 0.2, 1.0, 1.1],
    [0.2, 0.0, 1.0, 1.1],
    [1.0, 1.0, 0.0, 0.2],
    [1.1, 1.1, 0.2, 0.0],
], dtype=float)

out = wcKMedRange(D, kvals=[2, 3])
print(out["kvals"])
print(out["stats"].keys())
```

## Sequence-First Workflow

Use when your data is sequence array `(N, T)`.

Shared parallel contract:
- `seqclararange`, `seqclararange_family`, `bootclustrange`, and `seqnullcqi` all use `parallel=True` plus optional `n_jobs=` with the same worker-count semantics.
- A standalone `.py` script must invoke these parallel calls below
  `if __name__ == "__main__":`; this process-spawn guard prevents recursive
  module execution. It is not needed for an ordinary notebook cell.

### Typical path

1. `seqclararange(seqdata, kvals, seqdist_args=..., ...)`
2. Optional `bootclustrange(clustrange_obj, seqdata, ...)`
3. Optional `seqnullcqi(seqdata, clustrange_obj, ...)`

### Example

```python
# docs-audit: sequential id=workflow-sequence-first-smoke
import numpy as np
from weightedcluster import seqclararange

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
    random_state=7,
)

print(cr["kvals"])
print(cr["stats"].keys())
```

## Bootstrap Validation (`bootclustrange`)

Use this to quantify variability of quality indices under subsampling.
`bootclustrange` always recomputes full-data and resampled distances from `seqdata`; it does not reuse or accept a hidden `object["diss"]` payload.

```python
# docs-audit: sequential id=workflow-bootstrap-smoke
from weightedcluster import bootclustrange

bc = bootclustrange(
    cr,
    X,
    R=30,
    sample_size=4,
    sampling="random",
    random_state=7,
)

print(bc["meant"].keys())          # bootstrap means by stat
print(bc["stderr"].keys())         # bootstrap standard errors by stat
print(bc.additions["baseline_stats"].keys()) # full-data baseline stats
```

Sampling modes:
- `clustering` (default)
- `strata` (requires `strata=`)
- `medoids`
- `random`

## Null-Model Validation (`seqnullcqi`)

Use when you want observed-vs-null comparisons.

```python
# docs-audit: sequential id=workflow-null-smoke
from weightedcluster import seqnullcqi

nc = seqnullcqi(
    X,
    cr,
    R=20,
    model="combined",
    seqdist_args={"method": "HAM"},
    random_state=7,
)

print(nc["stats"].keys())              # stat names
print(nc["stats"]["ASW"].shape)       # (R, len(kvals))
print(nc["hclust.method"], nc["seqdist.args"])
```

Plot/print helpers:
- `plot_seqnullcqi(nc, stat="ASW", type="line")`
- `plot_seqnullcqi(nc, stat="ASW", type="boxplot")`
- `plot_seqnullcqi(nc, type="seqdplot")`
- `print_seqnullcqi(nc, norm=True, quant=0.95)`

## Choosing Between `seqclararange` and `wcKMedRange`

Use `seqclararange` when:
- you start from sequence data
- you want CLARA sampling logic
- you want R-style sequence clustering outputs

Use `wcKMedRange` when:
- you already computed distances
- you want repeated K-medoids range on fixed `diss`

## Plotting Overview

- `plot_clustrange` for single range object
- `clustrange_plot_data` for headless/semantic range plot arrays
- `print_clustrange` and `summary_clustrange` for R-style range tables
- `plot_clustrangefamily` for family/method comparisons
- `clustrangefamily_plot_data` for headless family panel arrays
- `print_clustrangefamily` and `summary_clustrangefamily` for cross-method rankings
- `plot_bootclustrange` for bootstrap distributions
- `plot_seqclararange_stability` for stability diagnostics
- `plot_clustassoc` for association curves
- `plot_proptree_*` for property trees

## R-Compatibility Notes

- Cluster labels are exposed as 1-based.
- R-style field naming is preserved where meaningful (e.g., `hclust.method`, `seqdist.args`).
- `wcCmpCluster` requires `maxcluster` and defaults to R's
  `pam_combine=True`; explicit zero-based `replay_starts` make the standalone
  PAM branch deterministic across runtimes. Its complete K range must satisfy
  `2 <= K <= N`; `maxcluster>N` fails before method dispatch.
- Property-tree permutations are currently serial-only. Both
  `wcPropertyClustering` and `seqpropclust` explicitly accept keyword-only
  `perm_n_jobs=None`, the `n_jobs=None` alias, and `progressbar=False`. `None`
  resolves to one worker. Supplying both aliases with non-`None` values
  conflicts; any worker count other than `1` and `progressbar=True` raise
  `NotImplementedError`.

## See Also

- `docs/API_INDEX.md`
- `docs/ARCHITECTURE.md`
- `docs/R_MIGRATION_GUIDE.md`
- `docs/vignettes/README.md`

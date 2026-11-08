# Migration Plan — converging the traminex / pyweightedcluster / flexscanx family

Companion to `ECOSYSTEM.md` and `DUPLICATES.md`. The completed migration's
master record lives in `PROJECT_TRAMINEX`; this consuming-project copy carries
the post-plan Phase 2c boundary note below. Its historical baseline was compared
with and synchronized from the traminex master on 2026-07-15 before that note
was added.
Written: 2026-06-12. **Status: COMPLETE (2026-06-13). All phases (0, 1, 2, 3)
are done; see the per-phase "status: COMPLETE" blocks below. There is no
Phase 4 — the migration plan is finished and awaiting maintainer review.**

> **Post-plan Phase 2c boundary (2026-07-15).** The completed A1–A8 migration
> remains historical truth, but pyweightedcluster no longer grows a second
> property tree on top of the shared pairwise primitive. `wcPropertyClustering`
> delegates tree growth to `traminex.disstree`, and `seqpropclust` delegates its
> tree stage to `traminex.seqtree`; the local adapter is structural only.
> `clustassoc` now delegates its exact discrepancy terms to
> `traminex.dissmfacw`, whose numeric/categorical mixed-design behavior is pinned
> to TraMineR 2.2-13. Sequence-distance consumption is locked to the explicit
> public `traminex.distances` submodule (`pairwise` and
> `pairwise_between_sets`) without expanding traminex's root API.

This is the written, step-by-step plan the maintainer asked for before any code
changes. It turns the decisions in `DUPLICATES.md` into an ordered, testable
sequence of small changes, each one reversible.

---

## Decisions this plan implements

1. **Build** faithful weighted discrepancy/tree primitives in traminex from the R
   source (the faithful maths currently exists only *inline* in pyweightedcluster,
   not as functions to move); then refactor pyweightedcluster to use them. (A5–A8)
2. **Replace** pyweightedcluster's private distance/cost code with traminex's
   R-validated, weight-aware versions. (A1–A3)
3. **flexscanx stays lightweight but identical:** it keeps its own sequence
   copies and must install/run **without** traminex; traminex is a *test-only*
   dependency used to prove those copies stay identical. (B1–B5)
4. **Keep** each project's genuinely own code (weighted PAM/k-medoids, quality,
   fuzzy, CLARA, seqnull in pyweightedcluster; scan statistics and
   `extract_sequences` in flexscanx).
5. The `disscenter` name clash is fixed inside step A7, not as a separate pass.

## Ground rules (apply to every step)

- **The acceptance test is agreement with the original R/C++ source**, not with
  either Python copy. R goldens are generated from the vendored originals
  (`PROJECT_TRAMINEX\SOURCES`, `PROJECT_PY_WEIGHTEDCLUSTER\orig_R_pkg`).
- **One item per commit.** Every project's existing test suite stays green at
  every step.
- **No big-bang merges.** Each change adds the new path, proves equivalence, then
  removes the old path.
- **Reversible.** Because changes are small and sequential, any step can be
  reverted by a single `git revert` without unwinding others.
- **Versioning.** Each consuming change that alters behaviour or dependencies
  bumps that project's patch version and notes it in its CHANGELOG.

---

## Phase 0 — Foundations (no behaviour change)

**0.1 Parity-test harness.** Add, in each project's `tests/`, a small helper that
loads R goldens (or live R via rpy2 where available) for the functions about to
change. pyweightedcluster already has `tests/r_parity_utils.py`; traminex already
has `tests/test_r_goldens_*`. Reuse these patterns.

**0.2 Pin the source-of-truth snapshots.** Record the exact R package versions of
the vendored originals (TraMineR, WeightedCluster) in `DUPLICATES.md` so goldens
are reproducible.

**0.3 Decide how the projects reference each other locally.** traminex is **not
on PyPI** — it is a local project. Before any "add traminex as a dependency"
step, choose the local mechanism (path dependency, editable install `pip install
-e`, or a private index) and document it. This applies to pyweightedcluster's
runtime use (Phase 1/2) and flexscanx's test-only use (Phase 3).

> **DECIDED (2026-06-12): editable installs into the shared environment,
> installed in dependency order.**
>
> - The historical environment at the time of this decision was
>   `%USERPROFILE%\miniforge3\envs\datascience_core`. The current canonical
>   development environment is Ubuntu 24.04, where traminex is installed with
>   `~/miniforge3/envs/datasci/bin/python -m pip install -e ~/Projects/PROJECT_TRAMINEX`
>   (done as part of Phase 0; this also registers pip metadata
>   `traminex 0.0.3`, which the previous ad-hoc setup lacked).
> - Consuming projects declare the dependency by **bare distribution name**
>   (`"traminex"`) — in `[project.dependencies]` for pyweightedcluster
>   (Phase 1) and in an optional `test` extra for flexscanx (Phase 0.4 /
>   Phase 3). Resolution works because traminex is installed *first*; pip then
>   sees the requirement as already satisfied and never contacts PyPI for it.
> - **Install order is therefore part of the mechanism:** traminex →
>   pyweightedcluster → flexscanx. Never run an install that would let pip try
>   to fetch `traminex` from PyPI (the name is unclaimed there — a
>   dependency-confusion risk); if pip ever attempts to download `traminex`,
>   that is a setup error: install the local traminex first.
> - Not chosen: absolute-path/`file://` dependencies in `pyproject.toml`
>   (machine-specific metadata baked into wheels) and a private index
>   (overkill for a single-machine family).
> - Historical note: flexscanx and pyweightedcluster were previously wired into
>   the environment via hand-rolled `.pth` files pointing at their `src/`
>   trees. That still works for imports, but provides no pip metadata, so it
>   cannot satisfy declared dependencies; new cross-project links must use the
>   editable-install mechanism above.

**0.4 Dev-only traminex for flexscanx.** Add traminex to flexscanx's *test/dev*
dependencies only (e.g. an optional `[test]` extra), never to its install
requirements. Confirm `pip install flexscanx` still works with traminex absent.

Exit criteria: harness runs; the local cross-project install mechanism is chosen
and documented; `flexscanx` installs and imports with traminex not installed.

### Phase 0 status: COMPLETE (2026-06-12). Exit criteria met.

Done, one item per commit, every suite at or above baseline:

- **0.1 harnesses.** traminex `tests/r_goldens_utils.py` + `tests/r_goldens/*.csv`
  (file-based goldens for the Phase 2 `dissvar`/`dissassoc`/`disscenter`/
  `disstree` work; same `%.15g` / `atol=1e-10` convention as the existing inline
  goldens). pyweightedcluster `tests/traminex_parity_utils.py` (traminex import
  gate, strict switch `WC_PARITY_STRICT=1`, state-matrix→STSList converters).
  flexscanx `tests/traminex_parity_utils.py` (same, `FLEXSCANX_PARITY_STRICT=1`,
  plus a subprocess test that `flexscanx.sequence` never imports traminex). Each
  has Phase-0 smoke tests; the real comparison tests are deferred to Phases 1–3.
- **0.2 version pins.** Recorded in `DUPLICATES.md` (synced to all three repos).
- **0.3 install mechanism.** Decided and realized (block above). Realizing it
  exposed and fixed a real bug: traminex's `pyproject.toml` carried an invalid
  `[tool.scikit-build.metadata]` block (`requires-dist = []`, a list where
  scikit-build-core expects a dict) that made **every** `pip install -e .` fail;
  removed in its own commit. traminex is now editable-installed as `0.0.3`.
- **0.4 test-only dep.** flexscanx gained a `[test]` extra (`pytest`,
  `traminex>=0.0.3`), never an install requirement; verified a flexscanx wheel
  installs/imports/runs in a clean venv with traminex absent, and that
  `flexscanx[test]` in the shared env resolves traminex to the local editable
  install without a PyPI fetch.

**Findings carried into later phases:**

1. **HAM normalization (affects Phase 1 / A1).** Both Python copies already
   length-normalize HAM and expose **no** `norm` kwarg: traminex
   `seqdist(method="HAM")` and pyweightedcluster `sequences_distance(metric="HAM")`
   both give `D = mismatches / length` (e.g. `0.667` for 4 mismatches over
   length 6). That equals R's `seqdist(method="HAM", norm="maxlength")`, **not**
   R's default `norm="none"` (raw count `4`). Consequence: the two Python copies
   **agree with each other**, so the old-local-vs-traminex leg of the A1 test is
   trivially green for HAM and proves nothing on its own. The load-bearing check
   is against the **R source**: the A1 test must pin traminex HAM to R
   `norm="maxlength"` (and decide explicitly whether pyweightedcluster's
   clustering wants the normalized or the raw `norm="none"` distances — this is a
   semantic choice, not just a tolerance). Re-confirm the same normalization
   question for OM/OMspell/LCS/DHD, where defaults may differ between the copies
   and R.
2. **Shared-env quirks (apply when running the suites).** flexscanx needs
   `PROJ_DATA` pointed at the env's proj share
   (`...\datascience_core\Library\share\proj`) or ~26 pyproj tests error with
   "no database context specified". Independently, 7 flexscanx tests fail on this
   machine because the env's numba rejects its NumPy ("Numba needs NumPy 1.26 or
   less"); this is a **pre-existing baseline**, unrelated to the migration, and
   should not be read as a regression. Fixing it means upgrading numba in the
   shared env (maintainer's call).

---

## Phase 1 — pyweightedcluster distances & costs → traminex (A1, A2, A3)

Highest value: it puts clustering on numbers proven to match the R package.

For each of `_seqdist.py` (A1), `_omspell.py` (A2), `costs.py` (A3):

1. Add traminex to pyweightedcluster's install dependencies (once, at A1), using
   the local mechanism chosen in 0.3.
2. Write a comparison test: identical inputs through the **old local** function
   and the **traminex** function, asserting equal within tolerance; include the
   weighted path (per-sequence weights) and confirm it matches R's
   `seqtrate(weighted=TRUE)` for TRATE costs.
3. Switch pyweightedcluster's internal call sites to traminex.
4. Delete the local module once nothing references it; keep a thin shim only if
   the public API exposed it.
5. Run the full clustering test suite (k-medoids, quality, CLARA) — results must
   be unchanged.

Order within the phase: A3 (costs) → A1 (distances) → A2 (omspell), because
distances depend on costs.

Exit criteria: pyweightedcluster computes no sequence distances of its own; its
clustering runs on traminex distances; all parity and clustering tests green.

### Phase 1 status: COMPLETE (2026-06-12). Exit criteria met.

Done, one item per commit, both suites green at every step (traminex
445 passed / 1 skipped; pyweightedcluster 216 passed / 13 skipped, also green
under `WC_PARITY_STRICT=1`). Order honoured: A3 (costs) → A1 (distances) →
A2 (OMspell), then A4.

- **A3 (costs).** Removed `weightedcluster.costs` — it had **no internal
  consumers** (the clustering pipeline takes caller-supplied
  `substitution_costs`), so it was an unvalidated duplicate of traminex's cost
  functions. traminex `seqsubm` is the cost home. New
  `tests/test_phase1_a3_costs_traminex_parity.py` pins CONSTANT and
  weighted/unweighted TRATE to R `TraMineR::seqcost`; the old local matrix was
  verified against R before removal.
- **A1 (distances).** `_seqdist.sequences_distance[_between]` delegates
  HAM/OM/LCS/DHD to `traminex.distances.pairwise`; declared `traminex>=0.0.3` as
  a runtime dependency. The private distance backends are no longer selected:
  the dead `_fallback` and `_numba` kernels were stripped during Phase 1, but
  eight foreign `_wc_core` exports remained until Phase 4 commit `aa72c99`,
  which removed their C++ helpers and bindings. The compiled extension now
  exports only pyweightedcluster-owned clustering/quality kernels.
  Normalization is **pinned to R** for the
  `normalize=True` form clustering uses: HAM == R `HAM(sm=unit, norm="maxlength")`,
  OM == R `OM(sm=2, indel=1, norm="maxlength")`, LCS == R `LCS(norm="gmean")`,
  DHD == R `seqdist(method="HAM"/"DHD", sm=...)`
  (`tests/test_phase1_a1_distances_traminex_parity.py`).
- **A2 (OMspell).** Removed `weightedcluster._omspell`; delegates to traminex's
  R-faithful OMspell (`tests/test_phase1_a2_omspell_traminex_parity.py`).
- **A4 (`normalize_metric`).** KEEP — a tiny metric-name canonicalizer (charter
  §4) that is the front door before delegating; folded into A1.

Findings carried into Phase 2 and beyond:

1. **One traminex change was required and made** (its own commit + R goldens):
   traminex's DHD silently ignored a user `sm`, a real faithfulness gap vs R.
   Added an explicit-`subm` DHD path (`pairwise(method="DHD", subm=...)`): static
   `(K,K)` == R `seqdist(method="HAM", sm=...)`, time-varying `(L,K,K)` == R
   `seqdist(method="DHD", sm=...)`. Goldens: `tests/r_goldens/DHD_subm_*.csv`,
   generator `scripts/r_goldens/make_goldens_dhd_subm.R`. **Precedent for Phase 2:**
   editing traminex to expose an R-faithful option it lacks is sanctioned, but
   needs its own commit + its own R-golden test.
2. **R-faithful corrections** made where the old copy diverged from R (no test
   pinned the old values; not clustering defaults): un-normalized LCS is now R's
   raw `|a|+|b|-2*lcs` (was half); OMspell now applies R's `-1` spell-duration
   shift.
3. **`normalize=True` is the clustering-relevant form** and is what the pins
   above lock; the raw forms are R `norm="none"`.

---

## Phase 2 — Build faithful weighted discrepancy & trees in traminex (A5–A8)

Important scoping correction (from the audit): there is **no ready-made
`dissvar`/`dissassoc`/`disscenter` function in pyweightedcluster to "move up."**
traminex has *simplified, unweighted* functions by those names; the *faithful*
weighted maths exists only **inline** inside pyweightedcluster, spread across
`seqtree.py` (`as_seqtree`), `proptree.py` (`seqpropclust`), `clustassoc.py`
(`_weighted_within_and_total`), and `quality.py`. So Phase 2 **builds** the
canonical primitives in traminex from the R source, using pyweightedcluster's
inline maths as a cross-check — then refactors *all* those inline call sites.

**2.1 Build `dissvar` (weighted discrepancy / pseudo-variance) in traminex.**
Port from R `dissvar.R` / the weighted-inertia C routine, cross-checked against
pyweightedcluster's inline weighted-pairwise-sum maths. Replace traminex's
simplified `dissvar`; keep the old behaviour only under a clearly different name
if anything used it.

**2.2 Build `dissassoc`** (weighted pseudo-F / pseudo-R² with permutation test)
in traminex from R `dissassoc.R`, the same way. Replace traminex's simplified
`dissassoc`.

**2.3 A7 — build the R-meaning `disscenter` and resolve the name clash.**
Introduce the weighted "distance of each case to its group centre" as traminex's
`disscenter` (matching R). **Rename** traminex's existing matrix double-centering
helper to an unambiguous name (e.g. `double_center_matrix`) with a deprecation
alias. This is the one place the name clash is addressed.

**2.4 Build the faithful weighted tree in traminex.** Port pyweightedcluster's
`as_seqtree` (weighted-inertia recursive split, faithful to R `as.seqtree`) into
traminex as the real `seqtree`/`disstree`. Relabel **both** of traminex's current
sklearn stand-ins — `disstree.py::disstree_fit` (MDS + tree) and
`seqtree.py::seqtree_fit` (feature-extraction + tree) — as the approximate
"exploratory" variants they actually are, not the TraMineR tree.

**2.5 Refactor pyweightedcluster to consume the new primitives.** This touches
**all** the inline sites: `seqtree.py`, `proptree.py`, `clustassoc.py`, and
`quality.py`. Replace each inline computation with a call to the traminex
primitive; prove identical results on the existing tests before deleting the
local maths. (Larger than a single import swap — budget accordingly.)

Each sub-step: add → parity-test against R → switch → remove old. One commit each.

Exit criteria: traminex is the single faithful, weighted home of `dissvar`,
`dissassoc`, `disscenter`, and the sequence tree; every pyweightedcluster inline
site uses them; the `disscenter` name is unambiguous family-wide; traminex's two
sklearn trees are clearly labelled as approximate.

### Phase 2 status: COMPLETE (2026-06-13). Exit criteria met.

Done, one item per commit, both suites green at every step (final: traminex
465 passed / 1 skipped; pyweightedcluster 224 passed / 13 skipped with live R,
also green under `WC_PARITY_STRICT=1`). All four primitives are R-golden
tested against live TraMineR 2.2-13, **including the weighted path**, on a
shared deterministic fixture (`tests/r_goldens/phase2_input_*.csv`, generator
`scripts/r_goldens/make_goldens_phase2_inputs.R`).

- **2.1 `dissvar` (traminex 0.0.4).** R-faithful weighted discrepancy; old
  dict-returning helper relabelled `dissvar_summary`. Added the reusable SS
  primitives `weighted_pairwise_sum` / `weighted_inertia`
  (`tmrWeightedInertiaDist` ports) that 2.2–2.5 build on.
  Test `tests/test_r_goldens_dissvar.py`.
- **2.2 `dissassoc` (traminex 0.0.5).** Weighted pseudo-ANOVA, per-group
  discrepancies, and [Pseudo F, Pseudo Fbf, Pseudo R2, Bartlett, Levene]
  with R's permutation conventions (`t[1]=t0`, `pval=mean(t>=t0)`; variants
  none/diss/replicate/rounded-replicate/group; R's `permute` and
  `random-sampling` not ported; `random_state` is a documented Python
  addition; R quirks preserved — Levene residuals constant under "diss").
  t0/tables R-pinned at R=1; p-values tested structurally (seeded
  reproducibility, 1/R floor, signal vs noise). Old effect size relabelled
  `dissassoc_effect` (`dissassoc_test` wraps it unchanged).
  Test `tests/test_r_goldens_dissassoc.py`.
- **2.3 `disscenter` (traminex 0.0.6) — name clash resolved.** R meaning
  (distance to weighted group centre; `allcenter`, `medoids_index`
  "first"/"all" 0-based, weighted, squared). Old double-centering helper →
  `double_center_matrix` + deprecated alias `disscenter_double_center`.
  Test `tests/test_r_goldens_disscenter.py`.
- **2.4 `disstree`/`seqtree` (traminex 0.0.7).** Full `disstree.R` port:
  weighted-inertia binary partitions of predictor levels (numeric/ordered
  thresholds, categorical subsets in R `combn` order, NA group), permutation
  gate (`DTNdissassocweighted`), node medoids, R2-based split schedule,
  `disstreeleaf`, `dissassoc` adjustment; `seqtree` reads weights from
  `seqdef(weights=)` like R's `attr(seqdata, "weights")`. Goldens pin
  preorder node tables, fitted leaves and adjustment t0 (unweighted,
  weighted "diss", and seqdef-weighted "replicate"). The sklearn stand-ins
  moved to `disstree_explore.py` / `seqtree_explore.py` with explicit
  "exploratory, NOT the TraMineR tree" labels; every historical public name
  still resolves. Test `tests/test_r_goldens_disstree.py`.
- **2.5 pyweightedcluster refactor (0.0.3, requires `traminex>=0.0.7`).**
  Inline weighted maths removed from all four sites — `seqtree.py`
  (`_weighted_pairwise_sum`), `_utils.py::pairwise_sum` (proptree splitter),
  `clustassoc.py::_weighted_within_and_total`, `quality.py` fallback moment
  loops (T/T2/W/W2 for CH/R2 and the HC moments) — all call
  `traminex.diss.weighted_pairwise_sum`. Equivalence proven before deletion:
  `tests/test_phase2_a5a8_traminex_parity.py` keeps verbatim copies of the
  removed maths as reference implementations (1e-12). The compiled
  `_wc_core` quality kernel (A9, WeightedCluster's own) is untouched.

Notes for Phase 3 and beyond: the Phase 2 goldens all live on one shared
deterministic 15-case fixture; regenerate every `phase2_*`/`dissassoc_*`/
`disscenter_*`/`disstree_*`/`seqtree_*` golden together if it ever changes.
traminex's editable install must be refreshed (`pip install -e`) after
version bumps so pip metadata matches `pyproject.toml`.

---

## Phase 3 — flexscanx sequence copies: identical, not imported (B1–B5)

flexscanx keeps its `sequence/` package and its standalone install. For each of
`seq_dist_hamming` (B1), `seq_trans` (B2), `seq_entropy` (B3), `seq_freq` (B4),
and the plots (B5):

1. Add a parity test (in the test-only environment, with traminex available)
   asserting the flexscanx copy matches traminex's corresponding function on the
   small `+1/-1/0` state space flexscanx uses.
2. Where the test reveals a difference, **fix the flexscanx copy to match
   traminex** (traminex is the source of truth). Do not change traminex to match
   flexscanx.
3. Update `sequence/NOTICE.txt` to record, per function, that it is a deliberate
   in-sync copy of the named traminex function, verified by parity test.
4. Confirm `flexscanx` still installs and runs with traminex absent.

`extract_sequences` (B6) has no traminex counterpart and gets no parity test.

Exit criteria: every shared flexscanx sequence function is parity-tested against
traminex; flexscanx still installs without traminex; NOTICE.txt is accurate.

### Phase 3 status: COMPLETE (2026-06-13). Exit criteria met.

Done, one item per commit, the flexscanx suite at or above baseline at every
step. After a clean C++ rebuild the suite was **386 passed, 1 failed, 3 skipped**
with `PROJ_DATA` set (re-measured 2026-06-13, Audit Phase A; +28 over the
358-pass Phase-3-start baseline from the new parity tests; **387/1/3** after
Audit Phase D, below). The single failure is the pre-existing numba-vs-NumPy
environment issue ("Numba needs NumPy 1.26 or less"), unrelated to this work. An
earlier "380 passed, 7 failed" count reflected a **stale compiled `.pyd`**
masking the current C++ source: 6 of those 7 failures clear on rebuild, leaving
only the one numba/NumPy failure. traminex was the test-time oracle (`0.0.7` at
Phase 3, re-confirmed against the Phase-B-fixed `0.0.8` in Audit Phase D); it is
itself R-anchored, so no live R was needed.

**Audit Phase D (2026-06-13):** re-ran B1–B5 against the Phase-B-fixed oracle
(traminex `0.0.8`) — all parity green, no flexscanx drift (the Phase B TRATE fix
touched only traminex's `distances/costs.py`, not the `seqtrate` / `seqtransn` /
`seqstatd` / `seqtab` parity targets). Strengthened two parity tests to be
load-bearing — **B2** now exercises R `seqtrate`'s absent-source zero-row
(`PA == 0` → zeroed, not `NaN`) and **B4**'s tolerance is tightened to `1e-12` —
and re-proved the standalone guarantee by building the wheel and running it in a
clean venv with traminex absent. Test-only: suite `386` → `387/1/3` (+1 = the new
B2 test); flexscanx stays `0.0.2`.

- **B1 (`seq_dist_hamming`) — DONE, no behaviour change.**
  `tests/test_traminex_parity_b1_hamming.py`: unweighted `norm=True` ==
  `traminex.seqdist(method="HAM")` (mismatches/length); `norm=False` == HAM*T
  == unit-weight explicit-subm DHD; condensed == squareform of the HAM matrix;
  per-position-weighted raw == `distances.pairwise(method="DHD", subm=...)`
  with `subm[t][a,b]=w_t*(a!=b)` (the Phase 1 explicit-subm DHD path).
  Documented (not drift): the weighted `norm=True` form divides by
  `sum(weights)` — flexscanx's own [0,1] convention, coinciding with traminex
  for uniform weights.
- **B2 (`seq_trans`) — DONE, no behaviour change; attribution corrected.**
  `tests/test_traminex_parity_b2_trans.py`: `normalize=True` ==
  `traminex.seqtrate` (== `seqtransn(normalize=True)`); `normalize=False` ==
  `seqtransn(normalize=False)` raw counts; explicit from->to orientation guard.
  Fixed the trap: NOTICE credited `seqtransn`, but the rate matrix's
  counterpart is `seqtrate` (seqtransn is the per-sequence transition *count*).
  `lag>1` is a flexscanx extension with no traminex counterpart.
- **B3 (`seq_entropy`) — DONE, behaviour fixed (flexscanx 0.0.1 -> 0.0.2).**
  The only drift found. `seq_entropy(by_time=True)` returned unnormalized bits;
  traminex `seqstatd`'s Entropy row is normalized by `log(K)` (K = alphabet
  size). Fixed the flexscanx copy: cross-sectional entropy is now normalized by
  default via a new `normalize` parameter (default `True` == seqstatd;
  `normalize=False` recovers the old bits). `by_time=False` overall-diversity
  is unchanged (flexscanx-own, no counterpart). Version bump + CHANGELOG +
  docs-metadata re-sync in the same commit.
  `tests/test_traminex_parity_b3_entropy.py` pins it to `seqstatd` (incl.
  asymmetric fixtures where a column is missing a state, so the denominator
  must be the full alphabet K).
- **B4 (`seq_freq`) — DONE, no behaviour change.**
  `tests/test_traminex_parity_b4_freq.py`: per-pattern counts and proportions
  == `traminex.seqtab` (dict-keyed by the "-"-joined pattern, so tie-break
  ordering is irrelevant); `top_n` truncation keeps the highest-count patterns.
- **B5 (plots) — DONE, no behaviour change; data-level parity.**
  `tests/test_traminex_parity_b5_plots.py`: `seq_dplot` stacked-bar heights ==
  `seqstatd` Frequencies; `seq_fplot` bar widths == `seq_freq` proportions
  (== seqtab via B4); `seq_iplot` places one image row per input sequence in
  input order (row 0 top), the same row->sequence convention as
  `traminex.sequence_index_plot`. state->color vs code->colormap is a
  documented styling-only difference.
- **B6 (`extract_sequences`) — untouched** (flexscanx's own; no parity target).

`sequence/NOTICE.txt` rewritten per function (correct traminex counterpart +
parity-test reference; stale `seq_dist_gower` entry removed; `seq_trans`
re-attributed to `seqtrate`). Strict-mode proof: in the shared env all 22
parity assertions execute under `FLEXSCANX_PARITY_STRICT=1` (none skip).
Standalone-install proof: a `flexscanx 0.0.2` wheel installs in a clean venv
**without** traminex, imports and runs the sequence helpers, `flexscanx.sequence`
never imports traminex, and the parity tests skip cleanly there.

**Phase 3 is the last phase. With it complete, MIGRATION_PLAN.md is complete —
there is no Phase 4.**

---

## What is explicitly NOT moved

- pyweightedcluster: weighted PAM/k-medoids, cluster quality, fuzzy, CLARA range,
  bootstrap, and `seqnull` — these are WeightedCluster's own and stay put.
- flexscanx: spatial scan statistics and `extract_sequences` — flexscanx's own.

## Suggested overall order

Phase 0 → Phase 1 (A3, A1, A2) → Phase 2 (2.1–2.5) → Phase 3 (B1–B5).
Phases 1 and 3 are independent and could run in parallel by different sessions;
Phase 2 should follow Phase 1 so the distance foundation is settled first.

## Rollback

Every step is a single small commit behind a green test gate. To undo, revert
that commit; because the old path is removed only after the new path is proven,
reverting restores a working state without touching later steps.

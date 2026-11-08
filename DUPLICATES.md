# Duplicate Function Inventory & Convergence Decisions


> **Reconciliation note (2026-08-24).** `DUPLICATES.md`'s family baseline copy
> lives in `PROJECT_TRAMINEX`. The corrections above — POSIX paths in place of
> Windows backslash paths, and the golden runtime restated as the active Ubuntu
> R 4.6.1 — were made **only in this repository's copy**, under `ECOSYSTEM.md`
> §7's two-way reconciliation rule, which forbids overwriting a sibling and
> allows a consumer to record a verified consumer-observed correction. The
> sibling copy was read and not modified.

Companion to `ECOSYSTEM.md`. The cross-project inventory baseline lives in
`PROJECT_TRAMINEX`; this consuming-project copy adds the completed Phase 2c
boundary below. Last reviewed: 2026-08-10 (reconciled the consuming distance
boundary with Traminex's audit repairs). Earlier: 2026-07-15 (recorded
pyweightedcluster's full property-tree delegation). Earlier: 2026-06-15
(clarified that **rflexscan is flexscanx's starting
point, not its source of truth** — flexscanx corrects rflexscan/FleXScan's
algorithmic/mathematical errors; only flexscanx's `sequence/` copies are an
R-anchored parity target, via traminex/TraMineR). Earlier: 2026-06-12 (revised
after an audit against the original R/C++ source).

This file lists every place where pyweightedcluster or flexscanx reimplements
something that belongs to traminex's domain, compares the two implementations
**against the original R/C++ packages (the source of truth)**, and records a
recommendation. Possible recommendations:

- **REPLACE** — drop the local copy, use traminex.
- **PROMOTE** — the better implementation currently lives in the *sibling*, not
  in traminex; move/port it up into traminex, then have the sibling use it.
- **KEEP** — deliberate local code; it is genuinely that project's own.
- **DECIDE** — needs the maintainer's judgement before acting.

> The analysis sections below preserve the pre-migration evidence and decision
> rationale. The dated status blocks are the current truth: A1–A8 and B1–B5 are
> complete, and the Phase 2c addendum records the later consuming-side boundary.
> See the full step-by-step `MIGRATION_PLAN.md` for the historical execution
> record.

## 2026-11-08 reviewed manifest update to traminex 0.2.3

The commitment moved from sibling revision `2ed89d77a75d67a4ac319c05da463a1758244503` (0.2.2) to
`2cbb3472736d0ccce32a291241983a12b8921122`, traminex's patch release 0.2.3. Three of the 83 committed files changed
(`distances/__init__.py`, `format.py`, `seqformat.py`); the path list is unchanged and the aggregate SHA-256 is now
`4a0a7a647903bc6d61bedc824ec35b2e1c4fb74a67e04c5fdd7f9085d2f3d99b`, recomputed by the procedure in
`docs/maintainers/README.md`. The delta was reviewed and measured against both revisions: two of its changes reach this
package, through the `seqdef` and `seqformat` calls of `seqclustname`. Complex-valued states are written as R's
`as.character` writes them (`1+1i`, formerly `(1+1j)`), and rows given as matrices of unequal widths are refused
(formerly their printed forms became states); `tests/test_traminex_0_2_3_consumer_contracts.py` pins both. The
third change, `seqdist` with TRATE costs whose largest cost is 0 and refseq index sets, is not reached: no module
under `src/` passes `refseq`.

## 2026-11-08 reviewed manifest update to traminex 0.2.2 (version only)

The commitment moved from sibling revision `7bebb7a4026f7f8037e1e196d005f4cb5ffa03a4` (0.2.1) to
`2ed89d77a75d67a4ac319c05da463a1758244503`, traminex's patch release 0.2.2, which removes the maintainer's local paths
from traminex's published files: no file under `traminex/` changed, so the 83-path inventory and the aggregate SHA-256
`e2ccb374b64010843e58b334299f9902edada5e5056128567afa728bbe39aabb` are unchanged, recomputed by the
procedure in `docs/maintainers/README.md`.

## 2026-11-08 reviewed manifest update (version only)

The commitment moved from sibling revision `928186e80ff263f4a5dca8270aca8407a20a12a9` to
`7bebb7a4026f7f8037e1e196d005f4cb5ffa03a4`, traminex's patch release 0.2.1: only the
version changed (no file under `traminex/` did), so the 83-path inventory and the aggregate SHA-256
`e2ccb374b64010843e58b334299f9902edada5e5056128567afa728bbe39aabb` are unchanged, recomputed by the
procedure in `docs/maintainers/README.md`.

## 2026-11-07 reviewed manifest update

The consuming-boundary commitment was re-pinned from clean sibling revision
`6e76440b8ed7fc4a1c885c78546425a877bac275` (digest `d6eafe7300d36493d0f814692fe25e51fd5170036eea827305200e12832e5443`) to clean sibling revision
`928186e80ff263f4a5dca8270aca8407a20a12a9` (traminex 0.2.0 plus its fixes of 2026-11-07),
aggregate SHA-256 `e2ccb374b64010843e58b334299f9902edada5e5056128567afa728bbe39aabb`, following
the reviewed manifest-update procedure in `docs/maintainers/README.md`, at the
maintainer's request to fix all known issues. The inventory grows from 79 to 83
paths: `_rformat.py`, `_rlinpack.py`, `_rsum.py` and `distances/_batched.py`
were added; none was removed. The delta was measured by running this suite
against the new revision (the parent supervision repository's study of
2026-11-02 and its `docs/FIXES_2026-11-07.md`): 28 tests failed before the
consumer changes. Nine came from this project's own multiplication order in
`proptree.py` (corrected to R's); six matched an older warning text (traminex
now gives R's); four from traminex's DHD refusing any non-finite cost (the
unobserved entries are now handed over as 0, with no value changed); two from
the asymmetric between-set direction, which now follows R's `refseq=list(a, b)`
as WeightedCluster's CLARA uses it; two from `dissmfacw`'s NaN for a term that
adds no rank (a recorded departure in `clustassoc`); and three were this
commitment's own tests. `pyproject.toml` raises the version floor to
`traminex>=0.2.0`; this project's environment imports the sibling checkout
again, as `AGENTS.md` requires.

## 2026-09-06 reviewed manifest update

The consuming-boundary commitment below was re-pinned from clean sibling
revision `1ebb46cd94321feb5bed59c50a864f405a2f4c59` (digest
`f2e0e09e3ecf658089639e361c4a0e50d422e921b929919e2f46cfb414e95ea3`) to clean
sibling revision `6e76440b8ed7fc4a1c885c78546425a877bac275`, aggregate
SHA-256 `d6eafe7300d36493d0f814692fe25e51fd5170036eea827305200e12832e5443`,
following the reviewed manifest-update procedure in
`docs/maintainers/README.md`. The 58-commit delta (49 package files, no
package file added or removed; the sorted 79-path inventory is unchanged) was
reviewed at every delegated boundary: STSList case weights keep the finite,
non-negative rule; the external missing/void-marker representation is
canonicalized so in-alphabet objects are unchanged; `weighted_pairwise_sum`'s
finite path is bit-identical; the statistics, tree, event, and visualization
entry points this project consumes are behavior-preserving for its inputs,
as the full ordinary and strict suites prove against the re-pinned provider.
One delegated contract changed, R-faithfully: public `traminex.seqdist`
now stops on a missing substitution cost for the OM family, exactly as R's
`seqdist` does, so `bootclustrange` and `seqnullcqi` inherit R's refusal for
`seqdist_args={"method": "OM"}` without `sm`/`substitution_costs`
(consumer regression:
`tests/test_audit_remediation_seqdist_boot.py::test_om_family_delegations_require_a_substitution_cost_like_r_seqdist`).
The low-level `sequences_distance` adapter keeps its documented `subm=2`
OM default. Ownership is unchanged.

## 2026-08-28 immutable consuming-boundary commitment

The bare version floor is insufficient to identify delegated behavior:
`traminex>=0.0.8` admitted sibling revisions that changed behavior-bearing Python
source while retaining version 0.0.8. pyweightedcluster now checks a
symlink-free Python source inventory for the reviewed traminex revision
recorded in `src/weightedcluster/_traminex_compatibility.json` — since the
2026-11-08 updates above, an 83-file inventory of `2cbb3472736d0ccce32a291241983a12b8921122`
with aggregate SHA-256
`4a0a7a647903bc6d61bedc824ec35b2e1c4fb74a67e04c5fdd7f9085d2f3d99b`
(from 2026-11-07 until the 0.2.3 update, `e2ccb374b64010843e58b334299f9902edada5e5056128567afa728bbe39aabb`; from 2026-09-06, `6e76440b8ed7fc4a1c885c78546425a877bac275`, aggregate SHA-256
`d6eafe7300d36493d0f814692fe25e51fd5170036eea827305200e12832e5443`; originally `1ebb46cd94321feb5bed59c50a864f405a2f4c59`, aggregate SHA-256
`f2e0e09e3ecf658089639e361c4a0e50d422e921b929919e2f46cfb414e95ea3`).
The packaged manifest and verifier reject missing, extra, changed, or symlinked
Python sources before any delegated traminex symbol is imported. Cache files,
compiled artifacts, and distribution metadata do not enter the digest.
Before/after open-file identities, pathname identities, and a final reinventory
detect mutation overlapping verification. Success is a cached, point-in-time
startup result; the provider is assumed stable afterward, and post-check
mutation requires process restart/reverification.

This is a consuming-side compatibility check, not duplicated traminex
mathematics and not a claim that every future provider revision is
incompatible. A reviewed manifest update requires a clean sibling revision,
review of its behavior-bearing delta, the affected boundary and provider test
runs, recomputation of the sorted `path + NUL + bytes + NUL` commitment, and
fresh editable and wheel-install verification. pyweightedcluster continues to
delegate the same sequence, distance, discrepancy, and tree ownership.

## 2026-08-10 consuming-boundary reconciliation

pyweightedcluster was reverified against the read-only Traminex revision
`46fdddefe6d466ce7d1c872aff17868ff848e9ac`. That revision corrects asymmetric
OM/HAM/DHD substitution costs to follow TraMineR's stable first-occurrence
distinct-sequence orientation, propagates the source asymmetry warning, and
keeps compiled and pure-provider behavior aligned. Consumer-level live-R tests
now pin both discriminating duplicate-row orders and require rectangular
between-set distances to equal the corresponding full-matrix cross-block.

Ownership is unchanged: Traminex computes the distances and pyweightedcluster's
`_seqdist.py` remains a translation-only adapter. At this 2026-08-10 checkpoint,
Traminex was still version 0.0.8 and had no configured external source
coordinate, so the exact revision was behavioral evidence, not an enforceable
package coordinate. The declared requirement remains `traminex>=0.0.8`, and
clean runners still require an authoritative `TRAMINEX_SOURCE`. The later
2026-08-28 block adds consuming-side content enforcement for that source. No
Traminex file was changed as part of either reconciliation.

## Phase 4 backend cleanup: COMPLETE (updated 2026-08-21)

The Phase 1 status below accurately records when `_fallback.py` and `_numba.py`
stopped owning distance kernels, but the compiled C++ remainder survived that
migration. Phase 4 commit `aa72c99` removed all eight foreign `seq_*` helpers and
pybind11 exports. Audit follow-up commit `737889b` removed the later-proven
unreachable native observation-silhouette binding. The installed `_wc_core`
now exposes exactly `weighted_medoids` and `cluster_quality`; sequence
distance calls continue to delegate to `traminex.distances`.

The same cleanup removed unused OpenMP discovery/linkage from the owned core.
The installed extension does not link `libgomp`. This updates current status
without changing the historical A1 rationale and chronology below.

## Phase 1 status: COMPLETE (2026-06-12) — A1, A2, A3, A4 done

pyweightedcluster's distance/cost stack now resolves to traminex, anchored on
TraMineR 2.2-13. One commit per item; both suites green at every step.

- **A3 (costs) — DONE.** `weightedcluster.costs` removed (no internal consumers;
  the clustering pipeline takes caller-supplied `substitution_costs`). traminex
  `seqsubm` is the cost home; new
  `tests/test_phase1_a3_costs_traminex_parity.py` pins CONSTANT and
  weighted/unweighted TRATE to R `TraMineR::seqcost`. The old local matrix was
  verified to agree with R before removal.
- **A1 (distances) — DONE.** `_seqdist.sequences_distance[_between]` delegates
  HAM/OM/LCS/DHD to `traminex.distances.pairwise`; the private 3-tier backend
  (compiled `_wc_core` / numba / pure `_fallback`) is no longer used for
  distances (dead distance kernels stripped from `_fallback.py`/`_numba.py`).
  Normalization is **pinned to R** for the `normalize=True` form clustering uses:
  HAM == R `HAM(sm=unit, norm="maxlength")`; OM == R `OM(sm=2, indel=1,
  norm="maxlength")`; LCS == R `LCS(norm="gmean")`; DHD == R
  `seqdist(method="HAM"/"DHD", sm=...)`. (R-faithful fix: `normalize=False` LCS is
  now R's raw `|a|+|b|-2*lcs`, not half of it; clustering only uses
  `normalize=True`.) Required one traminex change — DHD now honours an explicit
  `subm` (its own commit + R goldens `tests/r_goldens/DHD_subm_*.csv`), since R's
  DHD/HAM honour a user `sm` and traminex previously ignored it.
- **A2 (OMspell) — DONE.** `weightedcluster._omspell` removed;
  `metric="OMSPELL"` delegates to traminex's R-faithful OMspell. `timecost`->R
  `expcost`, `tpow=1`, `normalize=True`/`False` -> R `norm="maxlength"`/`"none"`
  (`tests/test_phase1_a2_omspell_traminex_parity.py`). The old local OMspell
  omitted R's `-1` spell-duration shift and was not R-faithful.
- **A4 (`normalize_metric`) — DONE (KEEP, folded into A1).**
  `weightedcluster._metrics.normalize_metric` stays as a deliberate tiny
  metric-name canonicalizer (charter section 4); it is the front door
  `_seqdist` uses before delegating to traminex, not a distance computation.

## Phase 2 status: COMPLETE (2026-06-13) — A5, A6, A7, A8 done

traminex is now the single faithful, **weighted** home of the discrepancy
analysis and the regression tree, anchored on TraMineR 2.2-13; every
pyweightedcluster inline site consumes it. One commit per item; both suites
green at every step (final: traminex 465 passed / 1 skipped; pyweightedcluster
224 passed / 13 skipped with live R, also under `WC_PARITY_STRICT=1`).

- **A5 (`dissvar`) — DONE** (traminex 0.0.4). R-faithful weighted discrepancy
  `dissvar(diss, weights=, squared=)`; old simplified dict version relabelled
  `dissvar_summary`. New reusable SS primitives `weighted_pairwise_sum` /
  `weighted_inertia` (ports of `tmrWeightedInertiaDist`). Goldens
  `tests/r_goldens/dissvar_phase2.csv` + shared fixture `phase2_input_*.csv`.
- **A6 (`dissassoc`) — DONE** (traminex 0.0.5). Weighted pseudo-ANOVA +
  [Pseudo F, Pseudo Fbf, Pseudo R2, Bartlett, Levene] with R's permutation
  conventions (variants none/diss/replicate/rounded-replicate/group; R's
  `permute`/`random-sampling` not ported; `random_state` is a Python
  addition). t0/tables R-pinned (`dissassoc_*.csv`); p-values stochastic,
  tested structurally. Old effect size relabelled `dissassoc_effect`.
- **A7 (`disscenter`) — DONE, name clash resolved** (traminex 0.0.6).
  `disscenter` now means R's distance-to-group-centre (incl. `allcenter`,
  `medoids_index="first"/"all"`, weighted, squared; goldens
  `disscenter_*.csv`). The old matrix double-centering helper is renamed
  `double_center_matrix` (deprecated alias `disscenter_double_center`).
- **A8 (`disstree`/`seqtree`) — DONE** (traminex 0.0.7). R-faithful
  weighted-inertia regression tree `traminex.disstree.disstree` (binary level
  partitions, permutation gate, medoids, split schedule, `disstreeleaf`,
  `dissassoc` adjustment) and `traminex.seqtree.seqtree` (weights from
  `seqdef(weights=)`). Goldens `disstree_*.csv` / `seqtree_*.csv` (node
  tables + fitted leaves + adjustment t0, weighted and unweighted). The two
  sklearn stand-ins are relabelled exploratory in
  `disstree_explore.py`/`seqtree_explore.py`; all historical names still
  resolve.
- **Consuming refactor (2.5) — DONE** (pyweightedcluster 0.0.3, dependency
  `traminex>=0.0.7`). Inline weighted maths removed from `seqtree.py`
  (`_weighted_pairwise_sum`), `_utils.py::pairwise_sum` (proptree's
  splitter), `clustassoc.py::_weighted_within_and_total` and `quality.py`'s
  fallback moment loops — all now call
  `traminex.diss.weighted_pairwise_sum`. Equivalence proven before deletion
  in `tests/test_phase2_a5a8_traminex_parity.py` (verbatim copies of the
  removed maths as references). The compiled `_wc_core` quality kernel (A9,
  WeightedCluster's own) is untouched.

## Phase 2c status: COMPLETE (2026-07-15) — full consuming boundary

- `traminex.dissmfacw` now preserves continuous numeric predictors while
  retaining R treatment coding for categorical predictors; mixed, weighted,
  and unweighted designs are live-R pinned to TraMineR 2.2-13.
- `weightedcluster.clustassoc` delegates all discrepancy terms to
  `traminex.dissmfacw`; only the WeightedCluster-specific `lm`/factor-only
  `nnet::multinom` BIC adapters and result mapping remain local.
- `wcPropertyClustering` and the tree stage of `seqpropclust` delegate to
  `traminex.disstree` and `traminex.seqtree`. The local adapter owns structure
  and WeightedCluster's frontier split schedule only—no split selection,
  discrepancy, permutation, or tree-growth mathematics.
- The stable distance boundary is the explicit `traminex.distances` submodule;
  `pairwise` and `pairwise_between_sets` remain outside the traminex root
  manifest, which stays at 226 names.

## Phase 3 status: COMPLETE (2026-06-13) — B1, B2, B3, B4, B5 done

flexscanx's `sequence/` copies are now proven behaviour-identical to their
traminex counterparts by parity tests that use traminex at **test time only**;
flexscanx still installs and runs **without** traminex. One commit per item; the
flexscanx suite stayed at or above baseline at every step. After a clean C++
rebuild the suite was **386 passed / 1 failed / 3 skipped** (re-measured
2026-06-13, Audit Phase A; **387/1/3** after Audit Phase D, below). The single
failure is the pre-existing numba-vs-NumPy environment issue —
`test_graph_contiguity_passes_use_index_false` does `from numba import njit`, and
numba refuses to import against NumPy > 1.26 ("Numba needs NumPy 1.26 or less") —
unrelated to this work. An earlier "380 passed / 7 failed" count reflected a
**stale compiled `.pyd`** masking the current C++ source; 6 of those 7 failures
clear on rebuild, leaving only the one numba/NumPy failure. Truth flowed
traminex -> flexscanx throughout.

**Audit Phase D (2026-06-13):** re-ran B1–B5 against the Phase-B-fixed oracle
(traminex `0.0.8`) — all parity green with no flexscanx drift, because the
Phase B TRATE fix touched only traminex's `distances/costs.py`, not the
`seqtrate` / `seqtransn` / `seqstatd` / `seqtab` parity targets. Two parity tests
were strengthened to be load-bearing: **B2** now exercises R `seqtrate`'s
absent-source zero-row (a state with `PA == 0` → its rate row is zeroed, not
`NaN`), and **B4**'s tolerance was tightened to `1e-12`. These are test-only
(suite `386` → `387/1/3`, the +1 being the new B2 test; same single numba/NumPy
failure). The standalone guarantee was re-proven by building the wheel and
importing/running it in a clean venv with traminex absent. No behaviour change —
flexscanx stays `0.0.2`.

- **B1 (`seq_dist_hamming`) — DONE, no behaviour change.** `norm=True` ==
  `seqdist(method="HAM")`; `norm=False` == HAM*T == unit-weight DHD; condensed
  == squareform of HAM; weighted raw == explicit-subm DHD
  (`subm[t][a,b]=w_t*(a!=b)`). Weighted `norm=True` divides by `sum(weights)` —
  a documented flexscanx convention that coincides with traminex for uniform
  weights. `tests/test_traminex_parity_b1_hamming.py`.
- **B2 (`seq_trans`) — DONE, no behaviour change; NOTICE re-attributed.**
  `normalize=True` == `seqtrate` (== `seqtransn(normalize=True)`);
  `normalize=False` == `seqtransn(normalize=False)`. NOTICE corrected:
  `seqtransn` is the per-sequence transition *count*; the rate matrix is
  `seqtrate`. `tests/test_traminex_parity_b2_trans.py`.
- **B3 (`seq_entropy`) — DONE, behaviour fixed (flexscanx 0.0.1 -> 0.0.2).**
  The only drift. `by_time=True` now matches `seqstatd`'s normalized Entropy
  row (divide by `log(K)`, K = alphabet size) via a new `normalize` parameter
  (default True; `normalize=False` keeps the old bits). `by_time=False`
  overall-diversity is unchanged. `tests/test_traminex_parity_b3_entropy.py`.
- **B4 (`seq_freq`) — DONE, no behaviour change.** Counts and proportions ==
  `seqtab`. `tests/test_traminex_parity_b4_freq.py`.
- **B5 (plots) — DONE, no behaviour change; data-level parity.** `seq_dplot`
  distribution == `seqstatd` Frequencies; `seq_fplot` widths == `seq_freq`
  (== seqtab); `seq_iplot` row order == `sequence_index_plot`. Styling differs
  by design. `tests/test_traminex_parity_b5_plots.py`.
- **B6 (`extract_sequences`) — untouched** (flexscanx's own; no parity target).

`sequence/NOTICE.txt` is now accurate per function. Proven under
`FLEXSCANX_PARITY_STRICT=1` (all assertions execute) and in a clean venv
without traminex (parity tests skip; standalone install/import/run works).
**This completes Phase 3, the final phase of MIGRATION_PLAN.md.**

## Decisions locked (2026-06-12)

- **Weighted discrepancy/tree code (A5–A8): approved.** Make traminex the
  faithful, weighted home. Audit refinement: there is no ready-made
  `dissvar`/`dissassoc`/`disscenter` function in pyweightedcluster to move — the
  faithful weighted maths is only *inline* across its clustering modules. So
  traminex **builds** the canonical primitives from the R source (cross-checked
  against that inline maths), then pyweightedcluster's inline sites are refactored
  to use them.
- **flexscanx (Section B): lightweight but identical.** flexscanx must **install
  and run without traminex** (no runtime dependency). Its `sequence/` copies are
  kept, but they must be made **byte-for-behaviour identical** to traminex's
  canonical versions and guarded by parity tests that use traminex only at
  **test time** (a development/test dependency, never an install requirement).
- **`disscenter` name clash: documented only for now.** No separate rename pass.
  It is resolved as part of executing A7 (when the weighted `disscenter` is
  promoted into traminex). **Resolved 2026-06-13** (see Phase 2 status above):
  `disscenter` = R meaning; double-centering = `double_center_matrix`.
- **Execution: a full written `MIGRATION_PLAN.md` comes first; no code changes
  until the maintainer approves it.**

---

## Pinned source-of-truth versions (updated by WeightedCluster 2.0 Phase 0)

R goldens and parity tests are generated against these exact vendored
snapshots. The WeightedCluster row was upgraded on 2026-07-17; the original
2026-06-12 1.8-1 record remains preserved as historical evidence below. If a
vendored original is ever upgraded again, regenerate the goldens and update
this table in the same commit.

| R package | Version | Release date | Vendored at |
|---|---|---|---|
| TraMineR | **2.2-13** | 2025-12-14 | `PROJECT_TRAMINEX/SOURCES/TraMineR` (archive `SOURCES/TraMineR_2.2-13.tar.gz`) |
| TraMineRextras | **0.6.9** | 2026-06-20 | `PROJECT_TRAMINEX/SOURCES/TraMineRextras` (archive `SOURCES/TraMineRextras_0.6.9.tar.gz`) |
| WeightedCluster | **2.0** | 2025-12-10 | `PROJECT_PY_WEIGHTEDCLUSTER/orig_R_pkg/WeightedCluster` (archive `orig_R_pkg/WeightedCluster_2.0.tar.gz`) |
| rflexscan | not pinned (partial reference copy, no `DESCRIPTION`; wraps FleXScan 3.1.2) | — | `PROJECT_FLEXSCANX/RFLEXSCAN_ORIGINAL` |

**Live oracle vs vendored archive (2026-09-21).** The table records the
*vendored* sources. The **live companion oracle enforced by this project's
parity tests is now TraMineR 2.2-14**. traminex's `master` still vendors
2.2-13; its `SOURCES/TraMineR_2.2-14.tar.gz` exists on the unmerged branch
`oracle-2214-remediation-20260920`. Update the TraMineR row to 2.2-14 when
that branch lands, in the same commit that regenerates any affected goldens.

Notes:

- Versions and release dates are taken from each vendored `DESCRIPTION` file
  (`Version:` and `Date/Publication:`, falling back to `Date:`).
- WeightedCluster 2.0 comes from
  `https://cran.r-project.org/src/contrib/WeightedCluster_2.0.tar.gz`; the exact
  archive SHA-256 is
  `6a03fbbc090ac92498a410faab1aeb88d686ff7a81dcfdcb97edacda44f393cc`.
  The historical readable tree is `orig_R_pkg/WeightedCluster-1.8-1`; its exact
  `docs/legacy/WeightedCluster_1.8-1.tar.gz` SHA-256 is
  `daf9987cee5ac9e706a39e72f54a64e49e9fe91b0bc3cfb5e5a94d5076b2df5b`.
- rflexscan is **not** a parity target — and, unlike TraMineR/WeightedCluster,
  **not the source of truth.** flexscanx is a new implementation that *starts
  from* rflexscan and **deliberately corrects its algorithmic/mathematical
  errors** and expands it; correctness is judged against the scan statistics'
  mathematical definitions, not against rflexscan's output (see `ECOSYSTEM.md`
  "Source of truth"). The only R-anchored parity target here is flexscanx's
  `sequence/` sub-package, tested against **traminex/TraMineR** (Section B). The
  vendored rflexscan copy is a reference/starting-point only, so the missing
  version pin is acceptable.
- Live-R goldens (where used instead of static goldens) run the active Ubuntu
  oracle, **R 4.6.2 on `x86_64-pc-linux-gnu`** (since 2026-11-02; R 4.6.1 before), with TraMineR 2.2-14 compiled
  from the vendored tarball. (Historical, superseded: goldens recorded before
  the move to the Ubuntu-only platform ran R 4.4.3 (ucrt), a Windows build.
  `AGENTS.md` and `ECOSYSTEM.md` make Ubuntu 24.04 the sole active platform, so
  that runtime is no longer current — finding F-26 of the 2026-08-24
  whole-codebase audit.) See
  `PROJECT_TRAMINEX/tests/test_r_goldens_distances.py` and
  `PROJECT_TRAMINEX/scripts/r_goldens/`.

## Source-of-truth facts established by reading the original R/C++ code

These facts (not assumptions) drive the recommendations below. Sources are the
vendored originals:
`PROJECT_TRAMINEX/SOURCES/TraMineR`,
`PROJECT_PY_WEIGHTEDCLUSTER/orig_R_pkg/WeightedCluster`,
`PROJECT_FLEXSCANX/RFLEXSCAN_ORIGINAL`.

1. **WeightedCluster depends on TraMineR.** Its `DESCRIPTION` says
   `Depends: ... TraMineR (>= 2.0-6)`. Its C/C++ `src/` contains **only**
   clustering algorithms (PAM, k-medoids, cluster quality) — **no distance,
   cost, or substitution-matrix code at all.** In R, all sequence distances
   come from TraMineR's `seqdist`. So any distance/cost code inside the Python
   `pyweightedcluster` is **not** a port of WeightedCluster — it is a re-creation
   of TraMineR functionality and belongs in traminex.

2. **Weighting originates in TraMineR, not WeightedCluster.** In the R source,
   per-sequence weights are stored as an attribute on the sequence object
   (`attr(seqdata, "weights")`, set by `seqdef(..., weights=)`). Every weighted
   computation reads that attribute and lives in TraMineR:
   - `seqtrate()` (transition rates → TRATE costs) — reads weights.
   - `dissvar()` (discrepancy / pseudo-variance) — takes `weights`.
   - `dissassoc()` (pseudo-F / pseudo-R², permutation test) — takes `weights`.
   - `disscenter()` (distance of each case to its group centre) — takes `weights`.
   - `disstree()` (regression tree on a distance matrix, weighted inertia
     splits) — takes `weights`.
   WeightedCluster's `as.seqtree.R` does **not** implement weighting itself; it
   reads `attr(seqdata, "weights")` and calls TraMineR's weighted internals
   (`dissvar`, `TraMineRInternalWeightedInertiaDist`, `dissassoc`).
   **Conclusion:** the canonical home for weighted distances, costs, discrepancy
   and trees is traminex. WeightedCluster's genuinely own weighted code is the
   **weighted PAM/k-medoids algorithm, cluster quality, fuzzy and CLARA**
   (its `src/*.cpp` and `R/kmedoids.R`, `clusterquality.R`, `fuzzyfunc.R`).

3. **rflexscan has no sequence-analysis code.** Its source contains spatial scan
   statistics only. So flexscanx's `sequence/` sub-package is a flexscanx-original
   addition, adapted from traminex (as its own `NOTICE.txt` already states). The
   source of truth for those functions is therefore TraMineR.

4. **rflexscan/FleXScan is flexscanx's starting point, not its oracle.** Unlike
   the traminex←TraMineR and pyweightedcluster←WeightedCluster relationships
   (faithful ports, R is the source of truth), flexscanx's spatial-scan core is a
   **new implementation that deliberately corrects critical algorithmic and
   mathematical errors in rflexscan/FleXScan** and may expand beyond it. Its
   acceptance test is **agreement with the scan statistics' mathematical
   definitions and flexscanx's own first-principles tests**, not reproduction of
   rflexscan's numbers. Where the two diverge and rflexscan is wrong, flexscanx's
   behaviour is the deliberate, documented correction — it is never adjusted to
   re-create an rflexscan bug.

---

## A. pyweightedcluster vs traminex

### A-distances (traminex is the better, faithful implementation → REPLACE)

| # | Local code (pyweightedcluster) | traminex equivalent | Source-of-truth check | Recommendation |
|---|---|---|---|---|
| A1 | `_seqdist.py` (HAM/OM/LCS/DHD/OMSPELL) on its own `_wc_core` | `traminex.seqdist` / `distances.pairwise`, ~18 methods, R-golden tested | WeightedCluster R has **no** distance code; distances are TraMineR's | **REPLACE** (top value) |
| A2 | `_omspell.py` | traminex OMSPELL/OMSLEN, with `test_omspell_omslen_parity.py` | same as A1 | **REPLACE** |
| A3 | `costs.py` (substitution/indel, transition-rate costs) | `traminex.distances.costs` — `trate_substitution_matrix(..., weights=)` **already weight-aware**, matching R `seqtrate(weighted=TRUE)` | TraMineR owns costs; traminex port handles weights | **REPLACE** |
| A4 | `_metrics.py` `normalize_metric` (name aliases) | traminex `_validation` normalization | small helper | **DECIDE** (fold in with A1) |

**Phase 0 harness finding — HAM normalization (recorded 2026-06-12).** Both
Python copies already length-normalize HAM and expose **no** `norm` kwarg:
traminex `seqdist(method="HAM")` and pyweightedcluster
`sequences_distance(metric="HAM")` both return `mismatches / length` (e.g.
`0.667` for 4 mismatches over length 6). That matches R
`seqdist(method="HAM", norm="maxlength")`, **not** R's default `norm="none"`
(raw `4`). Because the two copies agree with **each other**, the
old-local-vs-traminex leg of the A1 test is trivially green for HAM and proves
nothing alone — the load-bearing check is against the **R source**. So the A1
comparison must (a) pin traminex HAM to R `norm="maxlength"`, and (b) decide
explicitly whether pyweightedcluster's clustering should consume the normalized
distances or R's raw `norm="none"` — a semantics decision, not a tolerance one.
Re-confirm the same question for OM / OMspell / LCS / DHD before switching call
sites.

### A-discrepancy/trees — historical pre-Phase-2 snapshot

The following is the preserved audit state that motivated A5–A8; it is not a
description of current code. At that time, traminex's `diss*`/tree code was
**not faithful to the R source**; the faithful weighted maths exists only as
*inline* code inside pyweightedcluster's clustering modules, not as reusable
named functions. So the action is to **build** the canonical weighted primitives
in traminex (anchored on the R source), not merely to move a function:

- traminex `diss.py` defines **named but simplified** `dissvar`, `dissassoc`,
  `disscenter` (all **unweighted**, exported in the public API):
  - `dissvar(D, labels)` returns plain variance of pairwise distances, not R's
    weighted discrepancy (pseudo-variance).
  - `dissassoc(D, labels)` returns a simple `(between-within)/between` effect
    size, not R's weighted pseudo-F / pseudo-R² with permutation test.
  - `disscenter(D)` **double-centres a matrix** — a *different operation* from
    R's `disscenter` (distance of each case to its group centre). Same name,
    different meaning: **a real trap if imported blindly.**
- traminex has **two** non-faithful tree stand-ins, both unweighted and both
  sklearn-based — neither is R's weighted-inertia recursive split:
  - `disstree.py::disstree_fit` — classical MDS embedding, then a
    `DecisionTreeClassifier` (self-described "Minimal disstree analog").
  - `seqtree.py::seqtree_fit` — extracts duration/transition/summary features,
    then a `DecisionTreeClassifier` ("simple decision tree (seqtree-like)").
- **The faithful weighted logic is NOT a ready-made function anywhere.** It lives
  only *inline* inside pyweightedcluster, spread across several modules:
  `seqtree.py` (`as_seqtree`, `_weighted_pairwise_sum`), `proptree.py`
  (`seqpropclust` / `wcPropertyClustering`, `_best_split_for_variable`),
  `clustassoc.py` (`_weighted_within_and_total`), and `quality.py` (weighted PBC
  / ASW). pyweightedcluster has **no** functions named `dissvar`, `dissassoc`,
  or `disscenter` — confirmed by search. So this is not a "move the function"
  job; it is a "build the canonical primitive" job (see recommendation).

| # | Concept | Faithful version today | Recommendation |
|---|---|---|---|
| A5 | `dissvar` (weighted discrepancy / pseudo-variance) | none as a named function; the weighted-inertia maths is inline in pyweightedcluster | **BUILD in traminex** from the R source (`dissvar.R` / `WeightedInertia`), using pyweightedcluster's inline maths as a cross-check; replace traminex's simplified `dissvar` |
| A6 | `dissassoc` (weighted pseudo-F + permutation) | none as a named function | **BUILD in traminex** from `dissassoc.R`; replace traminex's simplified `dissassoc` |
| A7 | `disscenter` (distance-to-group-centre) | none as a named function (R's lives in TraMineR) | **BUILD in traminex** matching R; **rename** traminex's current matrix-centering `disscenter` to an unambiguous name with a deprecation alias |
| A8 | `seqtree` / `disstree` (weighted-inertia tree) | pyweightedcluster `as_seqtree` (faithful to R `as.seqtree`); also reused by `proptree.seqpropclust` | **BUILD the faithful tree in traminex** (porting `as_seqtree`'s weighted-inertia split), then relabel traminex's two sklearn stand-ins as the approximate variants they are |

After A5–A8 exist faithfully in traminex, refactor **all** the inline call sites
in pyweightedcluster (`seqtree.py`, `proptree.py`, `clustassoc.py`, `quality.py`)
to use the traminex primitives, so the family has one faithful copy in the
correct (R-mirrored) place. Scope note: because the logic is inline in ≥4 modules,
Phase 2's consuming-side refactor is larger than a single import swap.

### A-own (genuinely WeightedCluster's → KEEP)

| # | Local code | Why it stays | Recommendation |
|---|---|---|---|
| A9 | `kmedoids.py`, PAM, `quality.py`, `fuzzy.py`, `range.py`, CLARA (`seqclararange.py`), `boot.py` | WeightedCluster's own weighted algorithms (`src/*.cpp`, `R/kmedoids.R` etc.) | **KEEP** — never move to traminex |
| A10 | `seqnull.py` | WeightedCluster's own (`R/seqnull.R`); generates null sequences for cluster significance | **KEEP**; only its inner sequence-generation primitive could later borrow traminex `seqgen` |

## B. flexscanx vs traminex

rflexscan has no sequence code, so everything in `sequence/` is adapted from
traminex (TraMineR). The values handled are tiny (states are `+1/-1/0`).

Per the locked decision, flexscanx keeps its own copies (no runtime traminex
dependency) but each copy must be **identical in behaviour** to traminex's
version and **parity-tested against traminex at test time**.

| # | Local code (flexscanx) | traminex equivalent (parity target) | Recommendation |
|---|---|---|---|
| B1 | `sequence/distances.py` `seq_dist_hamming` | traminex Hamming (`seqdist method="HAM"`) | **KEEP-IDENTICAL** — add parity test vs traminex |
| B2 | `sequence/analysis.py` `seq_trans` (transition matrix) | `traminex.seqtransn` (named in `NOTICE.txt`) | **KEEP-IDENTICAL** — align to traminex, parity test |
| B3 | `sequence/analysis.py` `seq_entropy` | `traminex.seqstatd` (named in `NOTICE.txt`) | **KEEP-IDENTICAL** — align to traminex, parity test |
| B4 | `sequence/analysis.py` `seq_freq` | `traminex.seqtab` / `seqstatf` | **KEEP-IDENTICAL** — align to traminex, parity test |
| B5 | `sequence/plotting.py` `seq_iplot/_dplot/_fplot` | `traminex.sequence_index_plot` etc. | **KEEP-IDENTICAL** — match output conventions; visual parity check |
| B6 | `sequence/_utils.py` `extract_sequences` | none — turns scan results into sequences | **KEEP** (flexscanx's own job; no parity target) |

**flexscanx rule (decided):** flexscanx installs and runs **without** traminex.
traminex is a **test-only** dependency used to prove these copies stay identical;
if any parity test drifts, fix the flexscanx copy to match traminex (the source
of truth flows traminex → flexscanx, never the reverse).

## C. flexscanx vs pyweightedcluster — examined, no duplication found

The family premise is "flexscanx uses functions from TraMineR **and**
WeightedCluster." The first half is real (Section B). The second half is **not
realised in the code, and currently does not need to be:**

- flexscanx's clustering code (`_consensus.py`, `_stability.py`) operates on
  **spatial** clusters — sets of area IDs compared by Jaccard overlap
  (`match_clusters_jaccard`, `consensus_sweep`). That is flexscanx's own
  scan-statistic domain, **not** WeightedCluster's k-medoids/PAM clustering of
  sequences. No PAM, medoid, silhouette-on-sequences, or cluster-quality code
  from WeightedCluster is duplicated.
- flexscanx computes trajectory **distances** (`seq_dist_hamming`) but never
  **clusters** those trajectories itself, so it has no present need for
  WeightedCluster's algorithms.

**Conclusion:** there is no flexscanx↔pyweightedcluster duplicate to reconcile.
Future agents should not invent a WeightedCluster dependency in flexscanx to
satisfy the layering diagram; the "builds on weightedcluster" relationship is
latent (available if flexscanx ever clusters trajectories) but not active today.

## D. pyweightedcluster modules reviewed and confirmed WeightedCluster-own

For exhaustivity, these were checked against the R source and are **not** traminex
territory (they port WeightedCluster's own R files, no action needed):
`aggregate.py` (aggregatecases.R), `clustassoc.py` (clustassoc.R — covariate↔
clustering association, distinct from TraMineR's `dissassoc`), `cmpcluster.py`,
`range.py` (kmedrange/clustrange), `seqclustname.py`, `proptree.py`
(propclustering.R-facing structural adapter; tree growth is now traminex-owned),
`fuzzy.py`, `boot.py`. (`_labels.py`, formerly listed here, was removed at
`419d045`; 0-/1-based conversion now lives with its consumers — recorded by
the 2026-09-02 audit, finding WC-090209.)

---

## Historical recommended order (completed)

1. **A1 + A2 + A3** — point pyweightedcluster's clustering at traminex's
   R-validated, weight-aware distances and costs. Biggest correctness gain.
2. **A5–A8 (BUILD in traminex)** — build faithful weighted discrepancy and tree
   primitives in traminex from the R source (cross-checked against
   pyweightedcluster's inline maths); replace traminex's simplified `diss*` and
   relabel its two sklearn tree stand-ins; resolve the `disscenter` name clash as
   part of A7; then refactor pyweightedcluster's inline call sites to use them.
3. **B1–B5 (KEEP-IDENTICAL)** — align flexscanx's `sequence/` copies to
   traminex's behaviour and add parity tests (traminex used at test time only;
   flexscanx still installs without it).
4. **A4** — minor; fold in with A1.

Full sequencing, tests, and rollback are in `MIGRATION_PLAN.md`.

## Packaging reality (affects Phases 0–2)

traminex is **not published on PyPI**; it is a local project. "Add traminex as a
dependency of pyweightedcluster" therefore means a local/path or editable install
(and, for flexscanx, a local test-only install). Establishing how the three
projects reference each other locally (path dependency, editable install, or a
private index) is a Phase 0 prerequisite, not an afterthought.

**Resolved in Phase 0 (2026-06-12):** editable installs into the shared
`datasci` env, in dependency order (traminex first); consumers declare
bare-name `traminex` deps that resolve against the pre-installed local copy. Full
rationale and the dependency-confusion caveat are in `MIGRATION_PLAN.md`
Phase 0.3. The parity harnesses added in Phase 0.1 live at
`PROJECT_TRAMINEX/tests/r_goldens_utils.py`,
`PROJECT_PY_WEIGHTEDCLUSTER/tests/traminex_parity_utils.py`, and
`PROJECT_FLEXSCANX/tests/traminex_parity_utils.py`.

## Migration method (when the maintainer approves an item)

For each item, in order, never as a big batch:

1. **Anchor on the R/C++ source.** The acceptance test is agreement with the
   original package's output, not with either Python copy.
2. Add traminex as a declared dependency of the consuming project (once).
3. Write a comparison test: same inputs through the **old local** function, the
   **traminex** function, and (where feasible) the **R** reference; assert they
   match within tolerance.
4. Only when green, switch call sites to traminex and delete the local copy.
5. Keep each project's test suite green at every step. One item per commit.

This enforces "retain the best implementation" by evidence: the winner is the one
that matches the R/C++ source of truth, regardless of which project it lives in
today — which is why A1–A3 move *into* pyweightedcluster's call sites from
traminex, while A5–A8 move *out of* pyweightedcluster up into traminex.

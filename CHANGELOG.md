# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-11-08

A patch release, by the maintainer's decision of 2026-11-08. It requires traminex 0.2.3, which the import-time
commitment checks.

### Added
- **README: an "Install a release" section**, for visitors: one command that installs the traminex and
  pyweightedcluster release wheels from GitHub together, so that pip never looks for traminex on PyPI; the
  source package for other platforms.
- **Consumer regressions for traminex 0.2.3** (`tests/test_traminex_0_2_3_consumer_contracts.py`): the two
  changes of the delta that reach `seqclustname`, below.

### Changed
- **Re-pinned to traminex 0.2.3** (`2cbb3472736d0ccce32a291241983a12b8921122`, 2026-11-08). Three of the 83 committed
  files changed, so the digest is now `4a0a7a647903bc6d…`. Through `seqclustname`: complex-valued states are
  labelled as R's `as.character` writes them (`1+1i`, formerly `(1+1j)`), and rows given as matrices of unequal
  widths are refused. `DUPLICATES.md` records the review.

## [0.1.0] - 2026-11-08

The first published release, by the maintainer's decision of 2026-11-08 (0.0.3 was never released; the
project's initial release, 0.0.1, was never published). It requires traminex 0.2.2. The published source
distribution is lean: see the first entry below.

### Changed
- **The published tree stands alone**: six links of `AGENTS.md`, `docs/API_INDEX.md`, `docs/ARCHITECTURE.md` and
  `docs/R_MIGRATION_GUIDE.md` into left-out records became code-form mentions, and three tests that read files the
  source distribution has always left out (`orig_R_pkg/`, `benchmarks/results/`, `.github/`) skip when the file is
  absent, naming it. In this repository the files exist and the three tests run.
- **The published package is lean** (maintainer's decision of 2026-11-08, before the first public release): the
  source distribution leaves out the maintainers' records (`docs/maintainers/`, `docs/SUPERVISOR_NOTES.md`), the
  Bayesian B1 validation evidence (`tests/_bayesian_rarcat_b1_*.py`, `tests/fixtures/bayesian_rarcat_b1_*`) and
  the 16 test modules that read them; they stay in this repository, where the full suite runs. The
  files that stay no longer name this machine's folders (home paths written `~/`, the family folders by name,
  Windows paths from `%USERPROFILE%`), and the README names the left-out records instead of linking to them.
- **Re-pinned to traminex 0.2.2** (`2ed89d77a75d67a4ac319c05da463a1758244503`, 2026-11-08), traminex's patch release that removes the maintainer's
  local paths from its published files: a version-only move from `7bebb7a4`; the 83-file inventory and its digest are unchanged.
- **Re-pinned to traminex 0.2.1** (`7bebb7a4026f7f8037e1e196d005f4cb5ffa03a4`, 2026-11-08), traminex's patch release: a version-only
  move from `928186e8`; the 83-file inventory and its digest are unchanged.
- **Re-pinned to traminex `928186e80ff263f4a5dca8270aca8407a20a12a9`** (traminex 0.2.0 with its fixes of
  2026-11-07; owner decision D6 of the supervision repository, carried out at the owner's request "fix all
  known issues"), by the reviewed procedure: an 83-file commitment, aggregate SHA-256
  `e2ccb374b64010843e58b334299f9902edada5e5056128567afa728bbe39aabb`, and the floor `traminex>=0.2.0`, which
  is still insufficient on its own. `.venv` imports the traminex checkout again, as `AGENTS.md` requires,
  instead of the 6e76440b copy installed on 2026-10-27. `DUPLICATES.md` records the delta review.

### Fixed
- **Property-tree split schedule on exact ties** (supervision repository, 2026-11-02 measurement).
  `proptree.py` multiplied `R2 * vardis * n` left to right; R computes `R2 * (vardis * n)`
  (`propclustering.R:195-196`). With a traminex whose `vardis` equals R's to the last bit, an exact
  tie rounded apart and selected the other split, so `dtcut` gave another partition. R's order is used.
- **HAM and DHD with infinite costs for unobserved states** keep working with the current traminex,
  whose explicit-cost DHD refuses any non-finite cost, as R's DHD does. Such entries never enter a
  distance, so they are handed to traminex as 0; a test shows the distances equal those of any finite
  value there.
- **HNCdd ignores `m`, as R's `wfcmdd` does** (2026-09-21 targeted audit, M-3). R sets `m <- 1` in its
  HNCdd branch, so the medoid search and the functional use `u^1`; this port kept the caller's `m`, and
  from a fuzzy initial membership chose other medoids. `fcmdd(method="HNCdd")` accepts any finite `m`
  and uses 1; ten live-R comparisons from Dirichlet initialisations pin it.
- **Equal Bayesian root seeds are refused** (2026-09-21 targeted audit, L-3). Both replay domains draw
  their first stream from `SeedSequence(root).spawn(2)[0]`, so one integer passed as both roots gave the
  bootstrap weights and the prior-predictive draws the same bits. `bayes_rarcat` now raises before any
  fitting; the derivation is unchanged, so every replay with distinct roots is unchanged.

### Changed
- **`clustassoc` gives NaN where a clustering is collinear with the covariate**, as traminex's
  `dissmfacw` does for a term that adds no rank; WeightedCluster reports a finite value of an arbitrary
  QR direction there. A deliberate departure (option (a) of the family's open item, 2026-11-07); the two
  tests compare the other rows with R and assert R's finite value and this port's NaN for that row.
- **The categorical BIC's documented agreement with R is restated from measurement** (2026-09-21
  targeted audit, L-5). The value is the converged likelihood; R's `multinom` stops earlier, so R's BIC
  is higher by up to 3.2e-6, or 2.4e-3 when a cluster lacks a category (60 random cases); never lower.
  The earlier `1e-7` and `2e-4` bounds held for the fixtures only. A random-domain test pins it.
- Tests: the asymmetric-cost warning is matched with R's text, "Non-symmetric substitution cost
  matrix!", which traminex now gives; distances between two sets are compared with R's
  `refseq=list(a, b)`, which WeightedCluster's CLARA uses, rather than with the full matrix's cross
  block; the REML prototype test records the mismatch profile measured with MKL 2026.1; and the legend
  branch of the weighted fuzzy index panel is tested.
- **The live R reference moved to R 4.6.2 and lme4 2.1.0** (2026-11-02, owner decision for the
  family, recorded in the supervision repository's `docs/DECISIONS.md`). The version assertions in
  `tests/test_r_parity_rarcat.py` and `tests/test_rarcat_conformance_scaffolding.py` follow. The
  package itself does not use lme4; WeightedCluster's `rarcat` does, so 50 of the 2,361 tests need it,
  in those two modules and `tests/test_rarcat_reml_architecture.py` (measured with lme4 hidden from
  R). Under the new versions those modules give 67 passed and 1 failed in normal and strict mode; the
  failure is the known `test_profiled_prototypes_expose_remaining_architecture_gate_failures`, whose
  failing set is the same under lme4 2.0.6. `tests/rarcat_reml_architecture.py` still ports lme4
  2.0-6's convergence diagnostics; that port was not compared with 2.1.0.
- **The exact live companion oracle moved from TraMineR 2.2-13 to TraMineR
  2.2-14** (2026-09-21, maintainer decision: 2.2-14 is now canonical across the
  project family). `EXPECTED_TRAMINER_VERSION` in `tests/r_parity_utils.py` is
  the single enforcement point; `tests/test_oracle_provenance.py` now pins
  2.2-14 as the only accepted target and rejects 2.2-13 alongside every other
  release. The installed R library carries 2.2.14, so the 444 version-gated
  xfails become executable live-R checks again and the repository's strict
  zero-skip CI contract becomes satisfiable in this environment.
  `tests/fixtures/weightedcluster_2_0_namespace.json` was re-captured with
  `scripts/capture_weightedcluster_2_0_namespace.py --write` so the fixture,
  the pin, and the live environment agree; WeightedCluster itself is unchanged
  at 2.0, so only the recorded oracle stamp moved.
  Upstream evidence that the move is numerically inert: traminex's
  `69839eec` records that TraMineR's `src/` is byte-identical between the two
  releases and that 23 of 23 recomputed reference quantities are byte-identical,
  with one real upstream fix (`seqcost` no longer downgrades `with.missing`).
  An earlier version of this entry said this project never reaches that fix
  because it does not call `seqcost`; that was false (2026-09-21 targeted
  audit, L-6). `seqnullcqi` forwards `seqdist_args` to traminex, which calls
  `seqcost` with `with_missing=True` for `sm` values such as `CONSTANT` and
  `INDELSLOG` (not `TRATE`). The fix only alters a result when the data carry
  no missing value but an external reference does, because TraMineR's
  `seqdist` resets `with.missing` to FALSE otherwise (`seqdist.R:111-116`,
  identical in both releases). A cross-release differential over 73 R entries
  covering this project's ported surface found 63 identical and 10 differing,
  all explained: 5 are that fix resizing the substitution-cost matrix that
  direct `seqcost` calls return, and 5 are TraMineR's version stamp on
  sequence objects (`audit/targeted-2026-09-21/outputs/a1_compare_213_214.txt`).
  Current-oracle statements were updated in `README.md`, `ECOSYSTEM.md`,
  `OPEN_QUESTIONS_FOR_AUDIT.md`, `DUPLICATES.md`, `orig_R_pkg/README.md`,
  `docs/legacy/README.md`, `docs/maintainers/README.md`, the parity ledger, and
  `src/weightedcluster/_seqdist.py`. Dated historical records were deliberately
  left at 2.2-13.
- **Documented the live-oracle / vendored-archive distinction.** traminex's
  `master` still vendors `SOURCES/TraMineR_2.2-13.tar.gz`; its 2.2-14 archive
  exists only on the unmerged branch `oracle-2214-remediation-20260920`. The
  pinned-versions table in `DUPLICATES.md` therefore keeps its factually
  correct 2.2-13 row and carries a note to update it when that branch lands.

### Added
- **Bayesian RARCAT B5 completed and exposed publicly.** Independent holdout,
  immutable-evidence recalculation, numerical, diagnostic, mixture,
  failure-policy, persistence, ordinary, strict-R, coverage, static,
  distribution, and clean-install gates passed. The stable public surface is
  `weightedcluster.bayes_rarcat` with frozen `weightedcluster.BayesRARCATResult`
  and the bounded `bayesian` extra. The R-compatible name `rarcat` remains
  intentionally absent, so the parity ledger remains 18 strong, 0 partial,
  0 unverified, and 1 missing. **B5 complete; public `bayes_rarcat` available;
  R `rarcat` remains missing.** See
  `docs/maintainers/BAYESIAN_RARCAT_B5_VALIDATION_REPORT.md` and
  `docs/maintainers/BAYESIAN_RARCAT_B5_RELEASE_REPORT.md`.

### Fixed
- **A Python-versus-R value mismatch no longer passes the ordinary test run**
  (2026-09-21 targeted audit, L-2). Nine value comparators in
  `tests/test_r_parity_phase0.py` and `tests/test_r_parity_cmpcluster.py`
  reported a genuine divergence through `fail_or_xfail`, whose purpose is to
  xfail an unobtainable oracle, so an ordinary `pytest -q` recorded it as
  xfailed and exited 0; only the strict gate caught it. They now call the new
  `fail_parity`, which fails in every mode, and `fail_or_xfail` is left to its
  one legitimate caller, the oracle version gate. Under the audit's own
  in-memory perturbation of `wcClusterQuality`/`wcSilhouetteObs` output by a
  relative 1e-6, the ordinary run went from `8 xfailed`, exit 0, to `8 failed`,
  exit 1; unperturbed, all 20 tests pass. A test parses every test module and
  requires the version gate to be the only caller of `fail_or_xfail`. Because
  `XFailed` subclasses `Failed`, `pytest.raises(Failed)` alone also accepts an
  expected failure, so the new behavioural test requires the exact type.
- **`dtlabels` renders numeric thresholds as R does** (2026-09-21 targeted
  audit, L-8). R uses `format(x, digits = getOption("digits") - 2)`
  (`R/propclustering.R:254-255`); Python used `format(x, ".5g")`. Against live
  R on 5,474 values the old code differed on 1,644 -- 1,636 of them at or above
  1e5, where `1200000` became `1.2e+06`, and 8 near 1e-4, where R's `1e-04`
  became `0.0001`. The new `_r_format_number` reproduces R's rule (round to 5
  significant digits, drop trailing zeros, print fixed notation unless it is
  wider than scientific) and matches R on all 5,474, including NaN, both
  infinities and signed zero. The live-R `dtlabels` parity test never used a
  break of 1e5 or more, which is why it stayed green.
- **One identity rule for the Bayesian mixture step** (2026-09-21 targeted
  audit, L-4). `_bayesian_rarcat_mixture` carried its own `_same_identity`,
  `bool(left == right)`, which raised on `pd.NA` and on arrays and accepted a
  one-element array as identical, where `bayesian.py`'s version fails closed.
  The rule is now `bayesian._same_identity_value`, and the mixture module
  imports it and `_same_identity_sequence` instead of redefining them. Ordinary
  scalar labels behave exactly as before, including the exact-type rule; 114
  mixture, storage and public-facade tests pass unchanged.
- **The PAMonce swap tolerance is pinned across Python, C++ and R** (2026-09-21
  targeted audit, L-1). `cpp/wc_core.cpp` must carry R's `WEIGHTED_CLUST_TOL`
  as a literal, and nothing checked it matched `_constants.PAMONCE_SWAP_TOLERANCE`:
  on integer-valued, tie-heavy data every swap delta is an integer or a half, so
  no behavioural fixture separates -1e-10 from -1e-3. A test now reads the
  compiled kernel's comparison and R's `#define` and requires both to equal the
  Python owner; drifting either the C++ literal or the Python constant turns it
  red.
- **`seqclararange` crisp CLARA now breaks distance ties in R's medoid order**
  (2026-09-21 targeted audit, M-1). R orders the medoid columns by first
  appearance in the PAM membership, `mysample[ac2$aggIndex[unique(clustering)]]`
  (`R/clara_clustrange.R:127`), and assigns each case to the first minimum
  (`which.min`, `:193`). Python sorted those columns with `np.unique`, so a case
  equidistant from two medoids -- the normal case for integer-valued sequence
  dissimilarities -- could join a different cluster than in R. The returned
  `clustering`, the `DB` criterion, and therefore the iteration `criteria="db"`
  selects could all differ from R. The column order now follows first
  appearance. Verified against live R on the audit's own seeded inputs: public
  partitions identical in 90/90 comparisons (previously 38/90) with 0/90 DB
  divergences (previously 51/90), and across five prescribed iterations the
  selected iteration, medoids and partition match R in 150/150. The other
  criteria, the medoid sets and the aggregation were already identical and are
  unchanged. `test_cluster_crisp_breaks_distance_ties_in_r_medoid_order` pins
  the rule on a genuine tie and fails if the sort is reinstated.
- **`wcFuzzyRange` refuses the NCdd and HNCdd noise variants up front**
  (2026-09-21 targeted audit, M-2). The docstring advertised them, but both
  failed on every call deep in the quality step with an unrelated message,
  `medoids must have length equal to membership.shape[1]`: they return a
  K+1-column membership that `wcFuzzyQuality` cannot take. No R rule defines
  FPC, FPE or XBf on a noise column -- R has no `wcFuzzyRange`, and its CLARA
  noise branch keeps the column for `mean_diss` and PBM but drops it for DB and
  AMS (`R/clara_clustrange.R:179-189`) -- so choosing a treatment here would
  invent semantics. They now raise `NotImplementedError` naming `fcmdd`, which
  runs both variants directly. No working call is affected, since both
  previously always failed; FCMdd and PCMdd are unchanged, and `seqclararange`'s
  noise method, which calls `fcmdd` itself, is unaffected.
- **Documentation named TraMineR 2.2-13 as the live oracle after the move to
  2.2-14** (2026-09-21 targeted audit, L-7), in `docs/superpowers/CONTINUE.md`
  and in a self-reference inside this Unreleased section; a dated line in
  `OPEN_QUESTIONS_FOR_AUDIT.md` is reworded to the past tense. The oracle entry
  above also carried a false justification (L-6), corrected in place.
  `test_every_current_truth_document_names_the_enforced_traminer_oracle` now
  reads the version from `EXPECTED_TRAMINER_VERSION`, so the next move turns it
  red until every current-truth document agrees.
- **The stable silhouette branch keeps R's `DBL_MAX` sentinel when every
  other cluster has zero mass** (2026-09-07, finding WC-090307, found while
  verifying WC-090304). R leaves `b_ik` at `DBL_MAX` for such observations,
  so each positive-weight observation gets silhouette 1 and the aggregate
  ASWw is 1; live R returns that at ordinary and subnormal scales alike. The
  2026-08-24 F-05 fix restored the sentinel only in the ordinary float branch
  of `wcSilhouetteObs`; the exact stable branch that every tier takes on
  range-risk inputs returned NaN, so `wcSilhouetteObs`, `wcClusterQuality`,
  and all four kernels flipped from 1 to NaN under a positive rescaling of
  the distances. The stable branch now substitutes the same sentinel; a
  live-R-pinned regression test covers both scales through the public
  functions and the pure kernels.
- **The permissive quality kernel now takes the exact range-risk route on
  validated-domain inputs, so the publicly selected pure tier agrees with the
  compiled, Numba, and validated kernels on extreme-range distances and
  weights** (2026-09-07, finding WC-090304 of the 2026-09-03 cycle, closed
  under the maintainer's instruction to fix the pending known issues). The
  kernel kept R's float arithmetic on every input, so on distances mixing
  binary64 extremes (for example `1e-12` beside `1e12`) its CH/R2/CHsq/R2sq
  and PBC lost cancellation, and on subnormal distances or weights its
  ASW/ASWw were pushed toward 1 or NaN, where the other three tiers return
  the deliberately corrected values (the R-11/R-13 policy). On finite
  non-negative distances and weights the kernel now routes range-risk
  ordered-pair statistics to `_exact_pair_rank_statistics` and range-risk
  silhouettes to the stable exact accumulation shared with `wcSilhouetteObs`,
  aggregated by the same helper the validated tiers use. Ordinary inputs never
  enter that route, so the bit-identical R mirror the bootstrap replicate path
  relies on is unchanged there; on range-risk replicates the bootstrap now
  returns the corrected statistics rather than R's range-lost float values,
  which is the consequence of the contract choice recorded in
  `docs/maintainers/TECHNICAL_DEBT.md`. The R-permissive domain (negative or
  non-finite weights, non-finite distances) keeps R's arithmetic throughout,
  and the finite below-one-cluster ASW values (finding F-19) are still the
  unguarded R formula, now evaluated exactly where range risk is detected.
  The 240-case cross-tier corpus shows zero divergences across all four tiers
  after the change; regression tests pin the three smallest divergent corpus
  cases and the public pure-versus-compiled answer.
- **The exact range-risk quality path divides by a zero within-cluster
  dispersion the way R's kernel does** (2026-09-06, finding WC-090306).
  Live R returns `CH = CHsq = Inf` for perfectly separated clusters, and so
  do the float tiers on ordinary scales, but `_exact_pair_rank_statistics`
  guarded `ss_res == 0` into NaN, so a range-risk input with perfect clusters
  returned NaN from every tier that delegates to it, including the public
  compiled route. Zero denominators now follow binary64 division (`+/-Inf`,
  `0/0 = NaN`) while nonzero quotients stay exact; a regression test pins the
  ordinary, exact-path, and public range-risk values against live R.
- **The Numba and validated pure quality tiers now propagate a genuine
  0/0 observation silhouette into ASW/ASWw like live R, the compiled kernel,
  and the permissive kernel** (2026-09-03 audit-and-fix cycle, finding
  WC-090303). An observation at distance zero from every member of another
  cluster with an own-cluster mean of zero — for example an exact duplicate
  sequence assigned to a different cluster — has silhouette 0/0 = NaN, and
  R's kernel makes the overall and per-cluster aggregates NaN. The shared
  aggregation helper masked non-finite observation silhouettes instead, so
  those two tiers returned finite aggregates, the public answer depended on
  which backend was installed, and a common positive rescaling into the
  range-risk path flipped the public result from NaN to finite. The helper
  now propagates NaN across every positive-weight observation; live-R-pinned
  tier-parity and scale-invariance regression tests cover the family.
- **`seqnullcqi`'s workload warning and complexity claims now describe the
  fastcluster-port reclustering it actually runs** (2026-09-03 audit-and-fix
  cycle, finding WC-090302). The docstring, the runtime warning, and the
  `docs/API_INDEX.md` / `docs/R_MIGRATION_GUIDE.md` rows still claimed the
  pre-F-01 implementation's cubic Lance–Williams cost; since the 2026-08-24
  reroute the Ward.D reclustering is the quadratic-time
  nearest-neighbour-chain port. The workload heuristic now scales with
  `R * N**2`, anchored so N=100 with R=100 still warns; away from that point
  the warning boundary follows the corrected cost model, so mid-N
  few-replicate workloads the cubic model over-flagged fall silent and
  small-N many-replicate workloads of equal quadratic cost now warn. The
  pinning contract test asserts the corrected wording, the model, and the
  corrected rows of the two user documents.
- **The permissive pure quality kernel no longer corrupts ASW/ASWw when a
  zero-weight row carries an extreme distance** (2026-09-03 audit-and-fix
  cycle, finding WC-090301 — the same scaling-domain family as WC-090202).
  `cluster_quality_r_permissive` decided and applied its silhouette-loop
  distance scaling on the whole matrix, so a far distance carried by a
  zero-weight observation collapsed every active contribution to zero and
  returned ASW/ASWw = NaN where the compiled, Numba, and validated pure
  kernels return the R value. The scale decision now covers exactly the
  block the loop accumulates (positive-weight rows against nonzero-weight
  columns). Publicly reachable through `wcClusterQuality` when the compiled
  core and Numba are unavailable, and through the bootstrap replicate path;
  ordinary inputs take a value-identical path. Two tier-parity regression
  tests pin the adversarial configuration directly and through the public
  pure branch.
- **The 2026-09-02 read-only audit's twelve confirmed defects are closed**
  (report: `archive/audits/2026/AUDIT_REPORT_2026-09-02_WHOLE_CODEBASE_READONLY.md`).
  Numerical/runtime: the Numba cluster-quality tier no longer rescales by the
  whole-matrix maximum, so a zero-weight observation carrying a far distance
  cannot corrupt HGSD/PBC/CH/R2 relative to the compiled and pure kernels
  (WC-090202); `wcKMedoids` no longer crashes on valid degenerate input — the
  exact-objective helpers are deduplicated to the single guarded `_fallback`
  implementation (WC-090203). Contracts: the capability registry names the
  publicly selected pure quality implementation and the real conditional
  medoid selection (WC-090204); the checkable Bayesian envelope caps (n, K, B)
  fail before the B2 bootstrap (WC-090211); the two public Bayesian seeds
  share one strict integer gate (WC-090212). Documentation and truth
  surfaces: the README backend table, ARCHITECTURE module map, DUPLICATES
  inventory, `_validation` docstring, coverage ledger, current-cycle
  verification manifest, archive index, and the CLAUDE.md graph-snapshot pin
  are corrected and re-measured (WC-090201/05/07/09/10); the coverage gate
  now omits only the untraceable `@njit` bodies instead of all of
  `_numba.py` (WC-090208). Every code fix carries a red-green regression
  test; seven tests were added (2,339 -> 2,346 collected).
- **The confirmed 2026-08-31 audit validity defects are remediated without a
  production-algorithm change.** Namespace capture now enforces the resolved
  mandated-interpreter contract without assuming every test runner has its
  literal path. Historical RARCAT conformance tolerances and state assertions
  cover the measured MKL and OpenBLAS profiles while preserving the live-R
  numerical contracts and raw provider-specific diagnostics. CI installs the
  complete development/Numba/strict dependency sets, provisions R and native
  strict prerequisites, and restores temporary backend-probe mutations before
  collecting the complete suite.
- **The corrected 2026-08-27 whole-codebase audit is remediated.** L-01 and
  F-01 through F-12 have evidence-backed
  FIXED dispositions in
  `docs/maintainers/WHOLE_CODEBASE_REMEDIATION_2026_08_27_REPORT.md`. The
  cross-cutting pass and final gates also fixed nineteen related numerical and
  evidence-integrity defects: compiled PBC no
  longer replaces an exact positive pair mass with `1e-300`; Python PBM no
  longer replaces an exact positive within-cluster dispersion with `1e-12`;
  normalized one-cluster crispness is undefined like installed R; and
  fuzzy-CLARA full-data membership uses every valid positive `m - 1` through
  the stable exact-positive inverse-distance path. Fuzzy partition entropy now
  uses every exact positive membership with a masked zero contribution;
  flexible-beta linkage orders every distinct dissimilarity exactly and uses
  deterministic ordering only for exact ties; and optional classical-MDS Ward
  embedding normalizes before squaring and uses a scale-relative
  positive-spectrum cutoff. Fuzzy workflows now forward the original validated
  finite weights unchanged; each stage uses local exact/relative accumulators,
  so no shared normalized vector can erase a positive ratio. Crisp DB/XB
  and fuzzy XBf/AMS use stable finite-distance ratios; PBM alone restores its
  documented distance-squared units at the IEEE boundary; and compiled, Numba,
  and pure crisp-quality kernels preserve raw rank keys/local positives while
  stabilizing subnormal squares, finite squared reductions, and high/low
  pair-mass aggregates, including fourth-order and mixed-exponent cancellation.
  The actual public pure fallback and internal bootstrap route now retain raw
  rank keys before their separate moment/silhouette scaling, so a minimum
  positive distance remains distinct from zero even beside `1e308`.
  Ordinary finite HGSD weights now retain backend accumulation order;
  nonfinite R-permissive weight/distance paths no longer enter finite-only exact
  conversion; the public plotting wrapper checks this package's Matplotlib
  extra even after traminex wrapper caching; and the current manifest/archive
  surfaces reconcile their prose, live collection arithmetic, and authority.
  Representable absolute CH/CHsq magnitudes are restored. Crisp DB/XB, public
  PAM/PAMonce/KMedoids BUILD/swap/update/run-ranking objectives, fuzzy medoids/XBf, and PBM now retain local
  positive distance/membership exponents until ratios form, even when an
  unrelated far entry spans the binary64 range. SeqCLARA DB/AMS assembly
  retains membership/confidence and weight factors,
  while its crisp/fuzzy XB/PBM assembly retains a subnormal distance-functional
  component through the final cancellation. Exported observation
  silhouettes and FCMdd candidate selection avoid finite large-distance,
  exact-product, partial-subnormal, near-tie, local-subnormal, and common-weight-
  scale range loss while preserving exact ordinary output bits, ASW's absolute
  weighted-size correction, and ASWw's relative-weight semantics. This is a
  deliberate correction of installed R 2.0's inherited wrong result. The report also
  corrects Task 6 evidence provenance: the old aggregate files are labeled as
  handwritten summaries, while fresh current GREEN and reconstructed-baseline
  RED commands have raw output and metadata sidecars, with a durable
  repository-relative archive/overlay/import manifest for the baseline. The report retains U-01
  hosted/cross-architecture and U-02 archived-kernel limitations and does not
  claim absence of undiscovered defects. Fresh whole-tree final verification is
  recorded separately at handoff.
- **Delegated traminex behavior now has an enforced source commitment.** The
  bare metadata floor `traminex>=0.0.8` is insufficient because different
  behavior-bearing trees used that version. A packaged manifest commits the 79
  sorted Python sources of reviewed revision
  `1ebb46cd94321feb5bed59c50a864f405a2f4c59` by aggregate digest. A stdlib-only
  verifier runs before delegated imports and rejects absent distributions or
  packages, version drift, unsafe roots, symlinks, missing/extra source, and
  same-version content changes; only successful verification is cached.
  Descriptor and pathname identities plus a final reinventory detect source
  replacement overlapping the check. This is a point-in-time startup contract:
  the installed tree is assumed stable after success, and post-check mutation
  requires process restart/reverification.
  Editable CI and wheel smoke call the verifier explicitly, distributions ship
  the JSON, and maintainer documentation defines the reviewed manifest update
  procedure. This is local compatibility enforcement, not hosted CI/aarch64
  evidence and not an external traminex release coordinate.
- **Comparison K ranges and property-tree execution controls now fail at their
  public boundaries.** `wcCmpCluster` validates the canonical
  `2..maxcluster` range against `N` before DIANA, hierarchy, or PAM dispatch;
  `maxcluster=N` remains valid and `maxcluster>N` can no longer leave DIANA
  result K values inconsistent with family parameters. `wcPropertyClustering`
  and `seqpropclust` now expose keyword-only `perm_n_jobs=None`,
  `n_jobs=None`, and `progressbar=False`. `None` resolves to serial execution,
  non-`None` aliases conflict, invalid types fail eagerly, and unavailable
  worker counts/progress raise `NotImplementedError` instead of remaining
  opaque compatibility keywords. The migration guide now limits the omitted
  second-fanny candidate and its objective bound to R's public `fuzzy`/`noise`
  branches; HNCdd and PCMdd are identified as separately tested Python
  extensions.
- **Bayesian execution controls now fail before expensive fitting, and private
  bundle publication is truly no-replace on supported Linux filesystems.** A
  shared validator rejects non-boolean `parallel` and boolean, nonintegral, or
  nonpositive `n_jobs` before B2 partition construction or B3 PyMC fitting,
  while remaining enforced for direct B4 callers. Private result directories
  now use same-filesystem Linux `renameat2(RENAME_NOREPLACE)`, so an empty
  directory or symlink created in the final publication window cannot be
  replaced. Published bundles are revalidated. When the recorded identity
  is observed at a writer-managed pathname after failure, that existing path is
  reported on the original exception with an explicit concurrent-identity
  caveat. Failure handling performs no rename, `unlink`, `rmdir`, deletion, or
  overwrite, so a raced unrelated entry is never relocated. Missing primitives and
  explicit unsupported-operation errnos fail closed with `NotImplementedError`;
  operation errors such as invalid rename topology remain `OSError`.
- **Fuzzy index plots now preserve finite case-weight geometry and fail closed
  for nonfinite fuzzy weights.** `fuzzyseqplot(type="i"/"I")` uses public
  traminex row selection, palette, tick, label, and legend facilities with a
  narrow local variable-height cell adapter, so weights such as `0.1`, `2`, and
  `10` produce proportional row heights while unit weights retain ordinary
  geometry. The complete sidecar is validated before full, systematic, or
  random weighted selection, closing the path that silently substituted unit
  sampling probabilities for an infinite membership weight. Explicit index
  `weighted=False` remains valid and uses unit geometry;
  `members_weighted=False` retains its separate R membership-only meaning.
  Powered sidecars are checked in full before threshold filtering, invalid
  `weighted` controls fail before selection, missing/void translucency is
  preblended toward white, and exceptional tiny-weight display normalization
  preserves proportional geometry within the representable axis range while
  wider dynamic ranges and unrepresentable per-panel totals fail pre-figure.
- **`wcCmpCluster` now reproduces `fastcluster::hclust`, the `hclust` that
  WeightedCluster actually runs.** `orig_R_pkg/WeightedCluster/NAMESPACE`
  contains `importFrom("fastcluster", "hclust")`, so R's `wcCmpCluster` resolves
  `hclust` to fastcluster's implementation; this package reproduced
  `stats::hclust` instead. The two break equal-distance merges differently, and
  for `centroid` they disagree on the merge heights as well, so **every method
  returned a different partition from R whenever the dissimilarities contained
  ties** — the ordinary case for integer-valued sequence dissimilarities, not an
  edge case. All eight `hclust` methods, weighted and unweighted, now take one
  route through `weightedcluster._fastcluster_linkage`, validated against live
  `fastcluster::hclust` on 1,448 comparisons of merge order and merge heights
  with zero mismatches. `seqnullcqi`'s Ward.D reclustering moved with them.
  **Numbers change:** across 288 measured cells, partition agreement with R rose
  from 246/288 to 288/288, each of the ten quality statistics changed in 40 to 42
  cells, and `summary_clustrangefamily`'s top-ranked method-and-k disagreed with
  R's own summary in 15 of 120 rows before the fix and in none after it.
- **`seqclararange` seeds each CLARA iteration the way R does.** R builds a Ward
  hierarchy on the subsample and seeds from it; this package used random-restart
  PAM for the crisp path and a random-restart crisp fit for the fuzzy path, so
  for a given subsample R's clustering was deterministic while Python's was a
  search. Both paths now build R's Ward hierarchy, so the selected iteration,
  its medoids and its reported criteria follow R. See the migration guide for
  the one residual: R's second FCMdd candidate from `cluster::fanny` is not
  implemented.
- **`wcSilhouetteObs` no longer disagrees with the rest of the package on an
  asymmetric matrix.** It used the full matrix where the other three entry
  points raised, and returned different values with no error and no warning. All
  four now reduce to the lower triangle, as R's `as.dist` reads a matrix.
- **One distance-matrix contract across `wcClusterQuality`, `wcSilhouetteObs`,
  `as_clustrange` and `wcKMedoids`.** Only the lower triangle is data, as R's
  `as.dist` reads it. The three entry points that used to raise on an asymmetric
  or non-zero-diagonal matrix now compute R's value, `as_clustrange` accepts the
  negative distances `wcClusterQuality` already accepted, and a `UserWarning`
  fires when the discarded half disagrees. Note that R itself is not
  self-consistent on a *non-symmetric bare matrix* — its silhouette loop reads
  both triangles while its ordered-pair statistics read one — so Python follows
  R's `as.dist` reading there rather than R's matrix reading. The migration
  guide has the measured table. `wcKMedoids` keeps one documented
  narrowing: its compiled medoid search requires finite, non-negative distances.
  A `NaN` dissimilarity now raises `NaN dissimilarity value.` from the linkage,
  exactly as `fastcluster::hclust` and R's `wcCmpCluster` do.
- **An absent `matplotlib` names this package's extra.** `fuzzyseqplot` re-raised
  traminex's rewritten message, which told the user to install `traminex[plot]`.
- **An absent `tqdm` now warns** instead of silently ignoring `progressbar=True`.

### Changed
- **The traminex source commitment is re-pinned to clean sibling revision
  `6e76440b8ed7fc4a1c885c78546425a877bac275`** (aggregate SHA-256
  `d6eafe73…32e5443`) through the reviewed manifest-update procedure, closing
  the fail-closed canonical-environment state disclosed since 2026-08-30. The
  58-commit delta was reviewed at every delegated boundary and proved by the
  full ordinary and strict suites against the re-pinned provider. One
  delegated contract changed, R-faithfully: public `traminex.seqdist` now
  stops on a missing substitution cost for the OM family exactly as R's
  `seqdist` does, so `bootclustrange` and `seqnullcqi` inherit R's refusal
  for `seqdist_args={"method": "OM"}` without `sm`/`substitution_costs`; the
  earlier acceptance was a provider divergence, the `wcCmpCluster` tutorial's
  bootstrap examples now pass `sm=2`, and a consumer regression pins the
  refusal and the unblocked form. The low-level `sequences_distance` adapter
  keeps its documented `subm=2` OM default.
- **`pytest` no longer suppresses its own test count.** `pyproject.toml` set
  `addopts = "-q"`, which combined with the `-q` in every documented command and
  four CI steps to give `-qq`, hiding the "N passed" line. `addopts` is unset, so
  a single `-q` behaves as written; the documented commands are unchanged.
- **New optional extras `progress` (`tqdm`) and `threads` (`threadpoolctl`), and
  `plot` now includes `seaborn` and `mplcursors`.** Each changes observable
  behaviour and was previously declared nowhere or only under `dev`.
- **The optional Python `fastcluster` backend is removed.** No documented
  installation reached it, its numerics were never exercised, and a user who
  happened to have the package installed silently got a different code path. The
  linkage is now pure Python with no optional dependency, and
  `get_backend_info()` reports it as a `hierarchical_linkage` capability.
- `weightedcluster._validation.validate_weights` and
  `validate_clustering_labels`, and `weightedcluster.seqnull._freqs_per_time` and
  `_global_transition`, are removed. All four were private, unreferenced by any
  library path, and the last two duplicated statistics traminex owns and the
  production path already obtains from `seqstatd` and `seqtrate`.

### Documentation
- **The warning stream through `bootclustrange` and `seqnullcqi` widened because
  traminex advanced, not because of a regression here.** traminex moved to
  `d75e2e0334bf2966d643ed25be923a86dfac1268`, whose new cost-quality diagnostics
  emit a `CostDiagnosticMessage` and a `UserWarning` on `seqdist(..., method="OM",
  subm=<matrix>, ...)` where the previous revision was silent. Replaying the 107
  delegated calls this project makes against both revisions found **0 numeric and
  0 refusal differences and 6 warning differences**, all that same new pair. No
  code change was needed. This note exists so the next reader does not mistake
  the third suite warning for a regression.
- `README.md` and `docs/ARCHITECTURE.md` name the linkage reference
  implementation explicitly; both previously asserted R-compatibility for exactly
  the four routes that diverged.
- `README.md` no longer scopes the CLARA analysis-weight rescaling to weighted
  data. It applies to every input, and the factor is the original case count `n`
  in both cases.
- `docs/R_MIGRATION_GUIDE.md` states the distance-matrix contract in full and
  records three deliberate divergences: R's `wcCmpCluster` rejects an explicit
  `method="ward.D"`/`"ward.D2"` with `object 'hc' not found` while Python accepts
  them; R's `wcKMedoids` does not terminate on duplicated starting medoids while
  Python raises; and `seqclararange`'s fuzzy methods omit R's second `fanny`
  start.
- `docs/API_INDEX.md` documents the four package attributes that shadow
  same-named submodules, and the consequence a reader meets.
- `AGENTS.md` no longer lists `scikit-learn`, which no module imports.
- `CLAUDE.md` no longer points at a graph-first policy that does not exist or a
  graph index that does not cover this repository.

- **Closed the verified 2026-08-21 whole-codebase audit findings.** Current
  parity records now distinguish 18/0/0/1 export evidence from 22/3/0/3 S3
  presentation evidence, explicitly narrowing three reconstructed plot-payload
  claims. Result mappings, backend ownership, loaded thread limits, API and
  ecosystem identity, package doctests, wheel dependency propagation, archive
  authority, parallel-process guidance, and scientific-fixture retention now
  have executable contracts. Optional missing `mplcursors` emits a visible
  tooltip warning, and the directly imported `arviz-stats[xarray]` dependency
  is declared in the Bayesian and development extras.
- **Closed sequence-distance validation and delegation gaps from the
  2026-08-21 audit.** Fixed and position-dependent HAM/DHD costs now reject
  non-finite entries reachable by observed state pairs before low- or
  high-level traminex dispatch, while preserving intentional infinities for
  unobserved states. Delegation guards now spy on the exact
  `traminex.seqdist` calls in bootstrap/null workflows and pin the installed
  editable provider's version and source provenance.
- **Closed comparison routing, naming, and ARI-plot findings from the
  2026-08-21 audit.** `wcCmpCluster` now uses canonical, stable, deduplicated
  public keys for explicit methods, aliases, and `method="all"`. Precomputed
  `ward.D2` retains R's slot-ordered update for weighted and unweighted input:
  unique input dissimilarities can still create exact intermediate ties that
  SciPy resolves differently. The optional `embed_then_ward=True` coordinate
  route and other compatible unweighted methods retain their fast backends.
  Direct installed-R plot-data capture now records the upstream
  `ari.boxplot` first-K indexing defect; Python deliberately continues to use
  each K's selected iteration.
- **Completed the matrix PAM and medoid edge contracts identified on
  2026-08-21.** `wcKMedoids` and `wcKMedRange` now port WeightedCluster 2.0's
  square-matrix PAM loop while retaining its accurate condensed-input error.
  PAMonce uses the canonical scalar removal/addition loop, iterative KMedoids
  makes each medoid own its cluster on pseudometric duplicates, and multipass
  result counting now prefers identical solutions before comparing rounded
  objectives.
- **Repaired weighted CLARA and degenerate-quality semantics found by the
  2026-08-21 whole-codebase audit.** `seqclararange` now separates normalized
  sampling probabilities from aggregate analysis weights and consumes the
  canonical distance-normalization alias. Bootstrap range statistics replay
  WeightedCluster's unguarded native quality kernel, zero-denominator and
  zero-mass silhouettes retain their R non-finite/sentinel behavior, and the
  Numba HG/HGSD tier preserves native equal-distance accumulation order.
- **Repaired the confirmed 2026-08-15 R-fidelity findings.** Zero-total-weight
  clusters now preserve WeightedCluster's non-finite quality semantics across
  compiled, Numba, and pure tiers; centroid and median use the owned
  R-compatible linkage plus merge-count cuts for valid inversions; and DIANA
  now ports `cluster` 2.1.8.3's native banner/merge-order construction,
  including equal split levels. Optional linkage-backend selection is covered
  without exclusions, and the initial audited baseline is corrected to 1,703.
- **Preserved exact-k tied hierarchy cuts in `seqnullcqi`.** Hierarchical null
  replicates now use R-style merge-count cuts, so tied Ward.D heights no longer
  collapse requested partitions and silently discard finite quality results.
- **Repaired the executable `wcCmpCluster` tutorial.** Its `method="all"`
  result is now indexed with the actual `ward.D2` key and supplies the explicit
  palette required to plot all eleven methods. Current DIANA prose and
  documentation contracts name the exact native algorithm that is ported.
- **Repaired the executable `bootclustrange` tutorial.** Parallel performance
  examples now explicitly use the feasible random-subsample strategy instead
  of unintentionally selecting capacity-sensitive clustering strata.
- **Removed inert private scaffolding.** The shared-constants module now holds
  only values consumed by production code, with an AST contract preventing
  regressions, and three uncalled sequence-null helpers were deleted. The
  reachable optional-Seaborn and k-medoids fallback loaders are now measured,
  and property trees no longer pretend the required pandas dependency is
  optional. The final sequence-CLARA assembly boundary no longer accepts an
  ignored medoid callback.
- **Closed the 2026-08-12 whole-codebase audit findings.** PAMonce now follows
  WeightedCluster 2.0's scalar BUILD accumulation, best-improvement swaps, tie
  order, and zero-distance guard in every backend; public `initialclust`
  medoids are unambiguously one-based. Weighted silhouette and zero-mass
  quality behavior agree across backends, PBM uses its published square, and
  Davies–Bouldin retains valid zero contributions. Exact seeded child streams
  now use structured `SeedSequence` paths, so seeded outputs intentionally
  differ from releases that used collision-prone additive offsets. Sequence
  distance arguments and aliases are validated rather than silently ignored,
  fuzzy iteration/input-ownership contracts are enforced, large R-faithful
  hierarchical null workloads warn about cubic cost, and CI now proves each
  named compiled, Numba, and pure-Python tier.
- **`bootclustrange` orchestrator refactored without behavior changes.** The
  behavior-preserving ownership split moves input coercion to
  `_boot_inputs.py`, sampling/replay controls to `_boot_controls.py`, and
  result assembly to `_boot_result.py`. `boot.py` retains the public six-stage
  orchestrator, Traminex distance adapter, process-picklable worker wrapper,
  shared task runner, and unconditional runtime-context cleanup. Public
  signatures, validation order, sampling, randomness, process behavior,
  result schemas, copies, presentation, and live-R contracts are unchanged.
  See `docs/maintainers/BOOT_ORCHESTRATOR_REFACTOR_REPORT.md`.
- **`wcCmpCluster` orchestrator refactored without behavior changes.** The
  behavior-preserving ownership split moves NumPy/DIANA helpers to
  `_cmpcluster_algorithms.py`, method and replay validation to
  `_cmpcluster_controls.py`, and family-result assembly to
  `_cmpcluster_result.py`; `cmpcluster.py` retains the public orchestrator,
  linkage kernels, PAM helpers, and per-method dispatch. Public signatures,
  numerical behavior, live-R contracts, schemas, ordering, warnings,
  randomness, and input-mutation behavior are unchanged. See
  `docs/maintainers/CMPCLUSTER_ORCHESTRATOR_REFACTOR_REPORT.md`.
- **Post-B5 dependency cleanup implemented.** Live formula/design/label
  helpers now reside in private `_bayesian_rarcat_design.py`; the superseded
  Statsmodels GLM/REML/Fisher solver is test-only historical support in
  `tests/_historical_rarcat_solver.py`. Patsy remains in the `bayesian` and
  `dev` extras, while Statsmodels is absent from production and retained only
  in `dev`.
- **Reconciled the Traminex distance boundary after its 2026-08-10 audit
  repairs.** New consumer-level live-R tests pin stable first-occurrence
  orientation and warnings for asymmetric OM/static-DHD costs, including
  between-set assignment consistency. Distance ownership remains in Traminex;
  the local adapter and `traminex>=0.0.8` fail-closed dependency contract are
  unchanged. Installation documentation now distinguishes the Python `>=3.11`
  package requirement from the Python 3.11/3.12 CI-tested matrix.
- **Bayesian RARCAT B4 implemented privately.** New private mixture and storage
  modules combine completed B3 coefficient/effect draws with unchanged B2
  empirical partition masses, retain exact weighted support and a typed
  `BayesRARCATResult`, report within/between/total posterior variance, and
  provide deterministic serial/threaded post-processing, semantic plot data,
  headless text, and a versioned SHA-256-inventoried bundle using canonical
  JSON, safe NumPy payloads, and coordinate-preserving DataTree NetCDF files.
  Unequal-mass, rare-component, single-component, blocked-failure,
  serial/parallel, round-trip, and corruption gates pass. **B4 complete; B5 not
  started.** Independent B5 validation, optional Bayesian dependency metadata,
  public result stability, public `bayes_rarcat`, and compatibility `rarcat`
  remain absent. The parity ledger remains 18 strong, 0 partial, 0 unverified,
  and 1 missing.
- **Bayesian RARCAT B3 implemented privately.** The new private
  `weightedcluster._bayesian_rarcat_conditional` module validates the exact B2
  boundary, prepares scaled formula designs, fits each distinct aligned
  partition once with a direct-PyMC reference-category softmax model, and
  retains unmixed coefficient posteriors, continuous and categorical draw-wise
  marginal effects, prior/posterior predictive checks, diagnostics, warnings,
  replay seeds, and timing. The frozen numerical-reference, Bambi construction,
  simulation, exact-replay, failure, cache, degenerate-partition, and timing
  gates pass with zero failed components. **B3 complete; B4 not started.**
  Partition-mass integration, variance decomposition, serialization, public
  results/API, and Bayesian dependency metadata remain absent. The parity
  ledger remains 18 strong, 0 partial, 0 unverified, and 1 missing; neither
  public `rarcat` nor `bayes_rarcat` exists.
- **Bayesian RARCAT B2 implemented privately.** The new private
  `weightedcluster._bayesian_rarcat_partition` module provides isolated
  Bayesian-bootstrap and k-medoids replay streams, strict identity/input
  validation, fixed public-`wcKMedoids` replay, deterministic weighted and
  unweighted reference alignment diagnostics, lossless aligned-partition
  compression, membership probabilities, natural-log entropy, posterior
  co-clustering, ambiguity summaries, and complete replay provenance. It
  accepts no predictor or regression input and adds no public API or
  dependency. **B2 complete; B3 not started.** The parity ledger remains 18
  strong, 0 partial, 0 unverified, and 1 missing; public `rarcat` and
  `bayes_rarcat` remain absent.
- **Scientific governance corrected; Bayesian RARCAT B1 completed.** An
  exhaustive 134-prose-document audit separated scientific validity, software
  correctness, procedural compliance, and informational metadata. The new
  `docs/maintainers/SCIENTIFIC_GOVERNANCE.md` preserves raw evidence while
  allowing transparent errata to tests, orchestration, documentation, and
  decisions. `docs/maintainers/BAYESIAN_RARCAT_B1_GOVERNANCE_CORRECTION.md`
  reclassifies optional PyMC eigval metadata,
  artifact-replay drift, and the phase-confused launcher assertion without
  changing any seed, fit, attempt, or denominator. **B1 complete under
  materiality-based governance; B2 implementation authorized.** Historical
  entries below retain the original chronology and dispositions.
- **Bayesian RARCAT B1 fixed-null recovery-v2 study blocked.** The separately
  authorized execution used immutable source commit
  `3013c74f57186ae92d672a65a12ea81503319d9c` and completed exactly **100 /
  100** one-attempt positions. Every position failed solely on `all_finite`
  because `largest_eigval` and `smallest_eigval` were all-NaN sampler-stat
  fields; all other position gates passed. The terminal evidence is
  `docs/maintainers/BAYESIAN_RARCAT_B1_NULL_DIAGNOSTIC_RECOVERY_V2_REPORT.md`.
  B1 closure review is not eligible. **B1 fixed-null recovery-v2 diagnostic
  study blocked; B1 remains blocked; B2 blocked.**
- **Bayesian RARCAT B1 fixed-null recovery-v2 execution prompt verified.** The
  bounded delta-controller at
  `docs/maintainers/EXECUTE_BAYESIAN_RARCAT_B1_NULL_DIAGNOSTIC_RECOVERY_V2_PROMPT.md`
  pins the new protocol and review, reconciled runtime owners, and a complete
  old-to-v2 replacement map for seed, source, audit, evidence, and disposition
  identities while inheriting all scientific gates. No helper, seed, or study
  was created. **B1 fixed-null recovery-v2 execution prompt verified for
  separate execution; recovery-v2 study not yet executed; B1 remains blocked;
  B2 blocked**.
- **Bayesian RARCAT B1 fixed-null recovery-v2 protocol verified.** The bounded
  review at
  `docs/maintainers/BAYESIAN_RARCAT_B1_NULL_DIAGNOSTIC_RECOVERY_V2_PROTOCOL_REVIEW.md`
  independently reproduced the committed protocol identity, all three current
  runtime-reference owner/blob/hash tuples, the separate seed namespace, and
  the unchanged scientific and terminal rules. It creates no experimental
  artifact. **B1 fixed-null recovery-v2 protocol verified; execution-prompt
  authoring eligible; recovery-v2 study remains unauthorized; B1 remains
  blocked; B2 blocked**.
- **Bayesian RARCAT B1 fixed-null recovery-v2 protocol preregistered.**
  `docs/maintainers/BAYESIAN_RARCAT_B1_NULL_DIAGNOSTIC_RECOVERY_V2_PREREGISTRATION.md`
  preserves the immutable original and first-recovery terminal records, pins
  the actual owning Git objects for the current runtime-reference bytes, and
  defines disjoint seed, source, audit, and evidence namespaces without
  changing any scientific input or threshold. It creates no helper or seed and
  authorizes no execution. **B1 fixed-null recovery-v2 protocol preregistered;
  independent review required; B1 remains blocked; B2 blocked**.
- **Bayesian RARCAT B1 fixed-null recovery blocked before seed derivation.**
  Execution of
  `docs/maintainers/EXECUTE_BAYESIAN_RARCAT_B1_NULL_DIAGNOSTIC_RECOVERY_PROMPT.md`
  stopped at R0 because current frozen runtime-reference bytes did not match
  their pinned owning objects. Zero recovery seed calls occurred, **0 / 100
  recovery positions were attempted**, and no helper, launcher receipt, audit
  tree, manifest, dataset, fit, draw, or shard exists. The terminal record is
  `docs/maintainers/BAYESIAN_RARCAT_B1_NULL_DIAGNOSTIC_RECOVERY_REPORT.md`.
  **B1 fixed-null recovery diagnostic study blocked; B1 remains blocked; B2
  blocked.** Further recovery needs new prospective authority and an
  independently reviewed replacement protocol/controller.
- **Bayesian RARCAT B1 fixed-null recovery execution prompt verified.** The
  bounded new-session controller is
  `docs/maintainers/EXECUTE_BAYESIAN_RARCAT_B1_NULL_DIAGNOSTIC_RECOVERY_PROMPT.md`.
  It requires the complete recovery helper/tests to be committed before the
  clean literal-module launcher proof and any seed derivation; prohibits v1
  private-manifest access or denominator reuse; and preserves the frozen
  scientific, sampling, failure, and terminal-decision contracts. No recovery
  helper, seed, dataset, fit, or result was created. **B1 fixed-null recovery
  execution prompt verified for separate execution; recovery study not yet
  executed; B1 remains blocked; B2 blocked.**
- **Bayesian RARCAT B1 fixed-null recovery protocol independently verified.**
  The prospective non-retry contract is
  `docs/maintainers/BAYESIAN_RARCAT_B1_NULL_DIAGNOSTIC_RECOVERY_PREREGISTRATION.md`;
  its bounded review is
  `docs/maintainers/BAYESIAN_RARCAT_B1_NULL_DIAGNOSTIC_RECOVERY_PROTOCOL_REVIEW.md`.
  They preserve the immutable 100/100 v1 failure record, define a disjoint seed
  domain and separate denominator, and require a literal-module launcher proof
  before any seed derivation. No recovery helper, seed, data, fit, or execution
  was created. **B1 fixed-null recovery protocol verified; execution-prompt
  authoring eligible; recovery study remains unauthorized; B1 remains blocked;
  B2 blocked.**
- **Bayesian RARCAT B1 fixed-null diagnostic study executed and blocked.**
  `docs/maintainers/EXECUTE_BAYESIAN_RARCAT_B1_NULL_DIAGNOSTIC_ROBUSTNESS_PROMPT.md`
  committed the complete helper/tests at `5ba5ba1` before seed derivation and
  passed its clean immutable/environment preflight. The exact direct-script
  command then recorded all 100 one-attempt positions as failures before data
  generation because `tests._bayesian_rarcat_b1_spike` was not importable from
  that script boundary. No response, PyMC fit, posterior/predictive draw, or
  NetCDF exists. The independently recalculated rate is 100/100 (95% Wilson
  `[0.9630065017930143, 1.0]`); continuous summaries are unevaluable. The
  terminal record is
  `docs/maintainers/BAYESIAN_RARCAT_B1_NULL_DIAGNOSTIC_ROBUSTNESS_REPORT.md`.
  No retry or replacement is authorized: **B1 fixed-null diagnostic study
  blocked; B1 remains blocked; B2 blocked**.
- **Bayesian RARCAT B1 fixed-null diagnostic eligibility review passed.** The
  bounded record is
  `docs/maintainers/BAYESIAN_RARCAT_B1_NULL_DIAGNOSTIC_ELIGIBILITY_REVIEW.md`:
  the fixed-null diagnostic-study execution prompt is eligible for separate
  authoring, while study execution remains unauthorized. Immutable decision,
  preregistration, review, and replay-contract objects passed; the exact local
  environment and resource prerequisites are available. No seed, dataset, fit,
  result, execution helper, or execution prompt was created. Historical
  failures and dispositions remain unchanged, and **B1 remains blocked; B2
  blocked**.
- **Bayesian RARCAT B1 sign-recovery claim withdrawn prospectively.** The
  authorized decision is recorded in
  `docs/maintainers/BAYESIAN_RARCAT_B1_SIGN_RECOVERY_DECISION.md`: the
  sign-recovery claim and gate are withdrawn from prospective B1 acceptance,
  and a diagnostic-study eligibility review is required. The failed replay,
  `null/072`, and every historical disposition remain unchanged. No diagnostic
  execution is authorized, **B1 remains blocked; B2 blocked**, and the only
  next step is the later eligibility review.
- **Verified the next Bayesian RARCAT B1 requirement-decision boundary.** The
  corrected
  `docs/maintainers/ESTABLISH_BAYESIAN_RARCAT_B1_SIGN_RECOVERY_REQUIREMENT_PROMPT.md`
  implements the fixed prospective removal of the unsupported sign-recovery
  claim and gate. It requires no additional launch input, asks no questions,
  and prohibits historical rescoring, seed derivation, sampling, and diagnostic
  execution. The decision preserves **B1 remains blocked; B2 blocked**.
- **Bayesian RARCAT B1 design/replay review completed and blocked.** The
  non-experimental work was governed by
  `docs/maintainers/REVIEW_BAYESIAN_RARCAT_B1_DESIGN_REPLAY_PROMPT.md`; the
  durable record is
  `docs/maintainers/BAYESIAN_RARCAT_B1_DESIGN_REPLAY_REVIEW.md`; its source
  search found that the sign-recovery gate is unsupported. The committed
  future-study contract is
  `docs/maintainers/BAYESIAN_RARCAT_B1_NULL_DIAGNOSTIC_ROBUSTNESS_PREREGISTRATION.md`,
  but no seed was derived and no fit ran. At review completion, the separate
  sign decision was unresolved, no diagnostic-study execution prompt is
  eligible, and **B1 remains blocked; B2 blocked**. Historical replay failure
  and `null/072` are unchanged.
- **Whole-codebase audit and remediation.** Public and internal validation,
  native-kernel safety, backend controls, result schemas, live-R and frozen
  evidence harnesses, clean-runner CI, packaging smoke tests, and active
  documentation were audited source-first and strengthened with regression
  coverage. The durable findings and verification record is
  `docs/maintainers/WHOLE_CODEBASE_AUDIT_AND_REMEDIATION_REPORT.md`.
- **Removed two ineffective `seqpropclust` keywords.**
  `feature_include_transitions` and `feature_ngrams_max` never affected
  extraction and are no longer accepted. Use the effective `properties` and
  `max_k` controls; misspelled or obsolete keywords now fail explicitly.
- **Bayesian RARCAT B1 independently verified; B1 remains blocked; B2
  blocked.** The private spike executed through
  `docs/maintainers/IMPLEMENT_BAYESIAN_RARCAT_B1_PROMPT.md`; the bounded
  `docs/maintainers/VERIFY_BAYESIAN_RARCAT_B1_PROMPT.md` preserved and
  independently recalculated the immutable private spike, committed an ADEMP
  calibration preregistration before seed derivation, and executed an
  untouched 800-primary-fit plus 40-sensitivity-fit holdout. B1.1, B1.2, and
  B1.4 pass, but the authoring wrapper is not fully reproducible, one original
  null position is diagnostically invalid, and no justified sign-recovery gate
  exists. The durable decision is
  `docs/maintainers/BAYESIAN_RARCAT_B1_INDEPENDENT_VERIFICATION_REPORT.md`.
  Production source, package dependencies, public API, the parity ledger, and
  traminex were unchanged; no B2 work began.
- **Bayesian RARCAT redesign approved; implementation not started.** Exact R
  `rarcat` parity and its Phase 3A-3E GLM/`margins`/crossed-REML path are
  superseded. The active
  `docs/maintainers/BAYESIAN_RARCAT_REDESIGN.md` selects predictor-independent
  Bayesian-bootstrap weighted-k-medoids partitions, reference-label alignment,
  conditional Bayesian categorical regression, draw-wise marginal effects,
  and partition-mass-weighted posterior integration. Public `rarcat` remains
  an intentional R-parity gap, and planned `bayes_rarcat` is not yet
  implemented or exported.
- **Historical WeightedCluster 2.0 Phase 3A conformance reopening.** Its final
  exact-parity state was “Phase 3A conformance remediation pending; Phase 3B
  blocked.” Later independent counterevidence invalidated the former
  fixture-based completion claim for structured formula names, numeric label
  rendering, crossed-REML state/numerics, and GLM/AME tolerance validation.
  F1 grouping/order/alignment remains a regression gate. The contract-first
  implementation session was controlled by
  `docs/maintainers/IMPLEMENT_WEIGHTEDCLUSTER_2_0_PHASE3A_CONFORMANCE_METHOD_PROMPT.md`.
  No public `rarcat`, Phase 3B behavior, or runtime-R path was added. The
  evidence is retained to explain the later Bayesian redesign.
- **Documentation integrity remediation.** Current phase/ledger entry points,
  ineffective `seqpropclust` controls, distance-normalization semantics,
  inherited versus Python-only CLARA criterion directions, the guarded private
  bootstrap defect, fixed-Ward.D `seqnullcqi` metadata behavior, and semantic
  plotting parity boundaries are now stated precisely. Ledger-derived and
  cross-document regression contracts make those claims durable without
  changing product behavior, public signatures, or parity evidence levels; the
  evidence record is
  `docs/maintainers/DOCUMENTATION_INTEGRITY_REMEDIATION_REPORT.md`.
- **WeightedCluster 2.0 Phase 2 complete; Phase 3 next.** Ten bounded product
  commits close all 14 Phase 2 exports and all 17 owned S3/presentation
  mappings with direct live-R evidence, deterministic replay for stochastic
  workflows, public-traminex delegation, and exact or source-justified numeric
  contracts. The executable ledger now reads 18 strong, 0 partial,
  0 unverified, and 1 missing. `rarcat` remains absent and owned by Phase 3, so
  complete package parity is not complete. The durable evidence is recorded in
  `docs/maintainers/WEIGHTEDCLUSTER_2_0_PHASE2_VERIFICATION_REPORT.md`.
- **WeightedCluster 2.0 Phase 2 association closure.** `clustassoc` now pins
  exact result table order/classes/labels, numeric and factor models, complete
  cases, discrepancy delegation through public `traminex.dissmfacw`, malformed
  inputs, warnings, convergence, and source-visible zero-weight behavior.
  `plot_clustassoc` pins selected arrays, the installed source's legacy
  blank-label lookup, every base plot type, validated options, source errors,
  and headless Matplotlib rendering while retaining diagnostics off the exact
  four-column R table.
- **WeightedCluster 2.0 Phase 2 comparison-family closure.** `wcCmpCluster`
  now requires `maxcluster`, defaults to `pam_combine=True`, accepts matrix or
  condensed distances, implements all eight R hclust methods with weighted
  members, and covers PAM, DIANA, flexible-beta, every `pam.<method>` branch,
  and explicit zero-based PAM replay. Typed `clustrangefamily`/`allstats`
  metadata plus exact print, summary, semantic panel-data, layout, palette,
  and headless plot adapters are live-R pinned. The installed 2.0 Ward-name
  casing and grouped-method vector-stat plot failures are recorded as defects;
  Python keeps those workflows usable as deliberate safe translations.
- **WeightedCluster 2.0 Phase 2 cluster-range closure.** `wcKMedRange` now
  preserves ordered/nonconsecutive K values, forwards the R medoid options,
  accepts matrix or condensed distances, and reproduces R's replacement
  bootstrap through explicit zero-based sample replay and frequency-count
  weights. `as_clustrange` now covers ordered factor frames, hclust,
  no-tie twins/agnes, and property-tree dispatch with typed dict-compatible
  class/table metadata. New print, summary, semantic plot-data, confidence
  interval, and headless-render adapters are live-R pinned. Python
  deliberately keeps scalar-statistic and RHC-interval plotting usable after
  tests reproduced the corresponding data-frame-drop and missing-bootstrap-
  column errors in WeightedCluster 2.0.
- **Verified WeightedCluster 2.0 Phase 2 implementation prompt.**
  `docs/maintainers/IMPLEMENT_WEIGHTEDCLUSTER_2_0_PHASE2_PROMPT.md` provides
  the evidence-pinned new-session instructions for closing all 14 Phase 2
  exports and 17 owned S3/presentation mappings. It derives a dependency-safe
  ten-slice implementation order from the R/Python call graph, requires
  deterministic replay instead of cross-runtime seed comparison, preserves
  traminex ownership, and enforces final 18-inherited-export closure without
  claiming `rarcat` or complete package parity.
- **WeightedCluster 2.0 Phase 1 complete; Phase 2 next.** Four bounded
  live-R/TDD commits align `wcAggregateCases` (`b6a14b5`), `seqclustname`
  (`f35b49b`), `as.seqtree` (`47526fb`), and `fuzzyseqplot` (`0304969`). The
  exact aggregation-key/index, factor/categorical, hierarchy/tree, and fuzzy
  semantic plot-data translations are recorded in
  `docs/maintainers/WEIGHTEDCLUSTER_2_0_PHASE1_VERIFICATION_REPORT.md`. The
  executable ledger now reads 11 strong, 7 partial, 0 unverified, and 1
  missing. Complete package parity is not complete: Phase 2 still owns the
  seven partial exports and Phase 3 owns `rarcat`.
- **Verified WeightedCluster 2.0 Phase 1 implementation prompt.**
  `docs/maintainers/IMPLEMENT_WEIGHTEDCLUSTER_2_0_PHASE1_PROMPT.md` provides
  the evidence-pinned new-session instructions for closing
  `wcAggregateCases`, `seqclustname`, `as.seqtree`, and `fuzzyseqplot` in the
  required order. It records the exact source/live-oracle contracts, including
  the reproduced aggregation-key ordering and fuzzy factor/NULL branch, and
  requires RED-first live-R tests, traminex delegation, atomic commits, ledger
  reconciliation, and complete verification without claiming package parity.
- **Historical verification identifier corrected.** The Phase 4 record in the
  active handoff now names the real intentional-skip pytest node instead of a
  nonexistent shorthand identifier, with a contract preventing regression.
- **Durable WeightedCluster 2.0 Phase 0 verification.**
  `docs/maintainers/WEIGHTEDCLUSTER_2_0_PHASE0_VERIFICATION_REPORT.md` now
  preserves the exact implementation commits, source/oracle and namespace
  evidence, final ordinary/strict/coverage/build results, warnings,
  distribution exclusions, and retrospective RED failures. It explicitly
  distinguishes test detection power from original test-first chronology,
  which cannot be recovered from uncommitted session output.
- **WeightedCluster 2.0 Phase 0 complete.** The exact CRAN archive is committed
  at `orig_R_pkg/WeightedCluster_2.0.tar.gz` with SHA-256
  `6a03fbbc090ac92498a410faab1aeb88d686ff7a81dcfdcb97edacda44f393cc`,
  and `orig_R_pkg/WeightedCluster` is now the canonical readable 2.0 source.
  The historical 1.8-1 tree and exact archive remain at
  `orig_R_pkg/WeightedCluster-1.8-1` and
  `docs/legacy/WeightedCluster_1.8-1.tar.gz` (SHA-256
  `daf9987cee5ac9e706a39e72f54a64e49e9fe91b0bc3cfb5e5a94d5076b2df5b`).
  The shared live-R fixture now requires exactly WeightedCluster 2.0 and
  TraMineR 2.2-13. A deterministic capture script, a formals/namespace fixture,
  and machine-readable/readable ledgers account for 19 exports and 28 S3
  registrations without promoting the audited 7 strong, 7 partial, 4
  unverified, and 1 missing evidence levels. Phase 1 is next; complete package
  parity is not complete, and no product implementation changed in Phase 0.
- **WeightedCluster 2.0 complete-parity target.** The active roadmap now targets
  all 19 exports from the upstream package version reported by R as `2.0`,
  including RARCAT and its print/summary/plot behavior. The superseding plan
  also requires closure of the inherited aggregation, cluster-naming,
  `as.seqtree`, fuzzy-plot, stochastic-workflow, default-path, and
  `seqpropclust` property-family gaps before parity may be claimed. General
  `cmpcluster.py` / `boot.py` refactoring is deferred until that acceptance gate
  passes.
- **Phase 4 backend contracts and evidence.** Pure-Python and Numba cluster
  quality now reproduce all ten shared R statistics; public routing is declared
  and tested per operation rather than as a global three-tier ladder. Medoids
  publicly select compiled/pure only, quality selects compiled/Numba/pure,
  `wcSilhouetteObs` stays Python-only, private weighted-silhouette kernels remain
  unselected, `seqnull` is model/shape-conditional, and sequence distances stay
  owned by traminex. The compiled core now exports only its three owned kernels
  and no longer links OpenMP. `get_backend_info()` adds deterministic capability
  diagnostics while preserving its legacy strings. A reproducible harness and
  the host-specific
  `benchmarks/results/phase4-reference-ubuntu-x86_64.json` record correctness,
  cold/warm timing, thread limits, and provenance without a universal speed or
  artifact-portability claim.
- **`clustassoc` association and BIC parity.** Numeric and categorical
  covariates now match installed WeightedCluster 2.0 with and without weights.
  `Remaining` and `Unaccounted` delegate to traminex `dissmfacw`; numeric BIC
  reproduces `BIC(lm(..., weights=))`, categorical BIC reproduces the
  factor-only frequency-weighted `nnet::multinom` likelihood and parameter
  count, and separated fits use nnet's `abstol` convergence contract. Because
  SciPy and nnet take different BFGS line-search steps at a separated likelihood
  boundary, those finite stopping values are pinned within `2e-4`; non-boundary
  fixtures remain pinned within `1e-7`. Existing dict fields remain, with
  transparent BIC diagnostics and row names added.
- **Property clustering now consumes traminex trees.**
  `wcPropertyClustering` delegates faithful weighted tree growth to
  `traminex.disstree`; `seqpropclust` delegates its tree stage to
  `traminex.seqtree`. The pyweightedcluster adapter preserves its dictionary,
  cutting, labeling, pruning, plotting, and split-schedule contracts but owns no
  split, discrepancy, permutation-statistic, or tree-growth mathematics. Exact
  live-R checks cover all nine sequence-property families, property names/order,
  types and missingness, support/event-length boundaries, topology, splits,
  node statistics, medoids, fitted leaves, pruning, weights, all documented
  permutation modes, and semantic headless plot data. The legacy
  `random-sampling` tree mode composes public traminex tree and `dissassoc` APIs;
  `seqpropclust` defaults to R WeightedCluster's
  `weight.permutation="diss"`, so non-integer sequence weights work without an
  explicit override.
- **The live oracle expectation is exact.** Every shared parity fixture now
  requires WeightedCluster 2.0 and TraMineR 2.2-13; other editions are no longer
  accepted for this target. The canonical 2.0 and historical 1.8-1 source
  records remain explicitly distinct.
- **Ubuntu 24.04 is now the canonical development platform.** Active agent,
  contributor, installation, handoff, CI, and Phase 2c planning instructions use
  `~/miniforge3/envs/datasci/bin/python`; the ecosystem and
  migration inventories were compared with the traminex master copies before
  adding the consuming project's Phase 2c boundary records.
- **Runtime and build policy now follows the current traminex baseline.** The
  package requires Python >=3.11, NumPy >=1.26, pandas >=2.1, SciPy >=1.11.1,
  and traminex >=0.0.8; Black and Ruff target Python 3.11. Active CI and wheel
  matrices use Ubuntu 24.04 with Python 3.11/3.12. Because neither local
  repository has an authoritative remote coordinate, clean runners require an
  explicit maintainer-provided `TRAMINEX_SOURCE` and fail clearly when it is
  absent instead of guessing a dependency source.
- **Linux repository and release hygiene.** Git now enforces LF text and explicit
  binary attributes, local audit/coverage/build products are ignored, development
  dependencies are available through `.[dev,numba]`, and sdists exclude generated,
  historical, audit, graph, and vendored-oracle trees. The resulting sdist is
  approximately 208 KB instead of 6.3 MB. Removed a redundant `_OPENMP` definition
  that caused a GCC redefinition warning.

### Fixed
- **R-compatible centroid and median hierarchies.** Unweighted `wcCmpCluster`
  no longer routes centroid/median through SciPy or optional fastcluster,
  whose distance convention disagrees with R `hclust`. Hierarchy consumers
  now cut by merge count, preserving valid inversions for `wcCmpCluster`,
  hierarchy-initialized PAM, `as_clustrange`, and `as_seqtree`. The previous
  33/54 Python-only public failures are eliminated, and optional-linkage
  selection now has measured installed/absent branch tests with no coverage
  exclusion.
- **Exact zero-total cluster weights in `wcClusterQuality`.** The compiled,
  Numba, and pure-Python tiers now preserve WeightedCluster 2.0's `0/0`
  propagation: CH, R2, CHsq, and R2sq are undefined when any retained cluster
  has exactly zero total case weight. The same live-R contract now covers the
  public API, per-cluster silhouettes, and cluster-range bootstrap callers;
  arbitrarily small positive cluster mass remains finite and unchanged.
- **HC selection direction in `seqclararange`.** HC is now converted to a
  minimized selection objective, consistently with `clustrange` and
  `clustrangefamily`; all other criterion directions and non-finite fallback
  behavior remain unchanged.
- **Phase 1 fuzzy verification gaps.** `fuzzyseqplot` now preserves the exact
  live-R plot-driving data for out-of-range memberships, infinite powered
  weights, and empty retained memberships; maps `main` and
  `with.legend`/`with_legend` through a tested adapter; rejects duplicate or
  unsupported plot keywords; and lazily imports traminex plotting so a base
  wheel imports without Matplotlib. Rendering dependencies are exposed through
  `.[plot]` (`280d716`).
- **Phase 4 benchmark evidence integrity.** Numba cold-start rows now compare
  complete results with an independent compiled-or-pure direct-kernel oracle
  after the timed call instead of treating structural validity as correctness.
  Optional peak-RSS collection now launches one fresh process per
  operation/backend/scale/weight row, so `ru_maxrss` values cannot accumulate
  across backends. Benchmark schema 1.1 records the oracle and process contract,
  and executable backend-info examples no longer pin a host-specific Numba
  version or quote printed strings incorrectly.
- **The direct rpy2 parity test executes again on rpy2 3.6.** Removed the obsolete
  global `numpy2ri.activate()` call, made the fixed-initialization comparison
  deterministic, and compare label-invariant partitions because R identifies
  clusters by medoid indices rather than contiguous labels on this path.
- **`fuzzy.fcmdd` medoid candidate set now matches R `WeightedCluster:::wfcmdd`.**
  R restricts each cluster's medoid candidates to points not fully assigned
  (membership == 1) to another cluster (`apply(u[, -k], 1, max) < 1`); the Python
  port considered all points, so on the first sweep of a crisp init it could pick
  a medoid from outside the cluster and converge to a different optimum. Rare via
  the package's own k-medoids init (~1 in 600 random datasets) but common when a
  caller supplies an arbitrary initial membership. Now validated against live R at
  800/800 generic-position cases (was dozens diverging). A fallback preserves the
  pre-fix behaviour only for a degenerate empty-cluster init (which R errors on but
  the CLARA pipeline can produce on tiny subsamples), so no healthy init changes.
  This is a behaviour change for `fcmdd`/`wcFuzzyRange` on affected inputs.

### Testing
- **Verified WeightedCluster 2.0 Phase 0 implementation prompt.** A new-session
  prompt pins the official CRAN URL and SHA-256, preserves the 1.8-1 historical
  source, requires exact WeightedCluster 2.0 / TraMineR 2.2-13 fixture-wide
  enforcement, and specifies RED-first namespace/formals capture plus a
  machine-validated 19-export/28-S3 parity ledger. Its scope explicitly excludes
  product implementation and hands off the four inherited unverified exports
  to Phase 1 only after complete focused, strict, build, documentation, graph,
  and clean-tree verification.
- **Active documentation now has executable reliability gates.** Deterministic
  tests resolve in-scope local links, anchors, and indexed paths; classify every
  active Python/shell fence; execute all ordinary examples in per-document
  isolated state; and require bounded executable pairs for expensive examples.
  All eight active Quarto tutorials also pass isolated and complete-set
  headless execution with rendered cell-output evidence. The maintained audit
  ledger records the one-time external-link results and full render matrix.
- **Phase 2c post-rebase remediation.** A live-R regression now exercises
  `seqpropclust` with non-integer sequence weights and the implicit permutation
  default. The Ubuntu 24.04 R-parity matrix runs the complete test suite under
  `WC_PARITY_STRICT=1`, rather than only the legacy direct k-medoids check. The
  traminex knowledge graph was refreshed and verified against the mixed numeric
  and categorical `_build_design` implementation.
- **The CI coverage gate is executable and remains at 85%.** Coverage.py omits
  only `_numba.py`, whose JIT-compiled bodies are not observable as Python line
  execution and whose behavior remains covered by backend-parity tests. New
  direct tests cover the pure fallback, validation, thread restoration,
  cluster naming, range error handling, and plotting data adapters/branches;
  the exact CI command now reports 85.02% instead of failing at 75.40%.
- **Deterministic `wcKMedRange` parity anchors.** Live WeightedCluster 2.0 tests
  pin fixed-init `KMedoids`/`PAMonce` partitions, the shared ten R quality
  statistics, range structure, statistics given identical labels, and Python
  seed reproducibility. Installed R's `PAM` path is recorded as unimplemented;
  labels from different R/Python random-number generators are not compared.
- **Property-tree semantic parity.** Live WeightedCluster 2.0 tests pin
  `wcPropertyClustering` and the verified `seqpropclust` state/duration path,
  including weighted and unweighted trees, pruning/cutting/labels, split
  schedule, node medoids/statistics, fitted memberships, and within-runtime
  seeded permutation reproducibility.
- **R-parity breadth: bootstrap & null models (roadmap Phase 2b).** New
  `tests/test_r_parity_phase2b_bootstrap_null.py` pins `seqnull`'s null model exactly
  to R (`_global_transition` == `seqtrate`, `_freqs_per_time[0]` == `seqstatd$Frequencies[,1]`)
  and checks the structural contract of all six `seqnull` models, `bootclustrange`, and
  `seqnullcqi` against live R (WeightedCluster 2.0), plus Python seed-reproducibility for each.
  `bootclustrange` and `seqnullcqi` baselines are pinned exactly to R `as.clustrange`. The
  stochastic generation/resampling is compared structurally (R's 10 clustrange stat columns are
  the leading subset of Python's 14), not by value. Verified `deparse`-identical to the vendored
  1.8-1 `seqnull`/`bootclustrange`. Test-only; `tests/test_delegation_guard.py` untouched.
- **R-parity breadth: fuzzy & CLARA (roadmap Phase 2a).** New
  `tests/test_r_parity_phase2a_fuzzy_clara.py` pins `crispness` (exact) and the
  `fcmdd` objective/medoids to R's internal `WeightedCluster:::crispness` /
  `:::wfcmdd` (deterministic given a fixed initial membership), and checks
  `seqclararange`'s structural contract against R plus Python seed-reproducibility
  (the algorithm is stochastic, so values are not pinned). Test-only.
- **Phase 2a audit hardening.** The `fcmdd` parity test now also covers
  non-uniform weights (pinning R's raw-weights convention) and a user-supplied
  arbitrary init (the `fcmdd` candidate-set fix above). The `seqclararange`
  structural-contract driver now seeds R and uses a 16-sequence dataset at
  sample.size=10, removing an order-dependent failure (an unseeded, tiny-subsample
  draw made R's internal `cutree(., k)` fail in full-suite order). The
  oracle-version rationale was corrected: `deparse()` shows WeightedCluster 2.0
  rewrote `:::wfcmdd` (it is not the "purely additive" edition the CRAN NEWS
  implies), though it stays a sound oracle on the paths the tests exercise.
- **Historical parity-harness integrity record (roadmap Phase 1; superseded for
  active oracle policy by WeightedCluster 2.0 Phase 0).** The R-parity oracle
  version is now recorded at test time and a guard
  (`tests/test_oracle_provenance.py::test_oracle_is_accepted_edition`, via
  `unexpected_oracle_editions`) xfails only when the live R
  `WeightedCluster`/`TraMineR` is not a known-equivalent edition — accepted:
  WeightedCluster {1.8-0, 1.8-1, 2.0}, TraMineR {2.2-12, 2.2-13}, which are
  behaviourally identical for the ported functions per the CRAN NEWS (1.8-1 → 2.0
  adds only RARCAT; issue I-1, design D-1). The `r_parity_session` fixture now
  honest-fails (not silently skips) under `WC_PARITY_STRICT=1` when R is absent,
  via the new `fail_or_skip` gate mirroring the traminex import gate (issue I-2).
  Test-only; no product behaviour change. See
  `docs/superpowers/specs/2026-06-14-pyweightedcluster-roadmap-design.md`.
  The accepted-version range above is preserved as dated implementation
  history; the current shared fixture accepts exactly WeightedCluster 2.0 and
  TraMineR 2.2-14, as recorded at the top of this Unreleased section. It
  accepted TraMineR 2.2-13 until 2026-09-21.
- **Delegation hardening (roadmap Phase 3).** New `tests/test_delegation_guard.py`
  is a standing structural guard (no R, no traminex runtime values) proving the
  removed distance/cost modules stay absent, `_seqdist` keeps delegating distances
  to `traminex.distances`, the four diss sites keep consuming
  `traminex.diss.weighted_pairwise_sum`, and the public surface never re-exports
  `sequences_distance` (issue I-4). Test-only; no product behaviour change.

## [0.0.3] - 2026-06-13

### Changed
- **Migration Phase 2.5 (A5–A8; see MIGRATION_PLAN.md): the weighted
  pairwise-sum / inertia maths now comes from traminex**, where it is the
  R-anchored primitive `traminex.diss.weighted_pairwise_sum` (golden-tested
  against TraMineR 2.2-12 `tmrWeightedInertiaDist`; traminex 0.0.4–0.0.7
  also added the R-faithful `dissvar`, `dissassoc`, `disscenter` and
  `disstree`/`seqtree`). All four inline sites were refactored and the local
  maths removed, with results proven identical first
  (`tests/test_phase2_a5a8_traminex_parity.py` keeps verbatim copies of the
  removed maths as reference implementations):
  - `seqtree.py`: local `_weighted_pairwise_sum` deleted; `as_seqtree`'s
    weighted-inertia splits call traminex directly.
  - `_utils.py::pairwise_sum` (used by `proptree.py`): delegates to traminex
    (unit weights); `proptree.py` imports the traminex primitive directly.
  - `clustassoc.py::_weighted_within_and_total`: within/total weighted
    pairwise sums computed by traminex.
  - `quality.py` (pure-Python fallback of `wcClusterQuality`): the O(N²)
    moment loops for T/T2/W/W2 (CH/R2) and the HC pairwise moments are
    computed via traminex sums (the 0/1 between-indicator moments reduce to
    pair counts). The compiled `_wc_core` fast path (WeightedCluster's own
    C++ clustering-quality kernel, DUPLICATES A9) is unchanged.
  Declared dependency raised to `traminex>=0.0.7`. Suites green incl. the
  R-anchored Phase 1 parity tests and `WC_PARITY_STRICT=1`
  (224 passed / 13 skipped with live R).

## [0.0.2] - 2026-06-12

### Changed
- Migration Phase 1 (converge on traminex; see MIGRATION_PLAN.md):
  - **A1 (distances):** `weightedcluster._seqdist.sequences_distance[_between]`
    no longer computes HAM/OM/LCS/DHD itself; it delegates to
    `traminex.distances.pairwise` (the family's R-validated base) and only maps
    this project's `(metric, normalize)` API onto traminex's `method`/`norm`
    vocabulary. Declared `traminex>=0.0.3` as a runtime dependency. The
    `normalize=True` form used by clustering is pinned to R per metric: HAM ==
    R `HAM(sm=unit, norm="maxlength")`, OM == R `OM(sm=2, indel=1,
    norm="maxlength")`, LCS == R `LCS(norm="gmean")`, DHD == R
    `seqdist(method="HAM"/"DHD", sm=...)` (new
    `tests/test_phase1_a1_distances_traminex_parity.py`). DHD is served by
    traminex's new explicit-`subm` path. The private 3-tier distance backend
    (compiled `_wc_core` / numba / pure-Python `_fallback`) is no longer used for
    these metrics; the obsolete backend-distance parity tests were removed (the
    clustering-kernel backend-parity tests stay). One R-faithful behaviour
    correction: un-normalized (`normalize=False`) LCS is now R's raw
    `|a|+|b|-2*lcs` rather than half of it; `normalize=True` LCS is unchanged and
    clustering only ever uses `normalize=True`.
  - **A4 (`normalize_metric`):** kept as a deliberate tiny metric-name
    canonicalizer (the front door `_seqdist` uses before delegating to traminex),
    folded into A1 rather than replaced — a recorded decision (charter §4). Also
    stripped the now-dead pure-Python (`_fallback`) and numba (`_numba`) sequence
    *distance* kernels; their clustering kernels (weighted medoids, silhouette,
    cluster quality, seqnull) are untouched. After Phase 1 this project computes
    no sequence distances or costs of its own.
  - **A2 (OMspell):** removed the private `weightedcluster._omspell` kernel;
    `sequences_distance(metric="OMSPELL")` now delegates to traminex's R-faithful
    OMspell (`pairwise(method="OMSPELL")`). This project's `timecost` maps to R's
    `expcost`, durations use `tpow=1`, and `normalize=True`/`False` map to R
    `norm="maxlength"`/`"none"` (new
    `tests/test_phase1_a2_omspell_traminex_parity.py`). Correctness note: the old
    local OMspell omitted R's `-1` spell-duration shift and was not R-faithful;
    no test pinned its values.
  - **A3 (costs):** removed the private `weightedcluster.costs` module
    (`build_costs`, `build_substitution_costs_trate`,
    `build_substitution_costs_constant`). It had no internal consumers — the
    clustering pipeline takes caller-supplied `substitution_costs` — and was an
    unvalidated duplicate of traminex's R-anchored cost functions. Substitution
    and indel costs are now owned by traminex (`traminex.seqsubm`). New
    `tests/test_phase1_a3_costs_traminex_parity.py` pins traminex's CONSTANT and
    weighted/unweighted TRATE costs to R (`TraMineR::seqcost`, 2.2-12); the old
    local matrix was verified to agree with R before removal.

- Documentation cleanup and normalization:
  - Canonical docs retained in `docs/`
  - Active tutorials consolidated under `docs/vignettes/`
  - Maintainer-only backlog notes moved under `docs/maintainers/`
  - Historical milestone/draft docs moved under `archive/history/`
  - Root historical parity report moved under `archive/audits/`
  - Timestamped 2026 audit reports moved under `archive/audits/2026/`
  - Quarto kernel check artifacts moved under `archive/internal/quarto/`
  - Legacy and vendored upstream reference materials labeled explicitly as non-canonical
- Canonical docs updated for current code/API behavior:
  - `bootclustrange` return fields (`stats`/`meant`/`stderr`/`baseline_stats`)
  - `seqnullcqi` R-style structure (`stats` matrices, `hclust.method`, `seqdist.args`)
  - API export map aligned with `weightedcluster.__all__`
  - Ward and DIANA descriptions aligned with the live implementation
- Performance and reproducibility hardening across range/null/fuzzy/cmpcluster workflows:
  - Removed duplicate internal-index recomputation in `seqclararange`
  - Optimized fallback `HC` computation in `wcClusterQuality` (pairwise moment formula; no pair-array materialization)
  - Standardized `random_state` propagation in `wcFuzzyRange`, `seqnullcqi(kmedoid=True)`, and `wcCmpCluster` PAM paths
- Native build surface simplified to the shipped pybind11 extension

### Fixed
- `as_clustrange()` now keeps `kvals` and stats rows aligned for unordered clustering dictionaries.
- Pure-Python and numba `PBC` quality paths now match the compiled backend contract.
- Sparse/non-contiguous cluster labels are normalized consistently in cluster-quality computations.
- `bootclustrange()` no longer accepts invalid final retry samples after exhausted retries.
- `seqnullcqi()` no longer silently swaps hierarchical reclustering to k-medoids on runtime failure.
- Removed unreachable duplicate legacy code from `seqclararange()`.
- Narrowed broad runtime exception fallbacks across the package so real worker/runtime failures propagate.
- Removed stale references to non-official releases from active docs.
- Fixed broken internal links in architecture documentation.
- Replaced placeholder security contact guidance with actionable reporting instructions.

## [0.0.1] - Initial Public Release

### Added
- Initial Python port of R `WeightedCluster` core workflows.
- Core clustering APIs (`wcKMedoids`, `wcKMedRange`, `wcClusterQuality`, `wcSilhouetteObs`).
- Sequence APIs (`seqclararange`, `bootclustrange`, `seqnull`, `seqnullcqi`).
- Property-tree and association APIs (`seqpropclust`, `wcPropertyClustering`, `clustassoc`).
- Compiled core plus fallback backends.
- Initial test suite and documentation.

### Notes
- `0.0.1` is the only official release version.
- References to `0.2.x`, `0.3.x`, or `0.4.x` in archived files are historical draft artifacts.

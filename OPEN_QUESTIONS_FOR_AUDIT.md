# Historical Audit Questions and Resolutions

Status (2026-08-01): **closed as a live question list**. The whole-codebase
audit recorded in
`docs/maintainers/WHOLE_CODEBASE_AUDIT_AND_REMEDIATION_REPORT.md` resolved,
reclassified, or explicitly deferred every question that was still current.
The dated observations below are retained as historical evidence; they are not
active instructions or a current test-status report.

Current update (2026-08-21): the verified whole-codebase remediation report
supersedes earlier broad evidence descriptions. The export inventory remains
18 strong, 0 partial, 0 unverified, and 1 missing, while the separately counted
S3 inventory is 22 strong, 3 partial, 0 unverified, and 3 missing. See
`docs/maintainers/WHOLE_CODEBASE_AUDIT_2026_08_21_REMEDIATION_REPORT.md` and the
authoritative JSON parity ledger; the dated sections below remain historical.

Date compiled: 2026-06-11.
Compiled during a tidy/index pass that made **no changes to source logic, the C++ core, tests, or
`orig_R_pkg/`**. Everything below is a claim found in the repository (or an observation made during
the tidy pass) that the next audit should verify independently. Nothing here is asserted as true or
false — these are the things this pass could **not** verify by reading alone.

## WeightedCluster 2.0 Phase 0 — resolved (2026-07-17)

The active oracle/source and namespace-coverage questions are now executable.
WeightedCluster 2.0 and TraMineR 2.2-14 are the exact live target
(moved from 2.2-13 on 2026-09-21). The official
source is `https://cran.r-project.org/src/contrib/WeightedCluster_2.0.tar.gz`;
the committed `orig_R_pkg/WeightedCluster_2.0.tar.gz` has SHA-256
`6a03fbbc090ac92498a410faab1aeb88d686ff7a81dcfdcb97edacda44f393cc`,
and `orig_R_pkg/WeightedCluster` is its readable tree. WeightedCluster 1.8-1 is
historical only at `orig_R_pkg/WeightedCluster-1.8-1`; its exact
`docs/legacy/WeightedCluster_1.8-1.tar.gz` has SHA-256
`daf9987cee5ac9e706a39e72f54a64e49e9fe91b0bc3cfb5e5a94d5076b2df5b`.

The committed namespace fixture and executable ledger account for all 19
exports and 28 registered S3 methods. Phase 0 is complete, but complete package
parity is not complete.

## WeightedCluster 2.0 Phase 1 — resolved (2026-07-17)

Phase 1 complete; Phase 2 next. Exact live-R contracts now close the four
inherited unverified exports, and the executable ledger reads 11 strong, 7
partial, 0 unverified, and 1 missing. The durable evidence is
`docs/maintainers/WEIGHTEDCLUSTER_2_0_PHASE1_VERIFICATION_REPORT.md`. The seven
partial workflows remain Phase 2 work and `rarcat` remains missing Phase 3
work, so Complete package parity is not complete. The dated §3 and §4
observations below remain unchanged as historical audit evidence; this
resolution supersedes them for active work.

## WeightedCluster 2.0 Phase 2 and Phase 3 direction — resolved (2026-07-29)

Phase 2 closed all 18 inherited exports with strong live-R evidence. The
R-parity ledger therefore remains 18 strong, 0 partial, 0 unverified, and 1
missing. Complete package parity is not complete.

Repeated independent verification showed that the exact R RARCAT
GLM/`margins`/crossed-REML path was not a reliable Python implementation
target. The maintainer selected the predictor-independent modular Bayesian
replacement recorded in
`docs/maintainers/BAYESIAN_RARCAT_REDESIGN.md`. Exact R `rarcat` parity is no
longer active; public `rarcat` and planned `bayes_rarcat` are both absent, and
Bayesian implementation has not started. The older Phase 3A audit artifacts
remain historical evidence.

## Status update (2026-06-14)

Several items below were addressed by the traminex-convergence migration
(Phases 1–3; see `DUPLICATES.md` / `MIGRATION_PLAN.md`) and the 2026-06-14 roadmap
work (see `docs/superpowers/specs/2026-06-14-pyweightedcluster-roadmap-design.md`
and `CHANGELOG.md`):

- **§1 test count** — superseded: the suite is now ~255 tests (see `CHANGELOG.md`);
  re-measured this session at 0 failures / 0 errors with live R present.
- **§2 OMspell compiled/numba tier** — **obsolete**: `_omspell.py` was removed and
  `_seqdist.py` delegates *all* metrics (including OMSPELL) to traminex; no
  `_wc_core` distance import remains (guarded by `tests/test_delegation_guard.py`).
  The numba-tier-unexercised observation stands and is now stated honestly in the
  README / `docs/ARCHITECTURE.md` backend sections.
- **§3 parity-oracle provenance** — **addressed**: the live oracle version is now
  recorded and guarded (`tests/test_oracle_provenance.py`); the maintainer adopted
  the installed WeightedCluster 2.0 / TraMineR 2.2-13 as the oracle, which the CRAN
  NEWS shows is behaviourally identical to the pinned 1.8-1 for the ported
  functions (design decision D-1).

Later work resolved **§4 (R API coverage)** and **§5 (default-parameter
parity)** through the executable parity ledger and strict live-R contracts.
For **§6 (artifact integrity)**, local source, sdist, wheel, installed-package,
and native-extension checks are now release gates. Hosted multi-platform
artifact publication remains deliberately deferred pending an approved release
coordinate; that external release decision is not an open codebase-audit
finding.

## Status update (2026-07-17, Phase 4)

- **§2 three-backend numerical equivalence — resolved for the declared kernel
  contracts.** Commit `58663ab` corrected and R-pinned all ten shared fallback
  and Numba cluster-quality statistics. Commit `aeeb539` declared and tested
  real public dispatch per operation rather than assuming a global ladder.
  Commit `aa72c99` removed the eight foreign sequence-distance helpers and
  exports from `_wc_core`; sequence distances remain delegated to traminex.
  Commit `3ab7a6d` made those ownership, symbol-callability, and selection facts
  available through `get_backend_info()["capabilities"]`. The historical §2
  observations below remain as the audit evidence that motivated the work.
- Optional Numba was exercised with Numba 0.65.1 and NumPy 2.3.5. The accepted
  direct-kernel harness (`1802476`) and host-specific evidence (`49374b1`)
  separate correctness preflight, cold startup, and warm timing. They do not
  prove universal performance or artifact portability.
- **§6 remains unresolved for release but is deliberately deferred.** Local
  source and sdist checks do not establish hosted wheel or aarch64 support. Do
  not resume artifact work until traminex completes its further audit, its
  release-blocking findings are dispositioned, the maintainer declares it ready
  for its first release, and an approved clean-runner source coordinate exists.

## 1. The "210 passed, 7 skipped" test count

`archive/audits/2026/PARITY_AUDIT_REPORT_2026-04-07_20-32-43_JST.md` claims the full suite ran
`210 passed, 7 skipped, 2 warnings` on 2026-04-07 with the mandated interpreter
(`%USERPROFILE%\miniforge3\envs\datascience_core\python.exe`). This is a recorded claim, not
reproduced evidence.

- To verify: re-run `... -m pytest -ra` with the mandated interpreter and compare counts **and**
  the identity of the skipped tests (see §3 — which tests skip determines what "passed" means).
- A fresh observation from 2026-06-11 is recorded at the bottom of this file for comparison.

## 2. Three-backend numerical equivalence

`README.md` and `docs/ARCHITECTURE.md` describe a three-tier backend strategy
(compiled `weightedcluster._wc_core` → numba → pure Python). `tests/test_backend_parity_contracts.py`
exists, but this pass did not establish:

- Whether the three backends are numerically equivalent on every exported kernel, or only on the
  subset the contract tests cover.
- Whether any test environment has ever exercised **all three** tiers. Observed on 2026-06-11 in
  the mandated environment: `get_backend_info()` returns
  `{'compiled_core': 'available', 'numba': 'unavailable', 'pure_python': 'available'}` — i.e. the
  **numba tier was not importable in the mandated environment**, so numba-path tests presumably
  skip there. The April report does not state which backends its run exercised.
- Observable tier asymmetry worth auditing: `OMspell` distances have **no compiled and no numba
  implementation**. `src/weightedcluster/_seqdist.py` (around lines 158-179) attempts
  `from ._wc_core import seq_omspell`, but `cpp/wc_core.cpp` exports no `seq_omspell` symbol
  (see the `PYBIND11_MODULE` block, lines ~868-906), so the import always fails and execution
  always lands on pure-Python `_omspell.py` — even when the compiled core is present. The code
  comments acknowledge this ("Core implementation not yet available"). The audit should confirm
  whether OMspell results match R and whether the performance tier matters for any documented
  workflow.
  - **Update (2026-06-14): obsolete.** `_omspell.py` is deleted and `_seqdist.py`
    delegates OMSPELL (and all metrics) to traminex; no `from ._wc_core import
    seq_omspell` remains. See the Status update above and
    `tests/test_delegation_guard.py`.

## 3. Parity oracle provenance (circular-fixture risk)

The R-parity test suites do **not** use stored fixtures; they drive a **live R session via rpy2**
(`tests/r_parity_utils.py`). Three consequences the audit must check:

- `tests/r_parity_utils.py` loads the oracle with `importr("WeightedCluster")` (line ~229) —
  i.e. whatever WeightedCluster version is installed in the **system R library**, which is not
  necessarily the vendored ground truth `orig_R_pkg/WeightedCluster` (1.8-1). **Observed
  2026-06-11 in the mandated environment: the live oracle is WeightedCluster 2.0 on
  R 4.5.2 — a different major version than the vendored 1.8-1 the port claims to target.**
  The audit must decide whether 2.0 is an acceptable oracle for a 1.8-1 port (and whether the
  April run also used 2.0), or whether the vendored 1.8-1 must be installed for parity runs.
  - **Update (2026-06-14): addressed.** The oracle version is now recorded and
    guarded (`tests/test_oracle_provenance.py`); 2.0 was adopted as the oracle
    because the CRAN NEWS shows 1.8-1 → 2.0 adds only RARCAT (no ported function
    changed), so it is behaviourally identical for what we test (design D-1).
  - **Audit correction (2026-06-15, Phase 2a audit): the "no ported function
    changed" premise is FALSE.** `deparse(WeightedCluster:::wfcmdd)` on the
    installed 2.0 shows it was rewritten vs the vendored 1.8-1 source: a
    `if (is.null(weights)) weights <- rep(1, n)` guard; a refactored zero-distance
    membership branch; and a 2-step convergence test left as dead code (so 2.0
    converges on the 1-step delta only). 2.0 nonetheless remains a **sound** oracle:
    the rewrites are behaviourally equivalent on the paths the parity tests
    exercise, and the Python port matches the *installed 2.0*. The decision to use
    2.0 (D-1) stands; only the stated rationale was corrected (here, in
    `tests/r_parity_utils.py`, `tests/test_oracle_provenance.py`, `CHANGELOG.md`,
    and `docs/superpowers/CONTINUE.md`). Lesson for future audits: `deparse()` the
    installed function rather than trusting the NEWS for behavioural equivalence.
  - **Active resolution (2026-07-17):** this dated decision trail is historical.
    The exact target then became WeightedCluster 2.0 with TraMineR 2.2-13
    (superseded on 2026-09-21 by TraMineR 2.2-14); the shared
    fixture rejects every other version, and the canonical 2.0 source/archive is
    committed at the paths recorded in the top status update. Do not follow the
    historical instruction below to install 1.8-1 for active parity work.
- When rpy2 or the R package is unavailable, parity tests **skip** (`pytest.skip` in
  `r_parity_utils.py` line ~242) or **xfail** unless `WC_PARITY_STRICT=1` is set. A green suite in
  an R-less environment therefore proves nothing about parity. The audit should determine whether
  the April run's 7 skips include the parity sessions, and should run with `WC_PARITY_STRICT=1`
  in an environment where R + WeightedCluster 1.8-1 are confirmed present.
- Because the oracle is live R (not fixtures generated from the Python code), the classic
  circular-fixture risk is low **for these suites** — but `scripts/run_r_parity_phase0.py` and any
  "reference stats" mentioned in the April report should be checked for where their numbers come
  from before being trusted.
- `tests/test_parity_rpy2.py` is described in the April report as a "legacy path [that] still
  skips independently even though the newer parity harness is working" — verify whether it is
  redundant, broken, or covering something the new harness does not.

## 4. R API coverage claim

The April report claims "All upstream exported R entry points listed in
`orig_R_pkg/WeightedCluster/NAMESPACE` have a Python counterpart." This pass did not diff
`NAMESPACE` against the Python exports.

- To verify: enumerate `orig_R_pkg/WeightedCluster/NAMESPACE` exports and map each to a Python
  symbol; note semantic differences, not just name existence. `docs/API_INDEX.md` claims to be the
  canonical map and is the natural starting point, but it is itself unaudited.
- Known/asserted simplifications to scrutinize rather than accept: `wcCmpCluster(method='diana')`
  ("DIANA-style", per README Known Limitations), plotting helpers ("simplified"),
  `seqclustname` ("simplified" per its docstring), `wcCmpCluster` ("subset of R's wcCmpCluster"
  per its module docstring), `clustassoc` vs the unported R `dissmfacw` dependency (flagged
  AMBIGUOUS in the 2026-06-11 knowledge graph; see `graphify-out/GRAPH_REPORT.md`).

## 5. Default-parameter parity

The R defaults (e.g. `wcKMedoids` `method="PAMonce"`, `npass`, CLARA `sample_size` conventions
from Kaufman & Rousseeuw) versus the Python defaults were not compared in this pass. The phase-0
parity test pins `method="PAMonce", npass=0` on the R side while passing `npass=1, random_state=42`
on the Python side (`tests/test_r_parity_phase0.py` lines ~76-91) — the audit should confirm the
defaults-to-defaults behavior matches, not only the pinned configurations.

## 6. Official artifact integrity

`dist/` holds the wheel + sdist described as the official `0.0.1` release artifacts. They are
untracked (gitignored), so nothing in git history proves they were built from the current `main`.
The audit should rebuild from source with the mandated interpreter and compare, or at least record
the artifacts' hashes against a known-good build.

---

## Fresh observation (2026-06-11, tidy pass — reported as observation, not endorsement)

Environment: `%USERPROFILE%\miniforge3\envs\datascience_core\python.exe`, package imported from
the repo source tree (editable install), backends
`{'compiled_core': 'available', 'numba': 'unavailable', 'pure_python': 'available'}`.
Live R parity oracle: **WeightedCluster 2.0 on R 4.5.2** (rpy2 in ABI mode).

`pytest -ra` result: **210 passed, 7 skipped, 2 warnings in 18.13s** — numerically identical to
the April claim. Skip identities observed:

- 5 × "numba backend unavailable" (`test_backend_parity_contracts.py` ×3, `test_improvements.py` ×2)
- 1 × "Flexible-beta is very slow, test warning logic separately" (`test_improvements.py`)
- 1 × "rpy2 not available" (`test_parity_rpy2.py` — the legacy path; note the *newer* parity
  suites did initialize an R session in the same run, so this skip reason deserves scrutiny)

No XFAIL entries appeared in the `-ra` summary, so the rpy2-driven parity comparisons in the
newer suites ran and matched — against the 2.0 oracle (see §3). The numba tier was exercised by
zero tests in this environment (see §2).

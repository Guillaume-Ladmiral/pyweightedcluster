# CLAUDE.md — pyweightedcluster

This project's substantive guidance is tool-neutral and lives in files that
**every** agent should read, not just Claude:

- **`AGENTS.md`** — environment policy (interpreter, test/build/lint commands)
  and this project's role in the family.
- **`ECOSYSTEM.md`** — points to the four-project family charter (traminex /
  pyweightedcluster / py_seqplot / flexscanx; one master copy, in
  `../PROJECT_TRAMINEX/ECOSYSTEM.md`) and holds this project's source-of-truth rule.
- **`DUPLICATES.md`** — the function-by-function duplicate inventory and the
  convergence plan.

Start by reading `AGENTS.md` and `ECOSYSTEM.md`. Note in particular: this
project's genuinely own code is the weighted PAM/k-medoids clustering, cluster
quality, fuzzy/CLARA, `seqnull`, and the WeightedCluster-facing structural
adapters. Sequence **distances/costs/discrepancy/tree growth** belong to
traminex: property-tree growth now delegates to traminex's public `disstree` and
`seqtree` implementations, and `clustassoc` delegates discrepancy terms to
`dissmfacw`. Do not perform further cross-project migrations without explicit
maintainer authorization.

Use the exact interpreter and commands in `AGENTS.md`. R-parity and test runs
must retain the explicit `LC_ALL=C.UTF-8 LANG=C.UTF-8` prefix; this collation
contract is part of reproducible R/ICU comparison, not an optional shell hint.

## Codebase knowledge

**Updated 2026-09-02:** a codebase-memory graph index for this repository now
exists and may be used. It was observed indexed at 2026-08-31 13:02:53 UTC —
current at commit `0716722` — with 11,498 nodes and 36,384 edges, and appears
to refresh automatically as a watched project. Its only recorded parse gaps
are the vendored R package's C++ macro bodies under `orig_R_pkg/*/src/` and
two JSON fixtures; treat graph coverage as best-effort and verify cited paths
against the source. (Historical record: as of 2026-08-24 no index existed —
finding F-09 of that audit — and the then-dangling instruction to use one was
removed rather than made true by creating it. The maintainer authorized index
use on 2026-09-02.)

Discover code with ordinary reading and search; use the graph index for
structural queries when helpful. The `graphify-out/` directory, a
historical snapshot generated on 2026-08-10, was removed on 2026-10-26 (family
supervision plan, owner decisions D2 and D3); it remains in git history and was
never current authority.

# Project Family Charter — pointer for pyweightedcluster

**The family charter of traminex / pyweightedcluster / py_seqplot / flexscanx has
one copy: `ECOSYSTEM.md` in the sibling checkout `../PROJECT_TRAMINEX`.** Read it
for the domain, the layering, the ownership map, the working rule and the source of
truth of each project. The family's supervision layer (dated decisions, open items,
inventory and the family check) is in the supervision repository
`PROJECT_SEQANALYSIS_and_SPATSCAN` (the four projects' parent folder), file `docs/CHARTER.md`.

This file stopped being a copy on 2026-10-26 (family supervision plan, approved by
the owner): the copies had drifted from the master by 69 to 72 lines. It keeps only
what is specific to this project. A consumer repository may still record a newer
consumer-observed correction here when the master lags an independently verified
sibling rename or status; any later change to the master
must not overwrite newer sibling state, and such a correction is then reported to
the master.

## Domain

This family is social-science statistical software, not biology or genomics. A
"sequence" is a categorical state trajectory over time; "optimal matching",
"substitution costs" and "indel costs" are edit-distance parameters over social
states. `weightedcluster` clusters such trajectories.

## This project in the family

traminex (`../PROJECT_TRAMINEX`) is the base. pyweightedcluster builds on it and owns
clustering of sequences and cluster quality; the visualization layer is py_seqplot
(`PROJECT_PY_SEQPLOT`, package `py_seqplot`); flexscanx is the spatial-scan project.

## Source of truth for pyweightedcluster

WeightedCluster 2.0 is pyweightedcluster's canonical R source: its readable tree is
`orig_R_pkg/WeightedCluster`, and its exact CRAN artifact is
`orig_R_pkg/WeightedCluster_2.0.tar.gz` (SHA-256
`6a03fbbc090ac92498a410faab1aeb88d686ff7a81dcfdcb97edacda44f393cc`). The exact live
companion oracle is TraMineR 2.2-14. The prior WeightedCluster 1.8-1 source is
historical evidence at `orig_R_pkg/WeightedCluster-1.8-1`, with its exact archive at
`docs/legacy/WeightedCluster_1.8-1.tar.gz` (SHA-256
`daf9987cee5ac9e706a39e72f54a64e49e9fe91b0bc3cfb5e5a94d5076b2df5b`). When a Python
copy disagrees with its R original, the R original wins, except for things
deliberately added or changed, which must be written down as deliberate (see
`DUPLICATES.md`).

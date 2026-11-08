# Release Notes - PyWeightedCluster v0.0.1

**Release Type:** Initial Public Release  
**Status:** Canonical release notes for the Python port

## Overview

`v0.0.1` is the first and only official release of this Python port.

The project provides Python implementations aligned with the R `WeightedCluster` package naming and reference manual.

## Included Scope

- Core weighted clustering:
  - `wcKMedoids`
  - `wcKMedRange`
  - `wcClusterQuality`
  - `wcSilhouetteObs`
- Sequence clustering and validation:
  - `seqclararange`
  - `bootclustrange`
  - `seqnull`
  - `seqnullcqi`
- Property and association workflows:
  - `seqpropclust`
  - `wcPropertyClustering`
  - `clustassoc`
- Utility and plotting APIs exported by `weightedcluster`
- Compiled backend plus fallback backends
- Test suite and documentation

## Versioning Clarification

- `v0.0.1` is the only official release version.
- Any references in repository documents to `0.2.0`, `0.3.0`, or `0.4.0` were draft/internal artifacts and not completed releases.

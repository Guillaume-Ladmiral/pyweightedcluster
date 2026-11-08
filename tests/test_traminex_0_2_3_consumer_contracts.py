"""Consumer regressions for the re-pin to traminex 0.2.3 (2026-11-08).

The reviewed manifest update (``DUPLICATES.md``, "2026-11-08 reviewed manifest update to traminex 0.2.3")
found two changes of traminex's delta that reach this package, both through the ``seqdef`` and
``seqformat`` calls of ``seqclustname``. They were measured against traminex 0.2.2 and 0.2.3 (family
supervision repository, ``inventory/generated/patch-2026-11-08/traminex-delta/``):

- complex-valued states are written as R's ``as.character`` writes them (``1+1i``); traminex 0.2.2
  wrote Python's form (``(1+1j)``);
- rows given as matrices of unequal widths are refused as not a 2-D table of states; traminex 0.2.2
  made states of their printed forms (``"['A' 'B' 'C']"``).
"""

from __future__ import annotations

import numpy as np
import pytest

from weightedcluster import seqclustname

_DISS = np.array([[0.0, 1.0], [1.0, 0.0]])


def test_seqclustname_labels_complex_states_as_r_writes_them():
    rows = [[1 + 1j, 1 + 1j, 2 + 0j], [2 + 0j, 2 + 0j, 1 - 1j]]

    labels = seqclustname(rows, group=[1, 2], diss=_DISS)

    assert list(labels) == ["1+1i/2-2+0i/1", "2+0i/2-1-1i/1"]


def test_seqclustname_refuses_rows_given_as_matrices():
    rows = [
        np.array([["A", "B"], ["A", "B"]], dtype=object),
        np.array([["A", "B", "C"], ["A", "B", "C"]], dtype=object),
    ]

    with pytest.raises(ValueError, match="seqdef expects a 2D table of states"):
        seqclustname(rows, group=[1, 2], diss=_DISS)

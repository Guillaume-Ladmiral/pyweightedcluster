from __future__ import annotations

import re
from importlib.metadata import version
from pathlib import Path

import weightedcluster

INIT_SRC = Path(weightedcluster.__file__).resolve()


def test_version_matches_distribution_metadata():
    # The package __version__ must equal the installed distribution version, so
    # pyproject.toml is the single source of truth.
    assert weightedcluster.__version__ == version("pyweightedcluster")


def test_version_is_derived_not_hardcoded():
    # __version__ must be computed from importlib.metadata, not a hand-edited
    # literal (which is what caused the docs/version drift).
    src = INIT_SRC.read_text(encoding="utf-8")
    assert "importlib.metadata" in src
    assert re.search(r"""__version__\s*=\s*['"]\d""", src) is None

from __future__ import annotations

import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

EXECUTABLE_USER_DOC_FILES = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "README.md",
    REPO_ROOT / "docs" / "API_INDEX.md",
    REPO_ROOT / "docs" / "ARCHITECTURE.md",
    REPO_ROOT / "docs" / "WORKFLOW_GUIDE.md",
    REPO_ROOT / "docs" / "R_MIGRATION_GUIDE.md",
    REPO_ROOT / "docs" / "vignettes" / "README.md",
    *sorted((REPO_ROOT / "docs" / "vignettes").glob("*.qmd")),
]


def _extract_python_block_after_marker(text: str, marker: str) -> str:
    marker_idx = text.find(marker)
    if marker_idx < 0:
        raise AssertionError(f"marker not found: {marker!r}")
    fence_idx = text.find("```python", marker_idx)
    if fence_idx < 0:
        raise AssertionError(f"python fence not found after marker: {marker!r}")
    code_start = text.find("\n", fence_idx)
    code_end = text.find("```", code_start + 1)
    if code_start < 0 or code_end < 0:
        raise AssertionError(f"unterminated python fence after marker: {marker!r}")
    return text[code_start + 1 : code_end]


def _load_phase0_parity_script():
    script_path = REPO_ROOT / "scripts" / "run_r_parity_phase0.py"
    spec = importlib.util.spec_from_file_location("phase0_parity_script", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase0_parity_helper_script_emits_json_snapshot(monkeypatch):
    mod = _load_phase0_parity_script()
    stdout = io.StringIO()

    monkeypatch.setattr("sys.argv", ["run_r_parity_phase0.py", "--json"])
    with redirect_stdout(stdout):
        exit_code = int(mod.main())

    assert exit_code == 0
    payload = json.loads(stdout.getvalue())
    assert isinstance(payload.get("available"), bool)
    if payload["available"]:
        assert payload["passed"] is True
        assert payload["problems"] == []
        assert "wcKMedoids_partition_match" in payload
        assert "wcClusterQuality_stats_absdiff" in payload
        assert "wcSilhouetteObs_max_absdiff" in payload
        assert "as_clustrange_max_absdiff" in payload
    else:
        assert isinstance(payload.get("reason"), str)
        assert payload["reason"]


def test_phase0_parity_helper_fails_when_the_oracle_is_unavailable(monkeypatch):
    mod = _load_phase0_parity_script()
    stdout = io.StringIO()
    real_import = mod.importlib.import_module

    def fake_import(name):
        if name == "tests.r_parity_utils":
            return SimpleNamespace(
                canonical_partition=lambda labels: tuple(labels),
                create_r_parity_session=lambda: (None, "deliberately unavailable oracle"),
                exact_oracle_version_problems=lambda _versions: [],
            )
        if name == "weightedcluster":
            return SimpleNamespace(
                as_clustrange=None,
                wcClusterQuality=None,
                wcKMedoids=None,
                wcSilhouetteObs=None,
            )
        return real_import(name)

    monkeypatch.setattr(mod.importlib, "import_module", fake_import)
    monkeypatch.setattr("sys.argv", ["run_r_parity_phase0.py", "--json"])
    with redirect_stdout(stdout):
        exit_code = int(mod.main())

    payload = json.loads(stdout.getvalue())
    assert exit_code != 0
    assert payload == {
        "available": False,
        "passed": False,
        "reason": "deliberately unavailable oracle",
    }


def test_phase0_parity_helper_classifies_numerical_mismatch_as_failure():
    mod = _load_phase0_parity_script()
    evaluator = getattr(mod, "_parity_problems", None)
    assert callable(evaluator), "the CLI needs a centralized pass/fail contract"
    payload = {
        "wcKMedoids_partition_match": True,
        "wcKMedoids_discriminating_partition_match": True,
        "wcKMedoids_discriminating_objective_absdiff": 0.0,
        "wcKMedoids_stats_absdiff": {"PBC": 0.0},
        "wcClusterQuality_stats_absdiff": {"PBC": 1.0},
        "wcSilhouetteObs_max_absdiff": {"ASW": 0.0, "ASWw": 0.0},
        "as_clustrange_max_absdiff": 0.0,
    }

    assert evaluator(payload) == ["wcClusterQuality statistic PBC exceeds tolerance"]


def test_readme_backend_info_example_executes():
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    code = _extract_python_block_after_marker(text, "Check backend availability:")
    ns: dict[str, object] = {}
    exec(code, ns, ns)

    info = ns["get_backend_info"]()
    assert set(info.keys()) == {"compiled_core", "numba", "pure_python", "capabilities"}
    assert info["pure_python"] == "available"
    capabilities = info["capabilities"]
    assert capabilities["weighted_medoids"]["public_dispatch"] == [
        "compiled",
        "pure_python",
    ]
    assert capabilities["cluster_quality"]["public_dispatch"] == [
        "compiled",
        "numba",
        "pure_python",
    ]
    assert capabilities["observation_silhouette"]["selection"] == "python_only"
    assert capabilities["seqnull_spell_models"]["selection"] == "python_only"
    assert capabilities["seqnull_markov"]["selection"] == "conditional"
    assert capabilities["sequence_distances"]["owner"] == "traminex"
    assert capabilities["sequence_distances"]["selection"] == "delegated"


def test_readme_quick_example_executes():
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    code = _extract_python_block_after_marker(text, "## Quick Example")
    ns: dict[str, object] = {}
    exec(code, ns, ns)

    cr = ns["cr"]
    bc = ns["bc"]
    assert cr["kvals"] == [2, 3]
    assert "cluster2" in cr["clustering"]
    assert bc["kvals"] == [2, 3]
    assert "ASW" in bc["meant"]
    assert "ASW" in bc["stderr"]


def test_workflow_guide_sequence_bootstrap_and_null_examples_execute():
    text = (REPO_ROOT / "docs" / "WORKFLOW_GUIDE.md").read_text(encoding="utf-8")
    seq_code = _extract_python_block_after_marker(text, "## Sequence-First Workflow")
    boot_code = _extract_python_block_after_marker(
        text, "## Bootstrap Validation (`bootclustrange`)"
    )
    null_code = _extract_python_block_after_marker(text, "## Null-Model Validation (`seqnullcqi`)")

    ns: dict[str, object] = {}
    exec(seq_code, ns, ns)
    exec(boot_code, ns, ns)
    exec(null_code, ns, ns)

    cr = ns["cr"]
    bc = ns["bc"]
    nc = ns["nc"]
    assert cr["kvals"] == [2, 3]
    assert bc["kvals"] == [2, 3]
    assert tuple(nc["stats"]["ASW"].shape) == (20, 2)


def test_r_migration_guide_sequence_workflow_example_executes():
    text = (REPO_ROOT / "docs" / "R_MIGRATION_GUIDE.md").read_text(encoding="utf-8")
    code = _extract_python_block_after_marker(
        text, "## Example: R-style Sequence Workflow in Python"
    )

    ns: dict[str, object] = {}
    exec(code, ns, ns)

    cr = ns["cr"]
    bc = ns["bc"]
    nc = ns["nc"]
    assert cr["kvals"] == [2, 3]
    assert bc["kvals"] == [2, 3]
    assert tuple(nc["stats"]["ASW"].shape) == (20, 2)


def test_active_document_fences_have_complete_explicit_classification():
    from tests.docs_audit_helpers import validate_example_classifications

    assert validate_example_classifications(EXECUTABLE_USER_DOC_FILES) == []


def test_example_classification_reports_missing_and_stale_pair_ids(tmp_path):
    from tests.docs_audit_helpers import validate_example_classifications

    source = tmp_path / "examples.qmd"
    source.write_text(
        """# Examples

```{python}
print("unclassified")
```

```{python}
# docs-audit: expensive pair=missing-smoke
print("full workload")
```
""",
        encoding="utf-8",
    )

    assert validate_example_classifications([source]) == [
        "examples.qmd:3: Python fence has no docs-audit classification",
        "examples.qmd:7: expensive example references unknown bounded pair id " "'missing-smoke'",
    ]


def test_broken_document_example_reports_document_and_source_line(tmp_path):
    from tests.docs_audit_helpers import DocumentationExampleError, execute_document_examples

    source = tmp_path / "broken.md"
    source.write_text(
        """# Broken

```python
# docs-audit: smoke id=broken-smoke
raise RuntimeError("intentional failure")
```
""",
        encoding="utf-8",
    )

    with pytest.raises(
        DocumentationExampleError,
        match=r"broken\.md:3: smoke Python example failed: intentional failure",
    ):
        execute_document_examples([source], work_root=tmp_path / "work")


def test_all_active_documentation_blocks_execute_or_receive_class_specific_validation(tmp_path):
    from tests.docs_audit_helpers import execute_document_examples

    summary = execute_document_examples(
        EXECUTABLE_USER_DOC_FILES, work_root=tmp_path / "documentation-examples"
    )

    assert summary.executed_count == 55
    assert summary.expensive_count == 10
    assert summary.display_only_count == 7
    assert summary.shell_setup_count == 5
    assert summary.validated_count == 77
    assert summary.documents == tuple(EXECUTABLE_USER_DOC_FILES)

from __future__ import annotations

import ast
import doctest
import html
import importlib
import os
import pkgutil
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote_to_bytes, urlsplit

_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_REFERENCE_DEFINITION_RE = re.compile(
    r"^ {0,3}\[([^]]+)\]:\s*(?:<([^>]+)>|(\S+))" r"(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*$"
)
_INLINE_LINK_RE = re.compile(
    r"!?\[([^]]*)\]\(\s*(<[^>]*>|[^)\s]+)" r"(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*\)"
)
_REFERENCE_LINK_RE = re.compile(r"!?\[([^]]+)\]\[([^]]*)\]")
_SHORTCUT_REFERENCE_RE = re.compile(r"(?<!!)\[([^]]+)\]")
_AUTOLINK_RE = re.compile(r"<([A-Za-z][A-Za-z0-9+.-]*:[^ <>]+)>")
_INLINE_CODE_RE = re.compile(r"(`+)(.+?)\1")
_CODE_PATH_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
_HEADING_RE = re.compile(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_EXPLICIT_ANCHOR_RE = re.compile(
    r"<a\s+[^>]*(?:id|name)\s*=\s*['\"]([^'\"]+)['\"][^>]*>", re.IGNORECASE
)
_QUARTO_HEADING_ID_RE = re.compile(r"\s*\{[^}]*#([^\s.}]+)[^}]*\}\s*$")
_MALFORMED_PERCENT_RE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_DOCS_AUDIT_MARKER_RE = re.compile(r"^\s*#\s*docs-audit:\s*([\w-]+)(.*)$")

_ROOT_DOCUMENTATION_PREFIXES = ("archive/", "docs/", "graphify-out/", "orig_R_pkg/")


def _opens_a_fence(marker: str, info: str) -> bool:
    """Apply CommonMark's rule that a backtick fence carries no backtick.

    A line such as ```` ```` ```python path=null start=null ```` ```` is prose
    quoting fence syntax with inline code spans, not a fence opener.  Without
    this rule the scanner treats it as opening a four-backtick fence that
    nothing closes, and reports the document as having an unterminated fence.
    """
    return marker[0] != "`" or "`" not in info


@dataclass(frozen=True)
class DocumentLink:
    source: Path
    line: int
    target: str
    syntax: str
    kind: str


@dataclass(frozen=True)
class IndexedDocumentPath:
    source: Path
    line: int
    target: str
    resolved: Path
    expected_kind: str


@dataclass(frozen=True)
class FencedBlock:
    source: Path
    line: int
    info: str
    language: str
    code: str
    classification: str | None
    metadata: tuple[tuple[str, str], ...]
    marker_error: str | None = None

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)


@dataclass(frozen=True)
class ExampleRunSummary:
    documents: tuple[Path, ...]
    executed_count: int
    expensive_count: int
    display_only_count: int
    shell_setup_count: int
    validated_count: int


@dataclass(frozen=True)
class PackageDoctestSummary:
    module_count: int
    attempted: int
    failed: int


def run_package_doctests(package) -> PackageDoctestSummary:
    """Discover and run doctests from every importable package submodule."""
    names = [package.__name__]
    names.extend(
        f"{package.__name__}.{info.name}" for info in pkgutil.iter_modules(package.__path__)
    )
    attempted = 0
    failed = 0
    for name in sorted(names):
        result = doctest.testmod(importlib.import_module(name), verbose=False)
        attempted += result.attempted
        failed += result.failed
    return PackageDoctestSummary(len(names), attempted, failed)


class DocumentationExampleError(RuntimeError):
    pass


def _normalize_reference_label(label: str) -> str:
    return " ".join(label.split()).casefold()


def _lines_outside_fences(path: Path) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    fence_character: str | None = None
    fence_length = 0

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = _FENCE_RE.match(line)
        if match:
            marker = match.group(1)
            if fence_character is None:
                if _opens_a_fence(marker, match.group(2)):
                    fence_character = marker[0]
                    fence_length = len(marker)
                    continue
            elif marker[0] == fence_character and len(marker) >= fence_length:
                fence_character = None
                fence_length = 0
                continue
        if fence_character is None:
            lines.append((line_number, line))

    return lines


def _mask_inline_code(line: str) -> str:
    return _INLINE_CODE_RE.sub(lambda match: " " * len(match.group(0)), line)


def _mask_matches(line: str, matches: list[re.Match[str]]) -> str:
    characters = list(line)
    for match in matches:
        characters[match.start() : match.end()] = " " * (match.end() - match.start())
    return "".join(characters)


def _clean_destination(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    return html.unescape(target)


def _classify_target(target: str) -> str:
    scheme = urlsplit(target).scheme.casefold()
    if scheme in {"http", "https"}:
        return "external"
    if scheme == "mailto":
        return "mailto"
    if scheme:
        return "other-scheme"
    return "local"


def collect_document_links(paths: list[Path] | tuple[Path, ...]) -> list[DocumentLink]:
    links: list[DocumentLink] = []

    for path in paths:
        source_lines = _lines_outside_fences(path)
        definitions: dict[str, str] = {}
        for _, line in source_lines:
            definition = _REFERENCE_DEFINITION_RE.match(_mask_inline_code(line))
            if definition:
                target = definition.group(2) or definition.group(3)
                definitions[_normalize_reference_label(definition.group(1))] = _clean_destination(
                    target
                )

        for line_number, original_line in source_lines:
            line = _mask_inline_code(original_line)
            if _REFERENCE_DEFINITION_RE.match(line):
                continue

            consumed: list[re.Match[str]] = []
            for match in _INLINE_LINK_RE.finditer(line):
                target = _clean_destination(match.group(2))
                links.append(
                    DocumentLink(path, line_number, target, "inline", _classify_target(target))
                )
                consumed.append(match)

            line_without_inline = _mask_matches(line, consumed)
            consumed = []
            for match in _REFERENCE_LINK_RE.finditer(line_without_inline):
                label = match.group(2) or match.group(1)
                target = definitions.get(_normalize_reference_label(label))
                if target is not None:
                    links.append(
                        DocumentLink(
                            path,
                            line_number,
                            target,
                            "reference",
                            _classify_target(target),
                        )
                    )
                consumed.append(match)

            line_without_references = _mask_matches(line_without_inline, consumed)
            for match in _SHORTCUT_REFERENCE_RE.finditer(line_without_references):
                target = definitions.get(_normalize_reference_label(match.group(1)))
                if target is not None:
                    links.append(
                        DocumentLink(
                            path,
                            line_number,
                            target,
                            "shortcut-reference",
                            _classify_target(target),
                        )
                    )

            for match in _AUTOLINK_RE.finditer(line_without_references):
                target = _clean_destination(match.group(1))
                links.append(
                    DocumentLink(path, line_number, target, "autolink", _classify_target(target))
                )

    return links


def _strip_heading_markup(heading: str) -> str:
    heading = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", heading)
    heading = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", heading)
    heading = re.sub(r"<[^>]+>", "", heading)
    heading = heading.replace("`", "")
    return re.sub(r"[*~]", "", heading).strip()


def _github_heading_slug(heading: str) -> str:
    text = _strip_heading_markup(heading).casefold()
    text = re.sub(r"[^\w\-\s]", "", text, flags=re.UNICODE)
    return re.sub(r"\s", "-", text.strip())


def _quarto_heading_slug(heading: str) -> str:
    text = _strip_heading_markup(heading).casefold()
    text = "".join(
        character
        for character in text
        if character.isalnum() or character.isspace() or character in {"_", "-", "."}
    )
    text = re.sub(r"\s+", "-", text.strip())
    first_letter = next(
        (index for index, character in enumerate(text) if character.isalpha()), None
    )
    if first_letter is None:
        return "section"
    return text[first_letter:]


def _document_anchors(path: Path) -> set[str]:
    source_lines = _lines_outside_fences(path)
    anchors: set[str] = set()
    slug_counts: dict[str, int] = {}

    for _, line in source_lines:
        anchors.update(_EXPLICIT_ANCHOR_RE.findall(line))
        heading_match = _HEADING_RE.match(line)
        if not heading_match:
            continue

        heading = heading_match.group(1)
        explicit_id = _QUARTO_HEADING_ID_RE.search(heading)
        if explicit_id:
            anchors.add(explicit_id.group(1))
            heading = heading[: explicit_id.start()].rstrip()

        slug = (
            _quarto_heading_slug(heading)
            if path.suffix.casefold() == ".qmd"
            else _github_heading_slug(heading)
        )
        duplicate_index = slug_counts.get(slug, 0)
        slug_counts[slug] = duplicate_index + 1
        anchors.add(slug if duplicate_index == 0 else f"{slug}-{duplicate_index}")

    return anchors


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _decode_url_component(value: str) -> str:
    return unquote_to_bytes(value).decode("utf-8", errors="strict")


def validate_document_links(paths: list[Path] | tuple[Path, ...], *, repo_root: Path) -> list[str]:
    root = repo_root.resolve()
    issues: list[str] = []
    anchor_cache: dict[Path, set[str]] = {}

    for link in collect_document_links(paths):
        if link.kind != "local":
            continue

        source_display = _display_path(link.source, root)
        if _MALFORMED_PERCENT_RE.search(link.target):
            issues.append(
                f"{source_display}:{link.line}: malformed percent escape in local target "
                f"'{link.target}'"
            )
            continue

        parsed = urlsplit(link.target)
        try:
            decoded_path = _decode_url_component(parsed.path)
            decoded_fragment = _decode_url_component(parsed.fragment)
        except UnicodeDecodeError:
            issues.append(
                f"{source_display}:{link.line}: malformed UTF-8 escape in local target "
                f"'{link.target}'"
            )
            continue

        target_path = link.source if not decoded_path else link.source.parent / decoded_path
        resolved = target_path.resolve()
        if not resolved.is_relative_to(root):
            issues.append(
                f"{source_display}:{link.line}: local target escapes repository root: "
                f"'{link.target}'"
            )
            continue
        if not resolved.exists():
            issues.append(
                f"{source_display}:{link.line}: missing local file for target '{link.target}'"
            )
            continue
        if resolved.is_dir():
            issues.append(
                f"{source_display}:{link.line}: local target is a directory, not a file: "
                f"'{link.target}'"
            )
            continue

        if decoded_fragment:
            anchors = anchor_cache.setdefault(resolved, _document_anchors(resolved))
            if decoded_fragment not in anchors:
                target_display = _display_path(resolved, root)
                issues.append(
                    f"{source_display}:{link.line}: missing anchor '{decoded_fragment}' in "
                    f"{target_display} for target '{link.target}'"
                )

    return issues


def _looks_like_documentation_path(value: str) -> bool:
    if not value or value.startswith("/") or any(character.isspace() for character in value):
        return False
    path_part = value.split("#", 1)[0]
    if path_part.endswith((".md", ".qmd")):
        return True
    return path_part.endswith("/") and path_part.startswith(_ROOT_DOCUMENTATION_PREFIXES)


def _resolve_indexed_path(source: Path, target: str, repo_root: Path) -> Path:
    decoded = _decode_url_component(urlsplit(target).path)
    is_root_index = source.resolve() == (repo_root / "README.md").resolve()
    is_docs_index_top_level_readme = (
        source.resolve() == (repo_root / "docs" / "README.md").resolve() and decoded == "README.md"
    )
    if (
        is_root_index
        or is_docs_index_top_level_readme
        or decoded.startswith(_ROOT_DOCUMENTATION_PREFIXES)
    ):
        return (repo_root / decoded).resolve()
    return (source.parent / decoded).resolve()


def collect_indexed_document_paths(
    index_paths: list[Path] | tuple[Path, ...], *, repo_root: Path
) -> list[IndexedDocumentPath]:
    entries: list[IndexedDocumentPath] = []
    for source in index_paths:
        for line_number, line in _lines_outside_fences(source):
            for match in _CODE_PATH_RE.finditer(line):
                target = match.group(1).strip()
                if not _looks_like_documentation_path(target):
                    continue
                resolved = _resolve_indexed_path(source, target, repo_root)
                expected_kind = "directory" if urlsplit(target).path.endswith("/") else "file"
                entries.append(
                    IndexedDocumentPath(source, line_number, target, resolved, expected_kind)
                )
    return entries


def validate_indexed_document_paths(
    index_paths: list[Path] | tuple[Path, ...], *, repo_root: Path
) -> list[str]:
    root = repo_root.resolve()
    issues: list[str] = []

    for entry in collect_indexed_document_paths(index_paths, repo_root=root):
        source_display = _display_path(entry.source, root)
        if _MALFORMED_PERCENT_RE.search(entry.target):
            issues.append(
                f"{source_display}:{entry.line}: malformed percent escape in indexed path "
                f"'{entry.target}'"
            )
            continue
        if not entry.resolved.is_relative_to(root):
            issues.append(
                f"{source_display}:{entry.line}: indexed path escapes repository root: "
                f"'{entry.target}'"
            )
            continue
        if not entry.resolved.exists():
            issues.append(
                f"{source_display}:{entry.line}: missing indexed {entry.expected_kind} "
                f"'{entry.target}'"
            )
            continue
        if entry.expected_kind == "file" and not entry.resolved.is_file():
            issues.append(
                f"{source_display}:{entry.line}: indexed path is not a file: '{entry.target}'"
            )
        if entry.expected_kind == "directory" and not entry.resolved.is_dir():
            issues.append(
                f"{source_display}:{entry.line}: indexed path is not a directory: "
                f"'{entry.target}'"
            )

    return issues


def _fence_language(info: str) -> str:
    stripped = info.strip()
    if stripped.startswith("{") and "}" in stripped:
        stripped = stripped[1 : stripped.index("}")]
    return stripped.split(maxsplit=1)[0].split(",", 1)[0].casefold() if stripped else ""


def _parse_classification(code: str) -> tuple[str | None, tuple[tuple[str, str], ...], str | None]:
    markers = [match for line in code.splitlines() if (match := _DOCS_AUDIT_MARKER_RE.match(line))]
    if not markers:
        return None, (), None
    if len(markers) > 1:
        return None, (), "fence has more than one docs-audit classification"

    marker = markers[0]
    try:
        tokens = shlex.split(marker.group(2).strip())
    except ValueError as exc:
        return marker.group(1), (), f"invalid docs-audit metadata: {exc}"

    metadata: list[tuple[str, str]] = []
    for token in tokens:
        if "=" not in token:
            return marker.group(1), (), f"invalid docs-audit metadata token '{token}'"
        key, value = token.split("=", 1)
        if not key or not value:
            return marker.group(1), (), f"invalid docs-audit metadata token '{token}'"
        metadata.append((key, value))

    if len({key for key, _ in metadata}) != len(metadata):
        return marker.group(1), (), "docs-audit metadata contains a duplicate key"
    return marker.group(1), tuple(metadata), None


def collect_fenced_blocks(paths: list[Path] | tuple[Path, ...]) -> list[FencedBlock]:
    blocks: list[FencedBlock] = []
    for source in paths:
        lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
        fence_character: str | None = None
        fence_length = 0
        opening_line = 0
        info = ""
        code_lines: list[str] = []

        for line_number, line in enumerate(lines, start=1):
            match = _FENCE_RE.match(line.rstrip("\r\n"))
            if fence_character is None:
                if not match or not _opens_a_fence(match.group(1), match.group(2)):
                    continue
                marker = match.group(1)
                fence_character = marker[0]
                fence_length = len(marker)
                opening_line = line_number
                info = match.group(2).strip()
                code_lines = []
                continue

            if match:
                marker = match.group(1)
                if marker[0] == fence_character and len(marker) >= fence_length:
                    code = "".join(code_lines)
                    classification, metadata, marker_error = _parse_classification(code)
                    blocks.append(
                        FencedBlock(
                            source=source,
                            line=opening_line,
                            info=info,
                            language=_fence_language(info),
                            code=code,
                            classification=classification,
                            metadata=metadata,
                            marker_error=marker_error,
                        )
                    )
                    fence_character = None
                    fence_length = 0
                    continue
            code_lines.append(line)

        if fence_character is not None:
            display = _example_display_path(source)
            raise DocumentationExampleError(f"{display}:{opening_line}: unterminated code fence")

    return blocks


def _example_display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def validate_example_classifications(paths: list[Path] | tuple[Path, ...]) -> list[str]:
    blocks = collect_fenced_blocks(paths)
    issues: list[str] = []
    bounded_ids: dict[str, FencedBlock] = {}

    for block in blocks:
        if block.language not in {"python", "bash", "sh", "shell"}:
            continue
        display = _example_display_path(block.source)
        if block.marker_error:
            issues.append(f"{display}:{block.line}: {block.marker_error}")
            continue
        if block.classification is None:
            issues.append(
                f"{display}:{block.line}: {block.language.capitalize()} fence has no "
                "docs-audit classification"
            )
            continue

        metadata = block.metadata_dict
        if block.language == "python":
            allowed = {"display-only", "expensive", "sequential", "smoke"}
            if block.classification not in allowed:
                issues.append(
                    f"{display}:{block.line}: invalid Python docs-audit classification "
                    f"'{block.classification}'"
                )
                continue
            if block.classification == "display-only" and "reason" not in metadata:
                issues.append(
                    f"{display}:{block.line}: display-only example requires reason metadata"
                )
            if block.classification == "expensive" and "pair" not in metadata:
                issues.append(f"{display}:{block.line}: expensive example requires pair metadata")
            if block.classification in {"smoke", "sequential", "expensive"}:
                if block.source.suffix == ".qmd" and not block.info.startswith("{python"):
                    issues.append(
                        f"{display}:{block.line}: executable QMD example must use a "
                        "Quarto '{{python}}' fence"
                    )
                if re.search(r"^\s*#\|\s*eval:\s*false\s*$", block.code, re.MULTILINE):
                    issues.append(
                        f"{display}:{block.line}: executable example cannot set eval: false"
                    )
            bounded_id = metadata.get("id")
            if bounded_id:
                if block.classification not in {"smoke", "sequential"}:
                    issues.append(
                        f"{display}:{block.line}: bounded id '{bounded_id}' must belong to a "
                        "smoke or sequential example"
                    )
                elif bounded_id in bounded_ids:
                    issues.append(
                        f"{display}:{block.line}: duplicate bounded example id '{bounded_id}'"
                    )
                else:
                    bounded_ids[bounded_id] = block
        elif block.classification != "shell-setup":
            issues.append(
                f"{display}:{block.line}: shell fence must use the shell-setup classification"
            )
        elif "reason" not in metadata:
            issues.append(f"{display}:{block.line}: shell-setup example requires reason metadata")

    for block in blocks:
        if block.language != "python" or block.classification != "expensive":
            continue
        pair_id = block.metadata_dict.get("pair")
        if pair_id and pair_id not in bounded_ids:
            display = _example_display_path(block.source)
            issues.append(
                f"{display}:{block.line}: expensive example references unknown bounded pair id "
                f"'{pair_id}'"
            )

    return issues


def _assert_smoke_code_is_bounded(block: FencedBlock) -> None:
    display = _example_display_path(block.source)
    try:
        tree = ast.parse(block.code, filename=f"{display}:{block.line}", mode="exec")
    except SyntaxError as exc:
        raise DocumentationExampleError(
            f"{display}:{block.line}: {block.classification} Python example failed: {exc.msg}"
        ) from exc

    prohibited_import_roots = {"httpx", "requests", "socket", "subprocess", "urllib"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
            if roots & prohibited_import_roots:
                raise DocumentationExampleError(
                    f"{display}:{block.line}: ordinary example imports prohibited network or "
                    f"process module '{sorted(roots & prohibited_import_roots)[0]}'"
                )
        if isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            if root in prohibited_import_roots:
                raise DocumentationExampleError(
                    f"{display}:{block.line}: ordinary example imports prohibited network or "
                    f"process module '{root}'"
                )
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "parallel" and isinstance(keyword.value, ast.Constant):
                if keyword.value.value is True:
                    raise DocumentationExampleError(
                        f"{display}:{block.line}: ordinary example cannot enable process fan-out"
                    )
            if keyword.arg == "n_jobs" and isinstance(keyword.value, ast.Constant):
                value = keyword.value.value
                if isinstance(value, int) and value > 1:
                    raise DocumentationExampleError(
                        f"{display}:{block.line}: ordinary example cannot request n_jobs={value}"
                    )


def execute_document_examples(
    paths: list[Path] | tuple[Path, ...], *, work_root: Path
) -> ExampleRunSummary:
    documents = tuple(paths)
    classification_issues = validate_example_classifications(documents)
    if classification_issues:
        raise DocumentationExampleError("\n".join(classification_issues))

    blocks_by_document: dict[Path, list[FencedBlock]] = {path: [] for path in documents}
    for block in collect_fenced_blocks(documents):
        blocks_by_document[block.source].append(block)

    old_cwd = Path.cwd()
    old_backend = os.environ.get("MPLBACKEND")
    os.environ["MPLBACKEND"] = "Agg"
    work_root.mkdir(parents=True, exist_ok=True)

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.use("Agg", force=True)
    executed_count = 0
    expensive_count = 0
    display_only_count = 0
    shell_setup_count = 0
    validated_count = 0
    expensive_pairs: list[str] = []
    executed_bounded_ids: set[str] = set()

    try:
        for document_index, source in enumerate(documents):
            namespace: dict[str, object] = {
                "__file__": str(source),
                "__name__": "__documentation_example__",
            }
            document_workdir = work_root / f"{document_index:02d}-{source.stem}"
            document_workdir.mkdir(parents=True, exist_ok=True)
            random_state = np.random.get_state()
            os.chdir(document_workdir)
            try:
                for block in blocks_by_document[source]:
                    if block.language in {"bash", "sh", "shell"}:
                        shell_setup_count += 1
                        display = _example_display_path(source)
                        syntax = subprocess.run(
                            ["bash", "-n"],
                            input=block.code,
                            text=True,
                            capture_output=True,
                            check=False,
                        )
                        if syntax.returncode != 0:
                            raise DocumentationExampleError(
                                f"{display}:{block.line}: shell-setup example failed syntax "
                                f"validation: {syntax.stderr.strip()}"
                            )
                        validated_count += 1
                        continue
                    if block.language != "python":
                        continue
                    if block.classification == "display-only":
                        display_only_count += 1
                        display = _example_display_path(source)
                        try:
                            compile(block.code, f"{display}:{block.line}", "exec")
                        except SyntaxError as exc:
                            stripped = "\n".join(
                                line
                                for line in block.code.splitlines()
                                if _DOCS_AUDIT_MARKER_RE.match(line) is None
                            ).strip()
                            if re.match(r"^[A-Za-z_]\w*\s*\(", stripped):
                                try:
                                    compile(
                                        f"def {stripped}:\n    pass\n",
                                        f"{display}:{block.line}",
                                        "exec",
                                    )
                                except SyntaxError:
                                    raise DocumentationExampleError(
                                        f"{display}:{block.line}: display-only Python example "
                                        f"failed syntax validation: {exc.msg}"
                                    ) from exc
                            else:
                                raise DocumentationExampleError(
                                    f"{display}:{block.line}: display-only Python example failed "
                                    f"syntax validation: {exc.msg}"
                                ) from exc
                        validated_count += 1
                        continue
                    if block.classification == "expensive":
                        expensive_count += 1
                        display = _example_display_path(source)
                        try:
                            compile(block.code, f"{display}:{block.line}", "exec")
                        except SyntaxError as exc:
                            raise DocumentationExampleError(
                                f"{display}:{block.line}: expensive Python example failed "
                                f"syntax validation: {exc.msg}"
                            ) from exc
                        expensive_pairs.append(block.metadata_dict["pair"])
                        continue

                    _assert_smoke_code_is_bounded(block)
                    display = _example_display_path(source)
                    try:
                        compiled = compile(block.code, f"{display}:{block.line}", "exec")
                        exec(compiled, namespace, namespace)
                    except Exception as exc:
                        raise DocumentationExampleError(
                            f"{display}:{block.line}: {block.classification} Python example "
                            f"failed: {exc}"
                        ) from exc
                    executed_count += 1
                    validated_count += 1
                    bounded_id = block.metadata_dict.get("id")
                    if bounded_id is not None:
                        executed_bounded_ids.add(bounded_id)
            finally:
                plt.close("all")
                np.random.set_state(random_state)
                os.chdir(old_cwd)
    finally:
        plt.close("all")
        os.chdir(old_cwd)
        if old_backend is None:
            os.environ.pop("MPLBACKEND", None)
        else:
            os.environ["MPLBACKEND"] = old_backend

    missing_pairs = sorted(set(expensive_pairs) - executed_bounded_ids)
    if missing_pairs:
        raise DocumentationExampleError(
            "expensive documentation examples lack executed bounded counterparts: "
            + ", ".join(missing_pairs)
        )
    validated_count += len(expensive_pairs)

    return ExampleRunSummary(
        documents=documents,
        executed_count=executed_count,
        expensive_count=expensive_count,
        display_only_count=display_only_count,
        shell_setup_count=shell_setup_count,
        validated_count=validated_count,
    )

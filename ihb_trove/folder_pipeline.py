"""Recursive JSONL corpus runner with mirrored survive/eliminated outputs."""

import hashlib
import json
import sqlite3
import tempfile
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

from .pipelines import (
    build_book_pipeline,
    build_exact_dedup_pipeline,
    build_minhash_pipeline,
    build_sentence_dedup_pipeline,
)

_UID = "_ihb_uid"
_SOURCE = "ihb_source_relpath"
_LINE = "ihb_source_line"


def _records(folder: Path) -> Iterator[dict[str, Any]]:
    if not folder.exists():
        return
    for path in sorted(folder.rglob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"Unexpected non-object record in {path}")
                    yield value


def _has_records(folder: Path) -> bool:
    return any(path.stat().st_size for path in folder.rglob("*.jsonl")) if folder.exists() else False


def _uid(relative_path: Path, line_number: int) -> str:
    value = f"{relative_path.as_posix()}\0{line_number}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


class _OutputRouter:
    """Bound the number of open per-source JSONL files during final routing."""

    def __init__(self, root: Path, relative_paths: list[Path], max_open: int = 32) -> None:
        self.root = root
        self.allowed = {path.as_posix() for path in relative_paths}
        self.max_open = max_open
        self.open_files: OrderedDict[Path, TextIO] = OrderedDict()
        self.survive = 0
        self.eliminated = 0
        for relative in relative_paths:
            for category in ("survive", "eliminated"):
                path = root / category / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

    def write(self, record: dict[str, Any], category: str, stage: str | None = None) -> None:
        metadata = record.setdefault("metadata", {})
        relative = metadata.get(_SOURCE)
        if relative not in self.allowed:
            raise ValueError(f"Unrecognized source path in routed record: {relative!r}")
        path = self.root / category / relative
        if stage:
            metadata["ihb_exclusion_stage"] = stage
            if stage.endswith("_dedup"):
                metadata["filter_reason"] = stage
            else:
                metadata.setdefault("filter_reason", stage)
        metadata.pop(_UID, None)
        handle = self.open_files.pop(path, None)
        if handle is None:
            if len(self.open_files) >= self.max_open:
                _, old = self.open_files.popitem(last=False)
                old.close()
            handle = path.open("a", encoding="utf-8")
        self.open_files[path] = handle
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if category == "survive":
            self.survive += 1
        else:
            self.eliminated += 1

    def close(self) -> None:
        for handle in self.open_files.values():
            handle.close()
        self.open_files.clear()


def _prepare_input(input_root: Path, staged: Path, router: _OutputRouter) -> tuple[int, int]:
    total = valid = 0
    for relative_name in sorted(router.allowed):
        relative = Path(relative_name)
        destination = staged / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with (input_root / relative).open("r", encoding="utf-8") as src, destination.open("w", encoding="utf-8") as dst:
            for line_number, line in enumerate(src, start=1):
                if not line.strip():
                    continue
                total += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    record = {"text": line.rstrip("\n"), "id": f"{relative_name}#{line_number}"}
                    reason = "invalid_json"
                else:
                    reason = None
                    if not isinstance(record, dict):
                        record = {"text": line.rstrip("\n"), "id": f"{relative_name}#{line_number}"}
                        reason = "invalid_record"
                    elif not isinstance(record.get("text"), str) or not record["text"].strip():
                        reason = "missing_text"
                metadata = record.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {"source_metadata": metadata}
                metadata = dict(metadata)
                metadata.update({_UID: _uid(relative, line_number), _SOURCE: relative_name, _LINE: line_number})
                record["metadata"] = metadata
                record.setdefault("id", f"{relative_name}#{line_number}")
                if reason:
                    metadata["filter_reason"] = reason
                    router.write(record, "eliminated", "input_validation")
                else:
                    dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                    valid += 1
    return total, valid


def _route_dropped(previous: Path, current: Path, stage: str, router: _OutputRouter, db: sqlite3.Connection) -> None:
    db.execute("DROP TABLE IF EXISTS surviving_uids")
    db.execute("CREATE TABLE surviving_uids (uid TEXT PRIMARY KEY)")
    with db:
        db.executemany(
            "INSERT INTO surviving_uids (uid) VALUES (?)",
            ((record["metadata"][_UID],) for record in _records(current)),
        )
    for record in _records(previous):
        uid = record["metadata"][_UID]
        if db.execute("SELECT 1 FROM surviving_uids WHERE uid = ?", (uid,)).fetchone() is None:
            router.write(record, "eliminated", stage)


def run_folder_pipeline(
    input_folder: str | Path,
    output_folder: str | Path,
    *,
    tasks: int = 1,
    workers: int = 1,
    language: str = "vi",
    language_backend: str = "langdetect",
    page_markers: tuple[str, ...] = ("---",),
    max_repair_fraction: float = 0.30,
    remove_image_placeholders: bool = True,
    repair_long_line_loops: bool = True,
) -> dict[str, Any]:
    """Process recursive JSONL files globally and mirror each source under both outputs.

    Quality failures and whole-document dedup removals go to ``eliminated``.
    SentenceDedup's edits to surviving documents are written only to ``survive``.
    A nonempty output directory is refused to prevent accidental mixed runs.
    """
    source, output = Path(input_folder).resolve(), Path(output_folder).resolve()
    if not source.is_dir() or tasks < 1 or workers < 1:
        raise ValueError("input_folder must be a directory; tasks/workers must be positive")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("input and output directories must not overlap")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    paths = sorted(path.relative_to(source) for path in source.rglob("*.jsonl") if path.is_file())
    if not paths:
        raise ValueError(f"No JSONL files under {source}")
    output.mkdir(parents=True, exist_ok=True)
    router = _OutputRouter(output, paths)
    summary: dict[str, Any] = {"input_files": len(paths), "stages": {}}
    try:
        with tempfile.TemporaryDirectory(prefix="ihb-trove-", dir=output.parent) as tmp:
            work = Path(tmp)
            source_staged, stages = work / "input", work / "stages"
            total, valid = _prepare_input(source, source_staged, router)
            summary.update(input_records=total, validated_records=valid)
            if valid:
                book = build_book_pipeline(
                    source_staged,
                    stages,
                    glob_pattern="**/*.jsonl",
                    tasks=tasks,
                    workers=workers,
                    language=language,
                    language_backend=language_backend,
                    page_markers=page_markers,
                    repair_max_removed_fraction=max_repair_fraction,
                    remove_image_placeholders=remove_image_placeholders,
                    repair_long_line_loops=repair_long_line_loops,
                )
                book.run()
                filtered = stages / "filtered"
                for gate in ("repair", "language", "repetition", "book_quality"):
                    for record in _records(stages / "quarantine" / gate):
                        router.write(record, "eliminated", gate)
                summary["stages"]["filtered"] = sum(1 for _ in _records(filtered))
                previous = filtered
                with sqlite3.connect(work / "routing.sqlite3") as db:
                    for name, builder in (
                        ("exact", build_exact_dedup_pipeline),
                        ("sentence", build_sentence_dedup_pipeline),
                        ("minhash", build_minhash_pipeline),
                    ):
                        if not _has_records(previous):
                            summary["stages"][name] = 0
                            continue
                        kwargs: dict[str, Any] = {"tasks": tasks, "workers": workers}
                        if name != "exact":
                            kwargs["language"] = language
                        builder(previous, stages, **kwargs).run()
                        current = stages / name
                        _route_dropped(previous, current, f"{name}_dedup", router, db)
                        summary["stages"][name] = sum(1 for _ in _records(current))
                        previous = current
                for record in _records(previous):
                    router.write(record, "survive")
    finally:
        router.close()
    summary.update(survive=router.survive, eliminated=router.eliminated)
    if summary["survive"] + summary["eliminated"] != summary["input_records"]:
        raise RuntimeError("Routing count mismatch; inspect output before using it")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary

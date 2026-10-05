"""Recursive JSONL corpus runner with mirrored survive/eliminated outputs."""

import hashlib
import json
import logging
import os
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

import orjson
from tqdm import tqdm
from datatrove.utils.logging import logger, setup_default_logger

from .pipelines import (
    build_book_pipeline,
    build_exact_dedup_pipeline,
    build_minhash_pipeline,
    build_sentence_dedup_pipeline,
)

_UID = "_ihb_uid"
_SOURCE = "ihb_source_relpath"
_LINE = "ihb_source_line"
_AUTO_WORKER_LIMIT = 32
_PREP_WORKER_LIMIT = 16


def _info(run_log: logging.Logger, message: str, *args: Any) -> None:
    """Write a progress event to both the terminal and the persistent run log."""
    rendered = message.format(*args)
    logger.info(message, *args)
    run_log.info(rendered)


def _exception(run_log: logging.Logger, message: str, *args: Any) -> None:
    rendered = message.format(*args)
    logger.exception(message, *args)
    run_log.exception(rendered)


@contextmanager
def _timed_stage(run_log: logging.Logger, name: str) -> Iterator[None]:
    started = time.perf_counter()
    _info(run_log, "START stage: {}", name)
    try:
        yield
    except Exception:
        _exception(run_log, "FAILED stage: {}", name)
        raise
    finally:
        _info(run_log, "END stage: {} elapsed_seconds={:.2f}", name, time.perf_counter() - started)


def _records(
    folder: Path,
    *,
    progress: str | None = None,
    run_log: logging.Logger | None = None,
) -> Iterator[dict[str, Any]]:
    if not folder.exists():
        return
    description = progress or f"Read {folder.name or folder}"
    records = 0
    with tqdm(desc=description, unit="doc", mininterval=1.0, disable=None) as bar:
        for path in sorted(folder.rglob("*.jsonl")):
            with path.open("rb") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        value = orjson.loads(line)
                    except orjson.JSONDecodeError:
                        value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"Unexpected non-object record in {path}")
                    bar.update(1)
                    records += 1
                    if records % 100_000 == 0:
                        if run_log is not None:
                            _info(run_log, "Progress {}: {} records", description, records)
                    yield value
    if run_log is not None:
        _info(run_log, "Complete {}: {} records", description, records)


def _available_cpu_count() -> int:
    """Respect CPU affinity and cgroup quotas when selecting automatic workers."""
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max" and int(period) > 0:
            available = min(available, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, available)


def _resolve_parallelism(file_count: int, tasks: int | None, workers: int | None) -> tuple[int, int]:
    cpu_count = _available_cpu_count()
    if workers is None:
        workers = min(_AUTO_WORKER_LIMIT, cpu_count, file_count)
    if tasks is None:
        # A few tasks per worker balances uneven file sizes without creating
        # thousands of mostly empty DataTrove shards on large hosts.
        tasks = min(file_count, workers * 4)
    if tasks < 1 or workers < 1:
        raise ValueError("tasks and workers must be positive")
    tasks = min(tasks, file_count)
    return tasks, min(workers, tasks)


def _prepare_one_file(
    input_root: Path,
    staged: Path,
    invalid_root: Path,
    relative_name: str,
) -> tuple[int, int]:
    relative = Path(relative_name)
    destination = staged / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    invalid_path = invalid_root / relative
    total = valid = 0
    invalid_handle: BinaryIO | None = None
    try:
        with (input_root / relative).open("rb") as src, destination.open("wb") as dst:
            for line_number, line in enumerate(src, start=1):
                if not line.strip():
                    continue
                total += 1
                try:
                    record = orjson.loads(line)
                except orjson.JSONDecodeError:
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        record = {
                            "text": line.decode("utf-8").rstrip("\r\n"),
                            "id": f"{relative_name}#{line_number}",
                        }
                        reason = "invalid_json"
                    else:
                        reason = None
                else:
                    reason = None
                if reason is None and not isinstance(record, dict):
                    record = {
                        "text": line.decode("utf-8").rstrip("\r\n"),
                        "id": f"{relative_name}#{line_number}",
                    }
                    reason = "invalid_record"
                elif reason is None and (not isinstance(record.get("text"), str) or not record["text"].strip()):
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
                    if invalid_handle is None:
                        invalid_path.parent.mkdir(parents=True, exist_ok=True)
                        invalid_handle = invalid_path.open("wb")
                    invalid_handle.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
                else:
                    dst.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
                    valid += 1
    finally:
        if invalid_handle is not None:
            invalid_handle.close()
    return total, valid


def _has_records(folder: Path) -> bool:
    return any(path.stat().st_size for path in folder.rglob("*.jsonl")) if folder.exists() else False


def _uid(relative_path: Path, line_number: int) -> str:
    value = f"{relative_path.as_posix()}\0{line_number}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


class _OutputRouter:
    """Bound the number of open per-source JSONL files during final routing."""

    def __init__(
        self,
        root: Path,
        relative_paths: list[Path],
        run_log: logging.Logger,
        max_open: int = 32,
    ) -> None:
        self.root = root
        self.run_log = run_log
        self.allowed = {path.as_posix() for path in relative_paths}
        self.max_open = max_open
        self.open_files: OrderedDict[Path, BinaryIO] = OrderedDict()
        self.survive = 0
        self.eliminated = 0
        with tqdm(
            total=len(relative_paths) * 2,
            desc="Prepare output JSONL files",
            unit="file",
            mininterval=1.0,
            disable=None,
        ) as bar:
            for relative in relative_paths:
                for category in ("survive", "eliminated"):
                    path = root / category / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                    bar.update(1)
        _info(
            run_log,
            "Prepared output paths: files_per_category={} total_placeholders={}",
            len(relative_paths),
            len(relative_paths) * 2,
        )

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
            handle = path.open("ab")
        self.open_files[path] = handle
        handle.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
        if category == "survive":
            self.survive += 1
        else:
            self.eliminated += 1

    def close(self) -> None:
        for handle in self.open_files.values():
            handle.close()
        self.open_files.clear()


def _prepare_input(
    input_root: Path,
    staged: Path,
    invalid_root: Path,
    router: _OutputRouter,
    preparation_workers: int,
    run_log: logging.Logger,
) -> tuple[int, int]:
    total = valid = 0
    names = sorted(router.allowed)
    pool_size = min(preparation_workers, len(names))
    with ThreadPoolExecutor(max_workers=pool_size, thread_name_prefix="ihb-jsonl") as pool:
        futures = [
            pool.submit(_prepare_one_file, input_root, staged, invalid_root, relative_name)
            for relative_name in names
        ]
        with tqdm(
            total=len(futures),
            desc="Validate & stage JSONL",
            unit="file",
            mininterval=1.0,
            disable=None,
        ) as bar:
            for future in as_completed(futures):
                file_total, file_valid = future.result()
                total += file_total
                valid += file_valid
                bar.update(1)
                bar.set_postfix(records=total, valid=valid, refresh=False)
                if bar.n % max(1, len(futures) // 100) == 0 or bar.n == len(futures):
                    _info(
                        run_log,
                        "Input staging progress: files={}/{} records={} valid={}",
                        bar.n,
                        len(futures),
                        total,
                        valid,
                    )
    for record in _records(invalid_root, progress="Route invalid input", run_log=run_log):
        router.write(record, "eliminated", "input_validation")
    return total, valid


def _route_dropped(
    previous: Path,
    current: Path,
    stage: str,
    router: _OutputRouter,
    db: sqlite3.Connection,
    run_log: logging.Logger,
) -> int:
    db.execute("DROP TABLE IF EXISTS surviving_uids")
    db.execute("CREATE TABLE surviving_uids (uid TEXT PRIMARY KEY)")
    with db:
        db.executemany(
            "INSERT INTO surviving_uids (uid) VALUES (?)",
            (
                (record["metadata"][_UID],)
                for record in _records(current, progress=f"Index survivors after {stage}", run_log=run_log)
            ),
        )
    dropped = 0
    for record in _records(previous, progress=f"Route removed by {stage}", run_log=run_log):
        uid = record["metadata"][_UID]
        if db.execute("SELECT 1 FROM surviving_uids WHERE uid = ?", (uid,)).fetchone() is None:
            router.write(record, "eliminated", stage)
            dropped += 1
    return dropped


def run_folder_pipeline(
    input_folder: str | Path,
    output_folder: str | Path,
    *,
    tasks: int | None = None,
    workers: int | None = None,
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
    A live run log and persistent DataTrove task logs are written under output/logs.
    """
    source, output = Path(input_folder).resolve(), Path(output_folder).resolve()
    if not source.is_dir():
        raise ValueError("input_folder must be a directory")
    if (tasks is not None and tasks < 1) or (workers is not None and workers < 1):
        raise ValueError("tasks/workers must be positive when specified")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("input and output directories must not overlap")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    log_root = output / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / "run.log"
    run_log = logging.Logger("ihb_trove.run", level=logging.INFO)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    run_log.addHandler(file_handler)
    started = time.perf_counter()
    _info(run_log, "IHB-Trove run started")
    _info(run_log, "Input folder: {}", source)
    _info(run_log, "Output folder: {}", output)
    _info(run_log, "Detailed live log: {}", log_path)
    _info(
        run_log,
        "Options: tasks={} workers={} language={} language_backend={} page_markers={} "
        "max_repair_fraction={} remove_image_placeholders={} repair_long_line_loops={}",
        tasks,
        workers,
        language,
        language_backend,
        page_markers,
        max_repair_fraction,
        remove_image_placeholders,
        repair_long_line_loops,
    )
    try:
        return _run_folder_pipeline_impl(
            source,
            output,
            log_path,
            run_log=run_log,
            tasks=tasks,
            workers=workers,
            language=language,
            language_backend=language_backend,
            page_markers=page_markers,
            max_repair_fraction=max_repair_fraction,
            remove_image_placeholders=remove_image_placeholders,
            repair_long_line_loops=repair_long_line_loops,
        )
    except Exception:
        _exception(run_log, "IHB-Trove run failed")
        raise
    finally:
        try:
            _info(run_log, "IHB-Trove run finished elapsed_seconds={:.2f}", time.perf_counter() - started)
            file_handler.flush()
        finally:
            run_log.removeHandler(file_handler)
            file_handler.close()
        setup_default_logger()


def _run_folder_pipeline_impl(
    source: Path,
    output: Path,
    log_path: Path,
    *,
    run_log: logging.Logger,
    tasks: int | None,
    workers: int | None,
    language: str,
    language_backend: str,
    page_markers: tuple[str, ...],
    max_repair_fraction: float,
    remove_image_placeholders: bool,
    repair_long_line_loops: bool,
) -> dict[str, Any]:
    _info(run_log, "Scanning recursively for JSONL files")
    discovery_started = time.perf_counter()
    paths: list[Path] = []
    with tqdm(desc="Discover JSONL", unit="file", mininterval=1.0, disable=None) as bar:
        for path in source.rglob("*.jsonl"):
            if path.is_file():
                paths.append(path.relative_to(source))
            bar.update(1)
            if bar.n and bar.n % 10_000 == 0:
                _info(run_log, "Discovery progress: {} JSONL paths scanned", bar.n)
    paths.sort()
    _info(
        run_log,
        "JSONL discovery complete: files={} elapsed_seconds={:.2f}",
        len(paths),
        time.perf_counter() - discovery_started,
    )
    if not paths:
        raise ValueError(f"No JSONL files under {source}")
    available_cpus = _available_cpu_count()
    tasks, workers = _resolve_parallelism(len(paths), tasks, workers)
    preparation_workers = min(_PREP_WORKER_LIMIT, workers, len(paths))
    _info(
        run_log,
        "Parallelism selected: files={} available_cpus={} tasks={} workers={} preparation_workers={}",
        len(paths),
        available_cpus,
        tasks,
        workers,
        preparation_workers,
    )
    router = _OutputRouter(output, paths, run_log)
    summary: dict[str, Any] = {
        "input_files": len(paths),
        "log_file": str(log_path),
        "parallelism": {
            "tasks": tasks,
            "workers": workers,
            "preparation_workers": preparation_workers,
        },
        "stages": {},
    }
    try:
        with tempfile.TemporaryDirectory(prefix="ihb-trove-", dir=output.parent) as tmp:
            work = Path(tmp)
            source_staged, stages = work / "input", work / "stages"
            with _timed_stage(run_log, "input validation and parallel staging"):
                total, valid = _prepare_input(
                    source,
                    source_staged,
                    work / "invalid_input",
                    router,
                    preparation_workers,
                    run_log,
                )
            summary.update(input_records=total, validated_records=valid)
            _info(run_log, "Input records read={} valid={} invalid={}", total, valid, total - valid)
            if valid:
                with _timed_stage(run_log, "repair and quality filters"):
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
                        logging_root=output / "logs" / "datatrove",
                    )
                    book.run()
                filtered = stages / "filtered"
                with _timed_stage(run_log, "route quality exclusions"):
                    for gate in ("repair", "language", "repetition", "book_quality"):
                        gate_count = 0
                        for record in _records(
                            stages / "quarantine" / gate,
                            progress=f"Route exclusions: {gate}",
                            run_log=run_log,
                        ):
                            router.write(record, "eliminated", gate)
                            gate_count += 1
                        _info(run_log, "Excluded at {}: {} records", gate, gate_count)
                with _timed_stage(run_log, "count filtered records"):
                    summary["stages"]["filtered"] = sum(
                        1 for _ in _records(filtered, progress="Count filtered records", run_log=run_log)
                    )
                _info(run_log, "Records after quality filters: {}", summary["stages"]["filtered"])
                previous = filtered
                with sqlite3.connect(work / "routing.sqlite3") as db:
                    for name, builder in (
                        ("exact", build_exact_dedup_pipeline),
                        ("sentence", build_sentence_dedup_pipeline),
                        ("minhash", build_minhash_pipeline),
                    ):
                        if not _has_records(previous):
                            summary["stages"][name] = 0
                            _info(run_log, "Skipping {} dedup: no records remain", name)
                            continue
                        kwargs: dict[str, Any] = {"tasks": tasks, "workers": workers}
                        if name != "exact":
                            kwargs["language"] = language
                        kwargs["logging_root"] = output / "logs" / "datatrove"
                        with _timed_stage(run_log, f"{name} dedup"):
                            builder(previous, stages, **kwargs).run()
                        current = stages / name
                        with _timed_stage(run_log, f"route {name} dedup exclusions"):
                            dropped = _route_dropped(previous, current, f"{name}_dedup", router, db, run_log)
                        with _timed_stage(run_log, f"count {name} survivors"):
                            summary["stages"][name] = sum(
                                1 for _ in _records(current, progress=f"Count {name} survivors", run_log=run_log)
                            )
                        _info(
                            run_log,
                            "{} dedup complete: dropped={} survivors={}",
                            name,
                            dropped,
                            summary["stages"][name],
                        )
                        previous = current
                with _timed_stage(run_log, "write surviving records"):
                    for record in _records(previous, progress="Write survivors", run_log=run_log):
                        router.write(record, "survive")
    finally:
        router.close()
    summary.update(survive=router.survive, eliminated=router.eliminated)
    if summary["survive"] + summary["eliminated"] != summary["input_records"]:
        raise RuntimeError("Routing count mismatch; inspect output before using it")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _info(
        run_log,
        "Run summary: input={} validated={} survive={} eliminated={} stages={}",
        summary["input_records"],
        summary["validated_records"],
        summary["survive"],
        summary["eliminated"],
        summary["stages"],
    )
    _info(run_log, "Summary JSON: {}", output / "summary.json")
    return summary

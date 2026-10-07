"""Recursive JSONL corpus runner with mirrored survive/eliminated outputs."""

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

import fasteners
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
_CACHE_SCHEMA_VERSION = 1
_CACHE_DIRECTORY = ".ihb_trove_cache"
_STAGES = ("input_prepared", "book_quality", "exact", "sentence", "minhash", "routed")


def _info(run_log: logging.Logger, message: str, *args: Any) -> None:
    """Write a progress event to both the terminal and the persistent run log."""
    rendered = message.format(*args)
    logger.info(message, *args)
    run_log.info(rendered)


def _exception(run_log: logging.Logger, message: str, *args: Any) -> None:
    rendered = message.format(*args)
    logger.exception(message, *args)
    run_log.exception(rendered)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_checkpoint(cache_root: Path) -> dict[str, Any]:
    path = cache_root / "checkpoint.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Resume checkpoint is missing or unreadable: {path}") from exc
    if state.get("schema_version") != _CACHE_SCHEMA_VERSION:
        raise ValueError("Resume checkpoint version is incompatible with this IHB-Trove build")
    completed = state.get("completed_stages")
    if not isinstance(completed, list) or completed != list(_STAGES[: len(completed)]):
        raise ValueError("Resume checkpoint has an invalid completed-stage sequence")
    return state


def _save_checkpoint(cache_root: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = time.time()
    _atomic_write_json(cache_root / "checkpoint.json", state)


def _mark_stage_complete(cache_root: Path, state: dict[str, Any], stage: str) -> None:
    completed = state["completed_stages"]
    expected_index = len(completed)
    if expected_index >= len(_STAGES) or _STAGES[expected_index] != stage:
        raise RuntimeError(f"Cannot checkpoint out-of-order stage: {stage}")
    completed.append(stage)
    _save_checkpoint(cache_root, state)


def _clear_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


@contextmanager
def _output_lock(output: Path) -> Iterator[None]:
    """Prevent two processes from writing or resuming the same output at once."""
    lock_path = output.parent / f".{output.name}.ihb-trove.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = fasteners.InterProcessLock(str(lock_path))
    if not lock.acquire(blocking=False):
        raise RuntimeError(f"Another IHB-Trove process is using this output: {output}")
    try:
        yield
    finally:
        lock.release()


def _combined_fingerprint(fingerprints: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative_name, file_digest in sorted(fingerprints.items()):
        digest.update(relative_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


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
) -> tuple[int, int, str]:
    relative = Path(relative_name)
    destination = staged / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    invalid_path = invalid_root / relative
    total = valid = 0
    digest = hashlib.sha256()
    invalid_handle: BinaryIO | None = None
    try:
        with (input_root / relative).open("rb") as src, destination.open("wb") as dst:
            for line_number, line in enumerate(src, start=1):
                digest.update(line)
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
    return total, valid, digest.hexdigest()


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
    relative_names: list[str],
    preparation_workers: int,
    run_log: logging.Logger,
) -> tuple[int, int, dict[str, str]]:
    total = valid = 0
    names = sorted(relative_names)
    fingerprints: dict[str, str] = {}
    processed_files = 0
    pool_size = min(preparation_workers, len(names))
    with ThreadPoolExecutor(max_workers=pool_size, thread_name_prefix="ihb-jsonl") as pool:
        futures = {
            pool.submit(_prepare_one_file, input_root, staged, invalid_root, relative_name): relative_name
            for relative_name in names
        }
        with tqdm(
            total=len(futures),
            desc="Validate & stage JSONL",
            unit="file",
            mininterval=1.0,
            disable=None,
        ) as bar:
            for future in as_completed(futures):
                file_total, file_valid, file_fingerprint = future.result()
                total += file_total
                valid += file_valid
                fingerprints[futures[future]] = file_fingerprint
                processed_files += 1
                bar.update(1)
                bar.set_postfix(records=total, valid=valid, refresh=False)
                if processed_files % max(1, len(futures) // 100) == 0 or processed_files == len(futures):
                    _info(
                        run_log,
                        "Input staging progress: files={}/{} records={} valid={}",
                        processed_files,
                        len(futures),
                        total,
                        valid,
                    )
    return total, valid, fingerprints


def _hash_one_file(input_root: Path, relative_name: str) -> tuple[str, str]:
    digest = hashlib.sha256()
    with (input_root / relative_name).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return relative_name, digest.hexdigest()


def _verify_input_fingerprint(
    input_root: Path,
    relative_names: list[str],
    expected: str,
    workers: int,
    run_log: logging.Logger,
) -> None:
    _info(run_log, "Verifying input fingerprint before reusing cached stages")
    fingerprints: dict[str, str] = {}
    processed_files = 0
    pool_size = min(max(1, workers), len(relative_names))
    with ThreadPoolExecutor(max_workers=pool_size, thread_name_prefix="ihb-hash") as pool:
        futures = [pool.submit(_hash_one_file, input_root, name) for name in relative_names]
        with tqdm(
            total=len(futures),
            desc="Verify input files",
            unit="file",
            mininterval=1.0,
            disable=None,
        ) as bar:
            for future in as_completed(futures):
                name, file_digest = future.result()
                fingerprints[name] = file_digest
                processed_files += 1
                bar.update(1)
                if processed_files % max(1, len(futures) // 100) == 0 or processed_files == len(futures):
                    _info(run_log, "Input verification progress: files={}/{}", processed_files, len(futures))
    actual = _combined_fingerprint(fingerprints)
    if actual != expected:
        raise ValueError(
            "Input JSONL content changed since the checkpoint was created; "
            "cached stages cannot be resumed safely"
        )
    _info(run_log, "Input fingerprint matches the checkpoint")


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
    resume: bool = False,
) -> dict[str, Any]:
    """Process recursive JSONL files globally and mirror each source under both outputs.

    Quality failures and whole-document dedup removals go to ``eliminated``.
    SentenceDedup's edits to surviving documents are written only to ``survive``.
    Completed stages are checkpointed under ``output/.ihb_trove_cache``. Pass
    ``resume=True`` after interruption to reuse completed stages safely.
    """
    source, output = Path(input_folder).resolve(), Path(output_folder).resolve()
    if not source.is_dir():
        raise ValueError("input_folder must be a directory")
    if (tasks is not None and tasks < 1) or (workers is not None and workers < 1):
        raise ValueError("tasks/workers must be positive when specified")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("input and output directories must not overlap")
    output.parent.mkdir(parents=True, exist_ok=True)
    with _output_lock(output):
        cache_root = output / _CACHE_DIRECTORY
        checkpoint_path = cache_root / "checkpoint.json"
        summary_path = output / "summary.json"
        if output.exists() and any(output.iterdir()):
            if resume and checkpoint_path.is_file():
                state = _load_checkpoint(cache_root)
                state["resume_count"] = int(state.get("resume_count", 0)) + 1
                _save_checkpoint(cache_root, state)
            elif resume and summary_path.is_file():
                return json.loads(summary_path.read_text(encoding="utf-8"))
            elif resume:
                raise ValueError(
                    f"No IHB-Trove checkpoint found under {output}. This output came from a run "
                    "without resume support or has lost its checkpoint; it cannot be resumed safely."
                )
            else:
                raise FileExistsError(
                    f"Output directory is not empty: {output}. Use --resume to continue a checkpointed run."
                )
        else:
            output.mkdir(parents=True, exist_ok=True)
            cache_root.mkdir(parents=True, exist_ok=True)
            state = {
                "schema_version": _CACHE_SCHEMA_VERSION,
                "completed_stages": [],
                "resume_count": 0,
                "stats": {},
            }
            _save_checkpoint(cache_root, state)

        output.mkdir(parents=True, exist_ok=True)
        log_root = output / "logs"
        log_root.mkdir(parents=True, exist_ok=True)
        log_path = log_root / "run.log"
        run_log = logging.Logger(f"ihb_trove.run.{time.time_ns()}", level=logging.INFO)
        file_handler = logging.FileHandler(log_path, mode="a" if resume else "w", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        run_log.addHandler(file_handler)
        started = time.perf_counter()
        _info(run_log, "IHB-Trove run started resume={}", resume)
        _info(run_log, "Input folder: {}", source)
        _info(run_log, "Output folder: {}", output)
        _info(run_log, "Detailed live log: {}", log_path)
        _info(
            run_log,
            "Requested options: tasks={} workers={} language={} language_backend={} page_markers={} "
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
                cache_root,
                state,
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
            _exception(run_log, "IHB-Trove run failed; checkpoint retained for --resume")
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
    cache_root: Path,
    state: dict[str, Any],
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
    saved_config = state.get("config")
    if saved_config:
        saved_tasks = int(saved_config["tasks"])
        saved_workers = int(saved_config["workers"])
        if tasks is not None and min(tasks, len(paths)) != saved_tasks:
            raise ValueError("--tasks differs from the checkpoint; resume with the original task count")
        if workers is not None and min(workers, saved_tasks) != saved_workers:
            raise ValueError("--workers differs from the checkpoint; resume with the original worker count")
        tasks, workers = saved_tasks, saved_workers
    else:
        tasks, workers = _resolve_parallelism(len(paths), tasks, workers)
    preparation_workers = min(_PREP_WORKER_LIMIT, workers, len(paths))
    config = {
        "tasks": tasks,
        "workers": workers,
        "language": language,
        "language_backend": language_backend,
        "page_markers": list(page_markers),
        "max_repair_fraction": max_repair_fraction,
        "remove_image_placeholders": remove_image_placeholders,
        "repair_long_line_loops": repair_long_line_loops,
    }
    saved_source = state.get("source_root")
    if saved_source is not None and saved_source != str(source):
        raise ValueError("Input folder differs from the checkpoint")
    if saved_config is not None and saved_config != config:
        raise ValueError("Pipeline options differ from the checkpoint; resume with the original options")
    state["source_root"] = str(source)
    state["input_file_count"] = len(paths)
    state["config"] = config
    _save_checkpoint(cache_root, state)
    _info(
        run_log,
        "Parallelism selected: files={} available_cpus={} tasks={} workers={} preparation_workers={}",
        len(paths),
        available_cpus,
        tasks,
        workers,
        preparation_workers,
    )
    summary: dict[str, Any] = {
        "input_files": len(paths),
        "log_file": str(log_path),
        "parallelism": {
            "tasks": tasks,
            "workers": workers,
            "preparation_workers": preparation_workers,
        },
        "stages": {},
        "resumed_from_checkpoint": int(state.get("resume_count", 0)) > 0,
    }
    stats = state.setdefault("stats", {})
    relative_names = [path.as_posix() for path in paths]
    input_staged, invalid_root, stages = cache_root / "input", cache_root / "invalid_input", cache_root / "stages"

    if "input_prepared" in state["completed_stages"]:
        total, valid = int(stats["input_records"]), int(stats["validated_records"])
        _verify_input_fingerprint(
            source,
            relative_names,
            str(state["input_fingerprint"]),
            preparation_workers,
            run_log,
        )
    else:
        _clear_path(input_staged)
        _clear_path(invalid_root)
        with _timed_stage(run_log, "input validation and parallel staging"):
            total, valid, fingerprints = _prepare_input(
                source,
                input_staged,
                invalid_root,
                relative_names,
                preparation_workers,
                run_log,
            )
        stats.update(input_records=total, validated_records=valid)
        state["input_fingerprint"] = _combined_fingerprint(fingerprints)
        _mark_stage_complete(cache_root, state, "input_prepared")
    summary.update(input_records=total, validated_records=valid)
    _info(run_log, "Input records read={} valid={} invalid={}", total, valid, total - valid)

    filtered = stages / "filtered"
    if "book_quality" not in state["completed_stages"]:
        _clear_path(stages)
        if valid:
            with _timed_stage(run_log, "repair and quality filters"):
                build_book_pipeline(
                    input_staged,
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
                ).run()
        else:
            _info(run_log, "Skipping repair and quality filters: no valid input records")
        stats["filtered"] = sum(1 for _ in _records(filtered, progress="Count filtered records", run_log=run_log))
        _mark_stage_complete(cache_root, state, "book_quality")
    else:
        summary["stages"]["filtered"] = int(stats["filtered"])
    if "filtered" not in summary["stages"]:
        summary["stages"]["filtered"] = int(stats["filtered"])
    _info(run_log, "Records after quality filters: {}", summary["stages"]["filtered"])

    previous = filtered
    for name, builder in (
        ("exact", build_exact_dedup_pipeline),
        ("sentence", build_sentence_dedup_pipeline),
        ("minhash", build_minhash_pipeline),
    ):
        current = stages / name
        if name in state["completed_stages"]:
            stage_count = int(stats["stages"][name])
        else:
            _clear_path(current)
            _clear_path(stages / "work" / name)
            if _has_records(previous):
                kwargs: dict[str, Any] = {"tasks": tasks, "workers": workers}
                if name != "exact":
                    kwargs["language"] = language
                kwargs["logging_root"] = output / "logs" / "datatrove"
                with _timed_stage(run_log, f"{name} dedup"):
                    builder(previous, stages, **kwargs).run()
            else:
                _info(run_log, "Skipping {} dedup: no records remain", name)
            with _timed_stage(run_log, f"count {name} survivors"):
                stage_count = sum(
                    1 for _ in _records(current, progress=f"Count {name} survivors", run_log=run_log)
                )
            stats.setdefault("stages", {})[name] = stage_count
            _mark_stage_complete(cache_root, state, name)
        summary["stages"][name] = stage_count
        _info(run_log, "{} stage checkpoint available: survivors={}", name, stage_count)
        previous = current

    if "routed" in state["completed_stages"]:
        summary = stats["summary"]
        if not (output / "summary.json").is_file():
            _atomic_write_json(output / "summary.json", summary)
        _info(run_log, "Routing checkpoint already complete; returning saved summary")
        shutil.rmtree(cache_root, ignore_errors=True)
        return summary

    _clear_path(output / "survive")
    _clear_path(output / "eliminated")
    router = _OutputRouter(output, paths, run_log)
    try:
        with _timed_stage(run_log, "route all exclusions and survivors"):
            for record in _records(invalid_root, progress="Route invalid input", run_log=run_log):
                router.write(record, "eliminated", "input_validation")
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
            with sqlite3.connect(cache_root / "routing.sqlite3") as db:
                previous_stage = filtered
                for name in ("exact", "sentence", "minhash"):
                    current_stage = stages / name
                    dropped = _route_dropped(previous_stage, current_stage, f"{name}_dedup", router, db, run_log)
                    _info(run_log, "{} dedup exclusions routed: {} records", name, dropped)
                    previous_stage = current_stage
            for record in _records(previous, progress="Write survivors", run_log=run_log):
                router.write(record, "survive")
    finally:
        router.close()

    summary.update(survive=router.survive, eliminated=router.eliminated)
    summary["resume_count"] = int(state.get("resume_count", 0))
    if summary["survive"] + summary["eliminated"] != summary["input_records"]:
        raise RuntimeError("Routing count mismatch; inspect output before using it")
    stats["summary"] = summary
    _atomic_write_json(output / "summary.json", summary)
    _mark_stage_complete(cache_root, state, "routed")
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
    try:
        shutil.rmtree(cache_root)
        _info(run_log, "Removed completed-stage cache after successful publish")
    except OSError as exc:
        _info(run_log, "Could not remove completed-stage cache: {}", exc)
    return summary

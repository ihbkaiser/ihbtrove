"""Composable LocalPipelineExecutor jobs built from DataTrove blocks."""

from pathlib import Path
import sys

from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import (
    ExactDedupConfig,
    ExactDedupFilter,
    ExactDedupSignature,
    ExactFindDedups,
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
    MinhashDedupSignature,
    SentDedupConfig,
    SentenceDedupFilter,
    SentenceDedupSignature,
    SentenceFindDedups,
)
from datatrove.pipeline.filters import GopherRepetitionFilter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig

from .filters import BookQualityFilter
from .repair import (
    ImagePlaceholderRepair,
    LocalRepeatedSpanRepair,
    LongLineLoopRepair,
    PageStructureRepair,
    RepairMetrics,
    UnicodeNormalizer,
)
from .tokenization import UnicodeTokenizer

BOOK_GOPHER_FILTER_POLICY_VERSION = "0.3.0"
BOOK_QUALITY_POLICY_VERSION = "0.2.0"
DEDUP_TOKENIZER_POLICY_VERSION = "0.1.0"
_DEFAULT_DEDUP_FINDER_WORKERS = 16


def resolve_dedup_finder_workers(
    tasks: int,
    workers: int,
    requested: int | None = None,
) -> int:
    """Choose hash-range finder shards without exceeding available executor parallelism."""
    if tasks < 1 or workers < 1:
        raise ValueError("tasks and workers must be positive")
    target = _DEFAULT_DEDUP_FINDER_WORKERS if requested is None else requested
    if target < 1:
        raise ValueError("dedup_finder_workers must be positive")
    return min(target, tasks, workers)


def _writer(path: Path) -> JsonlWriter:
    return JsonlWriter(str(path), output_filename="${rank}.jsonl", compression=None)


def _reader(path: Path, glob_pattern: str = "*.jsonl") -> JsonlReader:
    return JsonlReader(str(path), glob_pattern=glob_pattern, file_progress=sys.stderr.isatty())


def _exact_text(doc: Document) -> str:
    return doc.text


def build_book_pipeline(
    source: str | Path,
    output_root: str | Path,
    *,
    glob_pattern: str = "*.jsonl",
    tasks: int = 1,
    workers: int = 1,
    page_markers: tuple[str, ...] = ("---",),
    remove_image_placeholders: bool = True,
    repair_long_line_loops: bool = True,
    dehyphenate_line_breaks: bool = False,
    max_consecutive_blank_lines: int | None = 2,
    logging_root: str | Path | None = None,
) -> LocalPipelineExecutor:
    """Run structural repair before permissive, language-neutral quality gates."""
    root = Path(output_root)
    logs = Path(logging_root) if logging_root is not None else root / "logs"
    return LocalPipelineExecutor(
        pipeline=[
            _reader(Path(source), glob_pattern),
            UnicodeNormalizer(
                dehyphenate_line_breaks=dehyphenate_line_breaks,
                max_consecutive_blank_lines=max_consecutive_blank_lines,
            ),
            *([ImagePlaceholderRepair()] if remove_image_placeholders else []),
            PageStructureRepair(page_markers=page_markers),
            LocalRepeatedSpanRepair(),
            *([LongLineLoopRepair()] if repair_long_line_loops else []),
            RepairMetrics(),
            GopherRepetitionFilter(
                # Use only language-neutral character checks. Repeated short
                # headings are not counted, and no locale tokenizer is assumed.
                dup_line_frac=None,
                dup_para_frac=None,
                dup_line_char_frac=0.50,
                dup_para_char_frac=0.50,
                top_n_grams=(),
                dup_n_grams=(),
                language=None,
                exclusion_writer=_writer(root / "quarantine" / "repetition"),
            ),
            BookQualityFilter(exclusion_writer=_writer(root / "quarantine" / "book_quality")),
            _writer(root / "filtered"),
        ],
        tasks=tasks,
        workers=workers,
        logging_dir=str(logs / "book"),
    )


def build_exact_dedup_pipeline(
    source: str | Path,
    output_root: str | Path,
    *,
    tasks: int = 1,
    workers: int = 1,
    finder_workers: int | None = None,
    depends: LocalPipelineExecutor | None = None,
    logging_root: str | Path | None = None,
) -> LocalPipelineExecutor:
    """Signature → global exact matching → filter, with stable reader sharding."""
    root = Path(output_root)
    logs = Path(logging_root) if logging_root is not None else root / "logs"
    finder_workers = resolve_dedup_finder_workers(tasks, workers, finder_workers)
    config = ExactDedupConfig(content_getter=_exact_text, hash_config=HashConfig(hash_fc="sha1"))
    signatures, duplicates = root / "work" / "exact" / "signatures", root / "work" / "exact" / "duplicates"
    signature_job = LocalPipelineExecutor(
        pipeline=[
            _reader(Path(source)),
            ExactDedupSignature(str(signatures), config, finder_workers=finder_workers),
        ],
        tasks=tasks,
        workers=workers,
        depends=depends,
        logging_dir=str(logs / "exact_signature"),
    )
    find_job = LocalPipelineExecutor(
        pipeline=[ExactFindDedups(str(signatures), str(duplicates), config)],
        tasks=finder_workers,
        workers=min(workers, finder_workers),
        depends=signature_job,
        logging_dir=str(logs / "exact_find"),
    )
    return LocalPipelineExecutor(
        pipeline=[_reader(Path(source)), ExactDedupFilter(str(duplicates), config), _writer(root / "exact")],
        tasks=tasks,
        workers=workers,
        depends=find_job,
        logging_dir=str(logs / "exact_filter"),
    )


def build_sentence_dedup_pipeline(
    source: str | Path,
    output_root: str | Path,
    *,
    tasks: int = 1,
    workers: int = 1,
    finder_workers: int | None = None,
    tokenizer_language: str | None = None,
    depends: LocalPipelineExecutor | None = None,
    logging_root: str | Path | None = None,
) -> LocalPipelineExecutor:
    """Use DataTrove SentenceDedup with a generic tokenizer by default."""
    root = Path(output_root)
    logs = Path(logging_root) if logging_root is not None else root / "logs"
    finder_workers = resolve_dedup_finder_workers(tasks, workers, finder_workers)
    config = SentDedupConfig(n_sentences=3, min_words_to_remove_span=25, hash_config=HashConfig(hash_fc="sha1"))
    signatures, duplicates = root / "work" / "sentence" / "signatures", root / "work" / "sentence" / "duplicates"
    tokenizer = tokenizer_language if tokenizer_language is not None else UnicodeTokenizer()
    signature_job = LocalPipelineExecutor(
        pipeline=[
            _reader(Path(source)),
            SentenceDedupSignature(
                str(signatures), config=config, language=tokenizer, finder_workers=finder_workers
            ),
        ],
        tasks=tasks,
        workers=workers,
        depends=depends,
        logging_dir=str(logs / "sentence_signature"),
    )
    find_job = LocalPipelineExecutor(
        pipeline=[SentenceFindDedups(str(signatures), str(duplicates), config=config)],
        tasks=finder_workers,
        workers=min(workers, finder_workers),
        depends=signature_job,
        logging_dir=str(logs / "sentence_find"),
    )
    return LocalPipelineExecutor(
        pipeline=[
            _reader(Path(source)),
            SentenceDedupFilter(str(duplicates), config=config, language=tokenizer),
            _writer(root / "sentence"),
        ],
        tasks=tasks,
        workers=workers,
        depends=find_job,
        logging_dir=str(logs / "sentence_filter"),
    )


def build_minhash_pipeline(
    source: str | Path,
    output_root: str | Path,
    *,
    tasks: int = 1,
    workers: int = 1,
    tokenizer_language: str | None = None,
    config: MinhashConfig | None = None,
    depends: LocalPipelineExecutor | None = None,
    logging_root: str | Path | None = None,
) -> LocalPipelineExecutor:
    """DataTrove MinHash signature → bucket match → cluster → filter."""
    root = Path(output_root)
    logs = Path(logging_root) if logging_root is not None else root / "logs"
    config = config or MinhashConfig(hash_config=HashConfig(hash_fc="sha1"))
    work = root / "work" / "minhash"
    signatures, pairs, removals = work / "signatures", work / "pairs", work / "removals"
    tokenizer = tokenizer_language if tokenizer_language is not None else UnicodeTokenizer()
    signature_job = LocalPipelineExecutor(
        pipeline=[_reader(Path(source)), MinhashDedupSignature(str(signatures), config=config, language=tokenizer)],
        tasks=tasks,
        workers=workers,
        depends=depends,
        logging_dir=str(logs / "minhash_signature"),
    )
    buckets_job = LocalPipelineExecutor(
        pipeline=[MinhashDedupBuckets(str(signatures), str(pairs), config=config)],
        tasks=config.num_buckets,
        workers=min(workers, config.num_buckets),
        depends=signature_job,
        logging_dir=str(logs / "minhash_buckets"),
    )
    cluster_job = LocalPipelineExecutor(
        pipeline=[MinhashDedupCluster(str(pairs), str(removals), config=config)],
        tasks=1,
        workers=1,
        depends=buckets_job,
        logging_dir=str(logs / "minhash_cluster"),
    )
    return LocalPipelineExecutor(
        pipeline=[_reader(Path(source)), MinhashDedupFilter(str(removals)), _writer(root / "minhash")],
        tasks=tasks,
        workers=workers,
        depends=cluster_job,
        logging_dir=str(logs / "minhash_filter"),
    )

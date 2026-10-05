"""Composable LocalPipelineExecutor jobs around unmodified DataTrove blocks."""

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
from datatrove.pipeline.filters import GopherRepetitionFilter, LanguageFilter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig

from .filters import BookQualityFilter, RepairQualityFilter
from .offline_language import OfflineLanguageFilter
from .repair import (
    ImagePlaceholderRepair,
    LocalRepeatedSpanRepair,
    LongLineLoopRepair,
    PageStructureRepair,
    RepairMetrics,
    UnicodeNormalizer,
)


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
    language: str = "vi",
    language_backend: str = "langdetect",
    page_markers: tuple[str, ...] = ("---",),
    remove_image_placeholders: bool = True,
    repair_long_line_loops: bool = True,
    repair_max_removed_fraction: float = 0.30,
    gopher_dup_n_grams: tuple[tuple[int, float], ...] = (
        (5, 0.20), (6, 0.19), (7, 0.18), (8, 0.17), (9, 0.16), (10, 0.15),
    ),
) -> LocalPipelineExecutor:
    """Run structural repair before language and traditional text quality gates."""
    root = Path(output_root)
    return LocalPipelineExecutor(
        pipeline=[
            _reader(Path(source), glob_pattern),
            UnicodeNormalizer(),
            *([ImagePlaceholderRepair()] if remove_image_placeholders else []),
            PageStructureRepair(page_markers=page_markers),
            LocalRepeatedSpanRepair(),
            *([LongLineLoopRepair()] if repair_long_line_loops else []),
            RepairMetrics(),
            RepairQualityFilter(
                max_removed_fraction=repair_max_removed_fraction,
                exclusion_writer=_writer(root / "quarantine" / "repair"),
            ),
            (
                OfflineLanguageFilter(
                    languages=(language,),
                    language_threshold=0.65,
                    exclusion_writer=_writer(root / "quarantine" / "language"),
                )
                if language_backend == "langdetect"
                else LanguageFilter(
                    languages=[language],
                    language_threshold=0.65,
                    backend=language_backend,
                    exclusion_writer=_writer(root / "quarantine" / "language"),
                )
            ),
            GopherRepetitionFilter(
                language=language,
                dup_n_grams=gopher_dup_n_grams,
                exclusion_writer=_writer(root / "quarantine" / "repetition"),
            ),
            BookQualityFilter(exclusion_writer=_writer(root / "quarantine" / "book_quality")),
            _writer(root / "filtered"),
        ],
        tasks=tasks,
        workers=workers,
        logging_dir=str(root / "logs" / "book"),
    )


def build_exact_dedup_pipeline(
    source: str | Path,
    output_root: str | Path,
    *,
    tasks: int = 1,
    workers: int = 1,
    depends: LocalPipelineExecutor | None = None,
) -> LocalPipelineExecutor:
    """Signature → global exact matching → filter, with stable reader sharding."""
    root = Path(output_root)
    config = ExactDedupConfig(content_getter=_exact_text, hash_config=HashConfig(hash_fc="sha1"))
    signatures, duplicates = root / "work" / "exact" / "signatures", root / "work" / "exact" / "duplicates"
    signature_job = LocalPipelineExecutor(
        pipeline=[_reader(Path(source)), ExactDedupSignature(str(signatures), config)],
        tasks=tasks,
        workers=workers,
        depends=depends,
        logging_dir=str(root / "logs" / "exact_signature"),
    )
    find_job = LocalPipelineExecutor(
        pipeline=[ExactFindDedups(str(signatures), str(duplicates), config)],
        tasks=1,
        workers=1,
        depends=signature_job,
        logging_dir=str(root / "logs" / "exact_find"),
    )
    return LocalPipelineExecutor(
        pipeline=[_reader(Path(source)), ExactDedupFilter(str(duplicates), config), _writer(root / "exact")],
        tasks=tasks,
        workers=workers,
        depends=find_job,
        logging_dir=str(root / "logs" / "exact_filter"),
    )


def build_sentence_dedup_pipeline(
    source: str | Path,
    output_root: str | Path,
    *,
    tasks: int = 1,
    workers: int = 1,
    language: str = "vi",
    depends: LocalPipelineExecutor | None = None,
) -> LocalPipelineExecutor:
    """Use DataTrove sentence dedup, preserving short repeated book sections."""
    root = Path(output_root)
    config = SentDedupConfig(n_sentences=3, min_words_to_remove_span=25, hash_config=HashConfig(hash_fc="sha1"))
    signatures, duplicates = root / "work" / "sentence" / "signatures", root / "work" / "sentence" / "duplicates"
    signature_job = LocalPipelineExecutor(
        pipeline=[_reader(Path(source)), SentenceDedupSignature(str(signatures), config=config, language=language)],
        tasks=tasks,
        workers=workers,
        depends=depends,
        logging_dir=str(root / "logs" / "sentence_signature"),
    )
    find_job = LocalPipelineExecutor(
        pipeline=[SentenceFindDedups(str(signatures), str(duplicates), config=config)],
        tasks=1,
        workers=1,
        depends=signature_job,
        logging_dir=str(root / "logs" / "sentence_find"),
    )
    return LocalPipelineExecutor(
        pipeline=[
            _reader(Path(source)),
            SentenceDedupFilter(str(duplicates), config=config, language=language),
            _writer(root / "sentence"),
        ],
        tasks=tasks,
        workers=workers,
        depends=find_job,
        logging_dir=str(root / "logs" / "sentence_filter"),
    )


def build_minhash_pipeline(
    source: str | Path,
    output_root: str | Path,
    *,
    tasks: int = 1,
    workers: int = 1,
    language: str = "vi",
    config: MinhashConfig | None = None,
    depends: LocalPipelineExecutor | None = None,
) -> LocalPipelineExecutor:
    """DataTrove MinHash signature → bucket match → cluster → filter."""
    root = Path(output_root)
    config = config or MinhashConfig(hash_config=HashConfig(hash_fc="sha1"))
    work = root / "work" / "minhash"
    signatures, pairs, removals = work / "signatures", work / "pairs", work / "removals"
    signature_job = LocalPipelineExecutor(
        pipeline=[_reader(Path(source)), MinhashDedupSignature(str(signatures), config=config, language=language)],
        tasks=tasks,
        workers=workers,
        depends=depends,
        logging_dir=str(root / "logs" / "minhash_signature"),
    )
    buckets_job = LocalPipelineExecutor(
        pipeline=[MinhashDedupBuckets(str(signatures), str(pairs), config=config)],
        tasks=config.num_buckets,
        workers=min(workers, config.num_buckets),
        depends=signature_job,
        logging_dir=str(root / "logs" / "minhash_buckets"),
    )
    cluster_job = LocalPipelineExecutor(
        pipeline=[MinhashDedupCluster(str(pairs), str(removals), config=config)],
        tasks=1,
        workers=1,
        depends=buckets_job,
        logging_dir=str(root / "logs" / "minhash_cluster"),
    )
    return LocalPipelineExecutor(
        pipeline=[_reader(Path(source)), MinhashDedupFilter(str(removals)), _writer(root / "minhash")],
        tasks=tasks,
        workers=workers,
        depends=cluster_job,
        logging_dir=str(root / "logs" / "minhash_filter"),
    )

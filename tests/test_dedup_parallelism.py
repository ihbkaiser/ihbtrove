"""The signature hash partitions and finder executor must use matching shards."""

from pathlib import Path

import pytest

import ihb_trove.pipelines as pipelines
from datatrove.pipeline.dedup import (
    ExactDedupSignature,
    ExactFindDedups,
    SentenceDedupSignature,
    SentenceFindDedups,
)


class CapturingExecutor:
    instances: list["CapturingExecutor"] = []

    def __init__(self, pipeline, *, tasks, workers, depends=None, logging_dir=None):
        self.pipeline = pipeline
        self.tasks = tasks
        self.workers = workers
        self.depends = depends
        self.logging_dir = logging_dir
        self.instances.append(self)


@pytest.fixture(autouse=True)
def capture_executors(monkeypatch: pytest.MonkeyPatch):
    CapturingExecutor.instances = []
    monkeypatch.setattr(pipelines, "LocalPipelineExecutor", CapturingExecutor)


@pytest.mark.parametrize(
    ("builder", "signature_type", "finder_type"),
    [
        (pipelines.build_exact_dedup_pipeline, ExactDedupSignature, ExactFindDedups),
        (pipelines.build_sentence_dedup_pipeline, SentenceDedupSignature, SentenceFindDedups),
    ],
)
def test_dedup_finder_parallelizes_matching_hash_partitions(
    tmp_path: Path,
    builder,
    signature_type,
    finder_type,
) -> None:
    builder(tmp_path / "input", tmp_path / "output", tasks=64, workers=32, finder_workers=8)

    signature_job = next(
        job for job in CapturingExecutor.instances if isinstance(job.pipeline[-1], signature_type)
    )
    finder_job = next(job for job in CapturingExecutor.instances if isinstance(job.pipeline[0], finder_type))
    assert signature_job.pipeline[-1].finder_workers == 8
    assert finder_job.tasks == 8
    assert finder_job.workers == 8


def test_finder_worker_count_defaults_and_caps_to_main_executor() -> None:
    assert pipelines.resolve_dedup_finder_workers(tasks=448, workers=112) == 16
    assert pipelines.resolve_dedup_finder_workers(tasks=4, workers=2) == 2
    assert pipelines.resolve_dedup_finder_workers(tasks=4, workers=2, requested=8) == 2

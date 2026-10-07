"""Regression tests for sentence IDs larger than uint16."""

from pathlib import Path

from datatrove.io import get_datafolder
from datatrove.pipeline.dedup import (
    SENTENCE_SIGNATURE_FORMAT_VERSION,
    SentDedupConfig,
    SentenceDedupFilter,
    SentenceDedupSignature,
    SentenceFindDedups,
)
from datatrove.pipeline.dedup.sentence_dedup import read_sigs
from datatrove.utils.hashing import HashConfig


def test_sentence_dedup_round_trips_sentence_id_65536(tmp_path: Path) -> None:
    config = SentDedupConfig(hash_config=HashConfig(hash_fc="sha1"))
    signatures_path = tmp_path / "signatures"
    duplicates_path = tmp_path / "duplicates"

    # Avoid loading a language tokenizer: this test exercises the signature
    # binary format directly, including the failing uint16 boundary value.
    signature_step = object.__new__(SentenceDedupSignature)
    signature_step.config = config
    signature_step.finder_workers = 1
    signature_step.output_folder = get_datafolder(str(signatures_path))
    signature_step.save_hashes(0, [(42, 0, 65_536), (42, 1, 65_536)])

    signature_files = signature_step.output_folder.list_files(glob_pattern="*.c4_sig")
    parsed = list(
        read_sigs(
            signature_step.output_folder.open_files(signature_files)[0],
            file_id=0,
            config=config,
        )
    )
    assert [signature.sent_id for signature in parsed] == [65_536, 65_536]

    SentenceFindDedups(str(signatures_path), str(duplicates_path), config=config).run(rank=0, world_size=1)
    duplicate_files = get_datafolder(str(duplicates_path)).list_files(glob_pattern="*.c4_dup")
    duplicate_folder = get_datafolder(str(duplicates_path))
    filter_step = object.__new__(SentenceDedupFilter)
    filter_step.data_folder = duplicate_folder
    with duplicate_folder.open_files(duplicate_files)[0] as duplicate_file:
        duplicates = filter_step.read_duplicates(duplicate_file)

    assert SENTENCE_SIGNATURE_FORMAT_VERSION == 2
    assert [(int(row["doc"]), int(row["sent"])) for row in duplicates] == [(1, 65_536)]

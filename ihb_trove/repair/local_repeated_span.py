"""Bounded, exact local deduplication of sufficiently long blocks."""

import hashlib
import re
from collections import defaultdict, deque

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep

from ._common import add_evidence, repair_metadata


_MODULUS = 1 << 64
_BASE = 1_000_003


def _canonical(value: str) -> str:
    return " ".join(value.split())


def _line_hash(line: str) -> int:
    digest = hashlib.blake2b(_canonical(line).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


class LocalRepeatedSpanRepair(PipelineStep):
    """Remove nearby exact duplicates; short headings never qualify by themselves.

    Paragraphs are compared within ``paragraph_window`` blocks. A second pass
    uses a stable rolling hash of 2..N consecutive nonblank lines, verifies
    exact equality after a hash hit, and searches only ``line_window`` lines back.
    """

    name = "IHB local repeated span"
    type = "🩹 - REPAIR"

    def __init__(
        self,
        min_block_chars: int = 100,
        min_block_words: int = 15,
        paragraph_window: int = 6,
        min_span_lines: int = 2,
        max_span_lines: int = 5,
        line_window: int = 30,
    ) -> None:
        super().__init__()
        if min_block_chars < 1 or min_block_words < 1 or paragraph_window < 1 or line_window < 1:
            raise ValueError("length thresholds and windows must be positive")
        if min_span_lines < 2 or max_span_lines < min_span_lines:
            raise ValueError("require 2 <= min_span_lines <= max_span_lines")
        self.min_block_chars = min_block_chars
        self.min_block_words = min_block_words
        self.paragraph_window = paragraph_window
        self.min_span_lines = min_span_lines
        self.max_span_lines = max_span_lines
        self.line_window = line_window

    def _eligible(self, value: str) -> bool:
        return len(value) >= self.min_block_chars and len(value.split()) >= self.min_block_words

    def _paragraph_pass(self, text: str) -> tuple[str, int]:
        blocks = re.split(r"\n[ \t]*\n+", text)
        recent: dict[str, int] = {}
        recent_queue: deque[tuple[int, str]] = deque()
        kept: list[str] = []
        removed = 0
        for position, block in enumerate(blocks):
            while recent_queue and position - recent_queue[0][0] > self.paragraph_window:
                old_position, old_key = recent_queue.popleft()
                if recent.get(old_key) == old_position:
                    del recent[old_key]
            key = _canonical(block)
            if self._eligible(key) and key in recent:
                removed += 1
            else:
                kept.append(block)
            if self._eligible(key):
                recent[key] = position
                recent_queue.append((position, key))
        return "\n\n".join(kept), removed

    def _line_pass(self, text: str) -> tuple[str, int]:
        lines = text.split("\n")
        hashes = [_line_hash(line) for line in lines]
        lengths = [len(_canonical(line)) for line in lines]
        words = [len(line.split()) for line in lines]
        prefix = [0]
        powers = [1]
        for value in hashes:
            prefix.append((prefix[-1] * _BASE + value) % _MODULUS)
            powers.append(powers[-1] * _BASE % _MODULUS)
        removed = [False] * len(lines)
        removed_count = 0

        for span in range(self.max_span_lines, self.min_span_lines - 1, -1):
            candidates: dict[int, deque[int]] = defaultdict(deque)
            for start in range(len(lines) - span + 1):
                end = start + span
                if any(removed[start:end]) or any(not line.strip() for line in lines[start:end]):
                    continue
                if sum(lengths[start:end]) < self.min_block_chars or sum(words[start:end]) < self.min_block_words:
                    continue
                signature = (prefix[end] - prefix[start] * powers[span]) % _MODULUS
                earlier = candidates[signature]
                while earlier and start - earlier[0] > self.line_window:
                    earlier.popleft()
                current = [_canonical(line) for line in lines[start:end]]
                match = next(
                    (
                        old
                        for old in reversed(earlier)
                        if old + span <= start
                        and not any(removed[old : old + span])
                        and [_canonical(line) for line in lines[old : old + span]] == current
                    ),
                    None,
                )
                if match is not None:
                    removed[start:end] = [True] * span
                    removed_count += 1
                else:
                    earlier.append(start)
        return "\n".join(line for index, line in enumerate(lines) if not removed[index]), removed_count

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for doc in data:
            with self.track_time():
                repair = repair_metadata(doc)
                text, paragraphs = self._paragraph_pass(doc.text)
                text, spans = self._line_pass(text)
                doc.text = text
                count = paragraphs + spans
                repair["duplicate_blocks_removed"] += count
                add_evidence(repair, count, 1.0)  # Hash hits are verified by exact text equality.
                self.stat_update("total")
                self.stat_update("forwarded")
                if count:
                    self.stat_update("duplicate_blocks_removed", value=count)
            yield doc

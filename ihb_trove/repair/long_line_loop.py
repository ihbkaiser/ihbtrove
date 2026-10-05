"""Collapse exact OCR copying loops contained in a single long line."""

import re

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep

from ._common import add_evidence, repair_metadata


class LongLineLoopRepair(PipelineStep):
    """Keep one copy of a repeated word span in an unusually long line.

    A run must repeat at least four times, contain a sufficiently long span,
    and remove a substantial number of characters. Word sequences are checked
    for exact equality; this does not infer missing OCR text.
    """

    name = "IHB long line loop"
    type = "🩹 - REPAIR"

    def __init__(
        self,
        min_line_chars: int = 1000,
        min_span_words: int = 4,
        max_span_words: int = 40,
        min_span_chars: int = 35,
        min_repetitions: int = 4,
        min_removed_run_chars: int = 120,
    ) -> None:
        super().__init__()
        if min_line_chars < 1 or min_span_words < 1 or max_span_words < min_span_words:
            raise ValueError("invalid line/span length limits")
        if min_span_chars < 1 or min_repetitions < 3 or min_removed_run_chars < 1:
            raise ValueError("invalid repetition limits")
        self.min_line_chars = min_line_chars
        self.min_span_words = min_span_words
        self.max_span_words = max_span_words
        self.min_span_chars = min_span_chars
        self.min_repetitions = min_repetitions
        self.min_removed_run_chars = min_removed_run_chars

    def _collapse_line(self, line: str) -> tuple[str, int]:
        if len(line) < self.min_line_chars:
            return line, 0
        tokens = list(re.finditer(r"\S+", line))
        words = [token.group() for token in tokens]
        parts: list[str] = []
        cursor = index = loops = 0
        while index + self.min_span_words * self.min_repetitions <= len(words):
            best: tuple[int, int, int] | None = None
            max_period = min(self.max_span_words, (len(words) - index) // self.min_repetitions)
            for period in range(self.min_span_words, max_period + 1):
                if words[index] != words[index + period]:
                    continue
                if tokens[index + period - 1].end() - tokens[index].start() < self.min_span_chars:
                    continue
                block = words[index : index + period]
                if block != words[index + period : index + 2 * period]:
                    continue
                repetitions = 2
                while (
                    index + (repetitions + 1) * period <= len(words)
                    and block == words[index + repetitions * period : index + (repetitions + 1) * period]
                ):
                    repetitions += 1
                if repetitions < self.min_repetitions:
                    continue
                gain = tokens[index + repetitions * period - 1].end() - tokens[index + period - 1].end()
                if gain >= self.min_removed_run_chars and (best is None or gain > best[0]):
                    best = gain, period, repetitions
            if best is None:
                index += 1
                continue
            _, period, repetitions = best
            parts.append(line[cursor : tokens[index + period - 1].end()])
            cursor = tokens[index + period * repetitions - 1].end()
            loops += 1
            index += period * repetitions
        parts.append(line[cursor:])
        return "".join(parts), loops

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for doc in data:
            with self.track_time():
                repair = repair_metadata(doc)
                lines: list[str] = []
                loops = removed = 0
                for line in doc.text.split("\n"):
                    collapsed, count = self._collapse_line(line)
                    lines.append(collapsed)
                    loops += count
                    removed += len(line) - len(collapsed)
                doc.text = "\n".join(lines)
                repair["long_line_loops_removed"] += loops
                repair["loop_chars_removed"] += removed
                add_evidence(repair, loops, 0.98)
                self.stat_update("total")
                self.stat_update("forwarded")
                if loops:
                    self.stat_update("long_line_loops_removed", value=loops)
            yield doc

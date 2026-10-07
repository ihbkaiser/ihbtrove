"""Language-neutral quality gates for repaired text documents."""

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter


class BookQualityFilter(BaseFilter):
    """Reject empty text and only the most extreme long-line repetition cases.

    The defaults intentionally avoid minimum document length and alphabetic
    fraction rules: short works, lists, non-Latin scripts, and structured text
    should remain eligible for downstream training and deduplication.
    """

    name = "IHB permissive text quality"

    def __init__(
        self,
        min_words: int = 0,
        max_repeated_line_char_fraction: float | None = 0.80,
        min_repeated_line_chars: int = 120,
        exclusion_writer: DiskWriter | None = None,
    ) -> None:
        super().__init__(exclusion_writer=exclusion_writer)
        if min_words < 0 or min_repeated_line_chars < 1:
            raise ValueError("min_words must be nonnegative and min_repeated_line_chars positive")
        if max_repeated_line_char_fraction is not None and not 0 <= max_repeated_line_char_fraction <= 1:
            raise ValueError("max_repeated_line_char_fraction must be within [0, 1] or None")
        self.min_words = min_words
        self.max_repeated_line_char_fraction = max_repeated_line_char_fraction
        self.min_repeated_line_chars = min_repeated_line_chars

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        text = doc.text
        if not text.strip():
            return False, "empty_text"

        if self.min_words and len(text.split()) < self.min_words:
            return False, "text_too_short"

        if self.max_repeated_line_char_fraction is None:
            return True

        seen: set[str] = set()
        repeated_chars = 0
        for line in text.splitlines():
            key = " ".join(line.split())
            if len(key) < self.min_repeated_line_chars:
                continue
            if key in seen:
                repeated_chars += len(key)
            seen.add(key)

        if repeated_chars / max(1, len(text)) > self.max_repeated_line_char_fraction:
            return False, "extreme_repeated_lines"
        return True

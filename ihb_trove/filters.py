"""Quality gates for repaired book documents."""

import re

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter


class RepairQualityFilter(BaseFilter):
    """Quarantine uncertain or overly destructive repair via exclusion_writer."""

    name = "IHB repair quality"

    def __init__(
        self,
        max_removed_fraction: float = 0.30,
        min_confidence: float = 0.55,
        exclusion_writer: DiskWriter | None = None,
    ) -> None:
        super().__init__(exclusion_writer=exclusion_writer)
        if not 0 <= max_removed_fraction <= 1 or not 0 <= min_confidence <= 1:
            raise ValueError("repair thresholds must be within [0, 1]")
        self.max_removed_fraction = max_removed_fraction
        self.min_confidence = min_confidence

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        repair = doc.metadata.get("repair")
        if not isinstance(repair, dict) or "repair_confidence" not in repair:
            return False, "missing_repair_metrics"
        if repair["removed_fraction"] > self.max_removed_fraction:
            return False, "excessive_repair"
        if repair["repair_confidence"] < self.min_confidence:
            return False, "uncertain_repair"
        return True


class BookQualityFilter(BaseFilter):
    """Book-friendly quality checks; ingredient lines need no terminal punctuation."""

    name = "IHB book quality"

    def __init__(
        self,
        min_words: int = 80,
        min_alpha_fraction: float = 0.45,
        max_repeated_line_char_fraction: float = 0.40,
        min_repeated_line_chars: int = 30,
        exclusion_writer: DiskWriter | None = None,
    ) -> None:
        super().__init__(exclusion_writer=exclusion_writer)
        if min_words < 1 or min_repeated_line_chars < 1:
            raise ValueError("minimum lengths must be positive")
        if not 0 <= min_alpha_fraction <= 1 or not 0 <= max_repeated_line_char_fraction <= 1:
            raise ValueError("fractions must be within [0, 1]")
        self.min_words = min_words
        self.min_alpha_fraction = min_alpha_fraction
        self.max_repeated_line_char_fraction = max_repeated_line_char_fraction
        self.min_repeated_line_chars = min_repeated_line_chars

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        text = doc.text
        if not text.strip():
            return False, "empty_book"
        if len(re.findall(r"\w+", text, flags=re.UNICODE)) < self.min_words:
            return False, "book_too_short"
        nonspace = sum(not char.isspace() for char in text)
        if not nonspace or sum(char.isalpha() for char in text) / nonspace < self.min_alpha_fraction:
            return False, "book_low_alpha_fraction"
        seen: set[str] = set()
        repeated_chars = 0
        for line in text.splitlines():
            key = " ".join(line.split())
            if len(key) < self.min_repeated_line_chars:
                continue
            if key in seen:
                repeated_chars += len(key)
            seen.add(key)
        if repeated_chars / len(text) > self.max_repeated_line_char_fraction:
            return False, "book_repeated_lines"
        return True

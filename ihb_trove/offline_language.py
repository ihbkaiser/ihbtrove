"""Deterministic CPU language gate for offline DataTrove installations."""

from langdetect import DetectorFactory, LangDetectException, detect_langs

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter

DetectorFactory.seed = 0


class OfflineLanguageFilter(BaseFilter):
    """Use langdetect without model downloads; retain DataTrove filter routing."""

    name = "IHB offline language"

    def __init__(
        self,
        languages: tuple[str, ...] = ("vi",),
        language_threshold: float = 0.65,
        exclusion_writer: DiskWriter | None = None,
    ) -> None:
        super().__init__(exclusion_writer=exclusion_writer)
        if not languages or not 0 <= language_threshold <= 1:
            raise ValueError("invalid languages or language_threshold")
        self.languages = languages
        self.language_threshold = language_threshold

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        try:
            estimates = detect_langs(doc.text[:20_000])
        except LangDetectException:
            return False, "language_unknown"
        if not estimates:
            return False, "language_unknown"
        top = estimates[0]
        doc.metadata["language"] = top.lang
        doc.metadata["language_score"] = round(top.prob, 6)
        if top.lang in self.languages and top.prob >= self.language_threshold:
            return True
        return False, "language_mismatch"

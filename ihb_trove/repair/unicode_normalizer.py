"""Conservative Unicode normalization that preserves Vietnamese accents."""

import unicodedata

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep

from ._common import repair_metadata


class UnicodeNormalizer(PipelineStep):
    name = "IHB Unicode normalizer"
    type = "🩹 - REPAIR"

    def __init__(self, form: str = "NFC", normalize_newlines: bool = True, remove_controls: bool = True) -> None:
        super().__init__()
        if form not in {"NFC", "NFKC", "NFD", "NFKD"}:
            raise ValueError(f"Unsupported Unicode form: {form}")
        self.form = form
        self.normalize_newlines = normalize_newlines
        self.remove_controls = remove_controls

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for doc in data:
            with self.track_time():
                repair = repair_metadata(doc)
                before = doc.text
                text = before.replace("\r\n", "\n").replace("\r", "\n") if self.normalize_newlines else before
                text = unicodedata.normalize(self.form, text)
                text = text.replace("\u00a0", " ").replace("\u200b", "")
                if self.remove_controls:
                    text = "".join(
                        char for char in text if char in "\n\t\f" or unicodedata.category(char) != "Cc"
                    )
                doc.text = text
                repair["unicode_chars_changed"] = abs(len(before) - len(text)) + (before != text)
                self.stat_update("total")
                self.stat_update("forwarded")
            yield doc

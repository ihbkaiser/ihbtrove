"""Discard text-only image references when the corpus has no image payloads."""

import re

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep

from ._common import add_evidence, repair_metadata


class ImagePlaceholderRepair(PipelineStep):
    """Remove Markdown/HTML image markup and generated unavailable-image notes.

    Enable this only for a text-only corpus without attached images. A mixed
    media corpus should keep its image references or use a different policy.
    """

    name = "IHB image placeholder"
    type = "🩹 - REPAIR"

    def __init__(
        self,
        remove_markdown: bool = True,
        remove_html: bool = True,
        remove_unavailable_notes: bool = True,
    ) -> None:
        super().__init__()
        self.remove_markdown = remove_markdown
        self.remove_html = remove_html
        self.remove_unavailable_notes = remove_unavailable_notes
        self._markdown = re.compile(r"!\[[^\]\n]*\]\([^\n)]*\)")
        self._html = re.compile(r"<img\b[^>]*>", re.IGNORECASE)

    @staticmethod
    def _is_generated_note(line: str) -> bool:
        value = line.strip().casefold()
        return value.startswith("note:") and (
            ("image" in value and "not available" in value)
            or ("image placeholder" in value and "text-based" in value)
        )

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for doc in data:
            with self.track_time():
                repair = repair_metadata(doc)
                kept: list[str] = []
                images = notes = 0
                for line in doc.text.split("\n"):
                    if self.remove_unavailable_notes and self._is_generated_note(line):
                        notes += 1
                        continue
                    updated = line
                    if self.remove_markdown:
                        updated, count = self._markdown.subn("", updated)
                        images += count
                    if self.remove_html:
                        updated, count = self._html.subn("", updated)
                        images += count
                    if updated.strip() or not line.strip():
                        kept.append(updated)
                doc.text = "\n".join(kept)
                repair["image_placeholders_removed"] += images
                repair["image_notes_removed"] += notes
                add_evidence(repair, images + notes, 1.0)
                self.stat_update("total")
                self.stat_update("forwarded")
                if images:
                    self.stat_update("image_placeholders_removed", value=images)
                if notes:
                    self.stat_update("image_notes_removed", value=notes)
            yield doc

"""Conservative Unicode normalization that preserves Vietnamese accents."""

import re
import unicodedata

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep

from ._common import repair_metadata


_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}
_LIGATURE_PATTERN = re.compile("|".join(re.escape(char) for char in _LIGATURES))
_SOFT_HYPHEN_BREAK = re.compile(r"(?<=[^\W\d_])\u00ad[ \t]*\n[ \t]*(?=[^\W\d_])", re.UNICODE)
# This visible-hyphen rule is deliberately opt-in: a line-final hyphen can be
# meaningful in names and compounds, so text-only heuristics cannot be certain.
_HYPHENATED_LINE_BREAK = re.compile(
    r"(?P<left>[^\W\d_]+)-[ \t]*\n[ \t]*(?P<right>[^\W\d_]+)", re.UNICODE
)


class UnicodeNormalizer(PipelineStep):
    name = "IHB Unicode normalizer"
    type = "🩹 - REPAIR"

    def __init__(
        self,
        form: str = "NFC",
        normalize_newlines: bool = True,
        remove_controls: bool = True,
        normalize_ligatures: bool = True,
        remove_soft_hyphens: bool = True,
        trim_trailing_whitespace: bool = True,
        max_consecutive_blank_lines: int | None = 2,
        dehyphenate_line_breaks: bool = False,
    ) -> None:
        super().__init__()
        if form not in {"NFC", "NFKC", "NFD", "NFKD"}:
            raise ValueError(f"Unsupported Unicode form: {form}")
        if max_consecutive_blank_lines is not None and max_consecutive_blank_lines < 0:
            raise ValueError("max_consecutive_blank_lines must be nonnegative or None")
        self.form = form
        self.normalize_newlines = normalize_newlines
        self.remove_controls = remove_controls
        self.normalize_ligatures = normalize_ligatures
        self.remove_soft_hyphens = remove_soft_hyphens
        self.trim_trailing_whitespace = trim_trailing_whitespace
        self.max_consecutive_blank_lines = max_consecutive_blank_lines
        self.dehyphenate_line_breaks = dehyphenate_line_breaks

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

                ligatures_normalized = 0
                if self.normalize_ligatures:
                    text, ligatures_normalized = _LIGATURE_PATTERN.subn(
                        lambda match: _LIGATURES[match.group()], text
                    )

                soft_hyphens_removed = soft_hyphen_line_joins = 0
                if self.remove_soft_hyphens:
                    text, soft_hyphen_line_joins = _SOFT_HYPHEN_BREAK.subn("", text)
                    soft_hyphens_removed = text.count("\u00ad") + soft_hyphen_line_joins
                    text = text.replace("\u00ad", "")

                line_end_hyphens_joined = 0
                if self.dehyphenate_line_breaks:
                    text, line_end_hyphens_joined = _HYPHENATED_LINE_BREAK.subn(
                        lambda match: match.group("left") + match.group("right"), text
                    )

                trailing_whitespace_removed = 0
                if self.trim_trailing_whitespace:
                    before_trim = text
                    text = re.sub(r"[ \t]+(?=\n|$)", "", text)
                    trailing_whitespace_removed = len(before_trim) - len(text)

                blank_lines_collapsed = 0
                if self.max_consecutive_blank_lines is not None:
                    max_newline_run = self.max_consecutive_blank_lines + 1
                    before_collapse = text
                    text = re.sub(rf"\n{{{max_newline_run + 1},}}", "\n" * max_newline_run, text)
                    blank_lines_collapsed = before_collapse.count("\n") - text.count("\n")

                doc.text = text
                repair["unicode_chars_changed"] = abs(len(before) - len(text)) + (before != text)
                repair["ligatures_normalized"] += ligatures_normalized
                repair["soft_hyphens_removed"] += soft_hyphens_removed
                repair["soft_hyphen_line_joins"] += soft_hyphen_line_joins
                repair["line_end_hyphens_joined"] += line_end_hyphens_joined
                repair["trailing_whitespace_removed"] += trailing_whitespace_removed
                repair["blank_lines_collapsed"] += blank_lines_collapsed
                self.stat_update("total")
                self.stat_update("forwarded")
                for key, value in (
                    ("ligatures_normalized", ligatures_normalized),
                    ("soft_hyphens_removed", soft_hyphens_removed),
                    ("soft_hyphen_line_joins", soft_hyphen_line_joins),
                    ("line_end_hyphens_joined", line_end_hyphens_joined),
                    ("trailing_whitespace_removed", trailing_whitespace_removed),
                    ("blank_lines_collapsed", blank_lines_collapsed),
                ):
                    if value:
                        self.stat_update(key, value=value)
            yield doc

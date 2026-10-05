"""Document-local page furniture learning and conservative boundary repair."""

import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable

from rapidfuzz.fuzz import ratio as fuzzy_ratio

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep

from ._common import add_evidence, repair_metadata


def _signature(line: str) -> str:
    value = unicodedata.normalize("NFKC", line).casefold()
    value = re.sub(r"\d+", "0", value)
    return re.sub(r"[^\w]+", " ", value, flags=re.UNICODE).strip()


def _nonempty_positions(lines: list[str], window: int) -> tuple[list[int], list[int]]:
    positions = [index for index, line in enumerate(lines) if line.strip()]
    return positions[:window], positions[-window:]


class PageStructureRepair(PipelineStep):
    """Learn recurring edge lines within one document, with no book-specific strings.

    Model candidates are taken from the first/last ``boundary_window`` nonempty
    lines of each page. Clusters require support on several distinct pages.
    Numeric lines additionally require a consistent printed-page offset.
    """

    name = "IHB page structure"
    type = "🩹 - REPAIR"

    def __init__(
        self,
        page_markers: Iterable[str] = ("---",),
        boundary_window: int = 2,
        min_recurrence_pages: int = 3,
        min_recurrence_fraction: float = 0.14,
        fuzzy_threshold: float = 86.0,
        min_furniture_chars: int = 20,
        max_furniture_chars: int = 100,
        min_join_left_chars: int = 25,
        min_join_left_words: int = 4,
    ) -> None:
        super().__init__()
        markers = tuple(page_markers)
        if not markers or any(not marker or "\n" in marker or "\f" in marker for marker in markers):
            raise ValueError("page_markers must contain nonempty single-line markers")
        if boundary_window < 1 or min_recurrence_pages < 2:
            raise ValueError("boundary_window >= 1 and min_recurrence_pages >= 2 are required")
        if not 0 < min_recurrence_fraction <= 1 or not 0 <= fuzzy_threshold <= 100:
            raise ValueError("invalid recurrence fraction or fuzzy threshold")
        self.page_markers = markers
        self.boundary_window = boundary_window
        self.min_recurrence_pages = min_recurrence_pages
        self.min_recurrence_fraction = min_recurrence_fraction
        self.fuzzy_threshold = fuzzy_threshold
        self.min_furniture_chars = min_furniture_chars
        self.max_furniture_chars = max_furniture_chars
        self.min_join_left_chars = min_join_left_chars
        self.min_join_left_words = min_join_left_words
        choices = "|".join(re.escape(marker) for marker in sorted(markers, key=len, reverse=True))
        self._boundary = re.compile(rf"(?m)^[ \t]*(?:{choices})[ \t]*(?:\n|$)|\f")

    def _is_furniture_candidate(self, value: str) -> bool:
        stripped = value.strip()
        return (
            self.min_furniture_chars <= len(stripped) <= self.max_furniture_chars
            and len(_signature(stripped).split()) >= 3
            and not stripped.endswith((".", "!", "?", ";", ":"))
        )

    def _furniture_clusters(self, pages: list[list[str]]) -> dict[tuple[str, str], float]:
        seen: dict[tuple[str, str], set[int]] = defaultdict(set)
        digit_values: dict[tuple[str, str], dict[tuple[str, ...], set[int]]] = defaultdict(lambda: defaultdict(set))
        for page_index, lines in enumerate(pages):
            first, last = _nonempty_positions(lines, self.boundary_window)
            for side, positions in (("header", first), ("footer", last)):
                for position in positions:
                    if self._is_furniture_candidate(lines[position]):
                        key = side, _signature(lines[position])
                        seen[key].add(page_index)
                        digit_values[key][tuple(re.findall(r"\d+", lines[position]))].add(page_index)

        supported: dict[tuple[str, str], float] = {}
        required = max(self.min_recurrence_pages, math.ceil(len(pages) * self.min_recurrence_fraction))
        for side in ("header", "footer"):
            signatures = sorted(
                (
                    (sig, page_ids)
                    for (kind, sig), page_ids in seen.items()
                    if kind == side
                    and (
                        not any(char.isdigit() for char in sig)
                        or max(map(len, digit_values[(kind, sig)].values())) >= required
                    )
                ),
                key=lambda item: (-len(item[1]), item[0]),
            )
            clusters: list[tuple[str, set[int], list[str]]] = []
            for sig, page_ids in signatures:
                for representative, cluster_pages, members in clusters:
                    if abs(len(sig) - len(representative)) > max(5, 0.2 * len(representative)):
                        continue
                    if fuzzy_ratio(sig, representative) >= self.fuzzy_threshold:
                        cluster_pages.update(page_ids)
                        members.append(sig)
                        break
                else:
                    clusters.append((sig, set(page_ids), [sig]))
            for _, cluster_pages, members in clusters:
                if len(cluster_pages) >= required:
                    support = len(cluster_pages) / len(pages)
                    for sig in members:
                        supported[(side, sig)] = support
        return supported

    @staticmethod
    def _number_candidate(line: str) -> tuple[int, str] | None:
        value = line.strip()
        if match := re.fullmatch(r"(\d{1,4})", value):
            return int(match.group(1)), ""
        if match := re.fullmatch(r"(.{4,75}?)[\s.·…-]+(\d{1,4})", value):
            prefix = _signature(match.group(1))
            if any(char.isalpha() for char in prefix):
                return int(match.group(2)), prefix
        return None

    def _pagination(self, pages: list[list[str]]) -> tuple[int | None, set[str], dict[int, int]]:
        offsets: Counter[int] = Counter()
        prefixes: dict[str, set[int]] = defaultdict(set)
        for page_index, lines in enumerate(pages):
            first, last = _nonempty_positions(lines, self.boundary_window)
            for position in set(first + last):
                candidate = self._number_candidate(lines[position])
                if candidate:
                    number, prefix = candidate
                    offsets[number - page_index] += 1
                    if prefix:
                        prefixes[prefix].add(page_index)
        if not offsets:
            return None, set(), {}
        offset, support = sorted(offsets.items(), key=lambda item: (-item[1], item[0]))[0]
        if support < self.min_recurrence_pages:
            return None, set(), {}
        required = max(self.min_recurrence_pages, math.ceil(len(pages) * self.min_recurrence_fraction))
        accepted_prefixes = {prefix for prefix, ids in prefixes.items() if len(ids) >= required}
        # A skipped/merged OCR page can shift the printed-page offset late in a
        # book. Trust a secondary offset only for a contiguous run of leading
        # page markers, never for isolated numbers in a table of contents.
        local_offsets: dict[int, int] = {}
        run: list[int] = []
        run_offset: int | None = None
        for page_index, lines in enumerate(pages + [[]]):
            first, _ = _nonempty_positions(lines, self.boundary_window)
            candidate = self._number_candidate(lines[first[0]]) if first else None
            current = (
                candidate[0] - page_index
                if candidate and (not candidate[1] or candidate[1] in accepted_prefixes)
                else None
            )
            if current != run_offset:
                if run_offset is not None and run_offset != offset and len(run) >= self.min_recurrence_pages:
                    local_offsets.update({index: run_offset for index in run})
                run, run_offset = [], current
            if current is not None:
                run.append(page_index)
        return offset, accepted_prefixes, local_offsets

    def _strip_inline_footer(self, line: str, clusters: dict[tuple[str, str], float]) -> tuple[str, float] | None:
        """Strip a learned footer pasted after an ellipsis at a page edge."""
        match = re.match(r"^(?P<body>.*?)(?:\.{2,}|…)[ \t]*(?P<tail>[^\n]{20,100})$", line)
        if not match or not match.group("body").strip():
            return None
        tail = _signature(match.group("tail"))
        for (side, signature), support in clusters.items():
            if side == "footer" and fuzzy_ratio(tail, signature) >= self.fuzzy_threshold:
                return match.group("body").rstrip(), support
        return None

    def _should_join(self, left: str, right: str) -> bool:
        if not left or not right or len(left) < self.min_join_left_chars:
            return False
        if len(left.split()) < self.min_join_left_words or not left[-1].isalpha():
            return False
        first = right[0]
        return first.isalpha() and first.islower() and not left.endswith((".", "!", "?", ":", ";", "…"))

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for doc in data:
            with self.track_time():
                repair = repair_metadata(doc)
                raw_pages = self._boundary.split(doc.text)
                pages = [page.split("\n") for page in raw_pages]
                repair["pages_detected"] = len(pages)
                repair["page_boundaries_detected"] = len(pages) - 1
                clusters = self._furniture_clusters(pages) if len(pages) >= self.min_recurrence_pages else {}
                offset, inline_prefixes, local_offsets = self._pagination(pages)
                cleaned: list[str] = []

                for page_index, lines in enumerate(pages):
                    first, last = _nonempty_positions(lines, self.boundary_window)
                    first_set, last_set = set(first), set(last)
                    remove: set[int] = set()
                    for position in sorted(first_set | last_set):
                        line = lines[position]
                        candidate = self._number_candidate(line)
                        expected_offset = local_offsets.get(page_index, offset)
                        if expected_offset is not None and candidate and candidate[0] - page_index == expected_offset:
                            if not candidate[1] or candidate[1] in inline_prefixes:
                                remove.add(position)
                                repair["page_numbers_removed"] += 1
                                add_evidence(repair, 1, 0.95)
                                continue
                        sig = _signature(line)
                        sides = (("header",) if position in first_set else ()) + (
                            ("footer",) if position in last_set else ()
                        )
                        for side in sides:
                            if (side, sig) in clusters:
                                remove.add(position)
                                repair[f"{side}s_removed"] += 1
                                add_evidence(repair, 1, 0.5 + 0.5 * clusters[(side, sig)])
                                break
                        if position not in remove and position in last_set:
                            inline = self._strip_inline_footer(line, clusters)
                            if inline:
                                lines[position] = inline[0]
                                repair["footers_removed"] += 1
                                add_evidence(repair, 1, 0.75 + 0.25 * inline[1])
                    cleaned.append("\n".join(line for index, line in enumerate(lines) if index not in remove).strip("\n"))

                output = ""
                for page in cleaned:
                    if not page:
                        continue
                    if not output:
                        output = page
                        continue
                    left = output.rsplit("\n", 1)[-1].strip()
                    right = page.split("\n", 1)[0].strip()
                    if self._should_join(left, right):
                        output = output.rstrip() + " " + page.lstrip()
                        repair["boundary_joins"] += 1
                        add_evidence(repair, 1, 0.9)
                    else:
                        output = output.rstrip() + "\n\n" + page.lstrip()
                doc.text = output
                self.stat_update("total")
                self.stat_update("forwarded")
                for key in ("headers_removed", "footers_removed", "page_numbers_removed", "boundary_joins"):
                    if repair[key]:
                        self.stat_update(key, value=repair[key])
            yield doc

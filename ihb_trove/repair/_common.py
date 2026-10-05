"""Shared, JSON-serializable repair metadata."""

from typing import Any

from datatrove.data import Document


REPAIR_VERSION = "0.3.0"
_COUNTERS = (
    "pages_detected",
    "page_boundaries_detected",
    "headers_removed",
    "footers_removed",
    "page_numbers_removed",
    "boundary_joins",
    "duplicate_blocks_removed",
    "image_placeholders_removed",
    "image_notes_removed",
    "long_line_loops_removed",
    "loop_chars_removed",
)


def repair_metadata(doc: Document) -> dict[str, Any]:
    """Initialize a new run without discarding unrelated document metadata."""
    repair = doc.metadata.setdefault("repair", {})
    if not isinstance(repair, dict):
        raise TypeError("document.metadata['repair'] must be a dictionary")
    if repair.get("repair_version") != REPAIR_VERSION or "chars_before" not in repair:
        repair.clear()
        repair.update({key: 0 for key in _COUNTERS})
        repair.update(
            chars_before=len(doc.text),
            chars_after=len(doc.text),
            removed_fraction=0.0,
            repair_confidence=1.0,
            repair_version=REPAIR_VERSION,
            _evidence_sum=0.0,
            _evidence_count=0,
        )
    return repair


def add_evidence(repair: dict[str, Any], count: int, confidence: float) -> None:
    repair["_evidence_sum"] = repair.get("_evidence_sum", 0.0) + count * max(0.0, min(1.0, confidence))
    repair["_evidence_count"] = repair.get("_evidence_count", 0) + count

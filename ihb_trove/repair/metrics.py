"""Finalize repair statistics before filtering or serialization."""

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep

from ._common import repair_metadata


class RepairMetrics(PipelineStep):
    name = "IHB repair metrics"
    type = "📊 - REPAIR"

    def __init__(self, removal_penalty_start: float = 0.20, removal_penalty_weight: float = 0.7) -> None:
        super().__init__()
        if not 0 <= removal_penalty_start <= 1 or removal_penalty_weight < 0:
            raise ValueError("invalid confidence penalty parameters")
        self.removal_penalty_start = removal_penalty_start
        self.removal_penalty_weight = removal_penalty_weight

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for doc in data:
            with self.track_time():
                repair = repair_metadata(doc)
                before = repair["chars_before"]
                after = len(doc.text)
                fraction = max(0.0, before - after) / before if before else 0.0
                count = repair.pop("_evidence_count", 0)
                evidence = repair.pop("_evidence_sum", 0.0)
                mean_evidence = evidence / count if count else 1.0
                confidence = 0.4 + 0.6 * mean_evidence
                confidence -= max(0.0, fraction - self.removal_penalty_start) * self.removal_penalty_weight
                repair["chars_after"] = after
                repair["removed_fraction"] = round(fraction, 6)
                repair["repair_confidence"] = round(max(0.0, min(1.0, confidence)), 6)
                self.stat_update("total")
                self.stat_update("forwarded")
                self.stat_update("chars_before", value=before)
                self.stat_update("chars_after", value=after)
            yield doc

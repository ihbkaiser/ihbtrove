"""End-to-end routing across recursive sources and global exact dedup."""

import json
from pathlib import Path

import pytest

import ihb_trove.folder_pipeline as folder_pipeline
from ihb_trove.folder_pipeline import run_folder_pipeline


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_recursive_mirror_invalid_records_and_global_duplicate(tmp_path: Path) -> None:
    source, output = tmp_path / "input", tmp_path / "output"
    (source / "nested").mkdir(parents=True)
    text = "\n".join(
        f"Người nấu chọn rau{index} cùng củ{index}, ngâm nước{index} rồi cắt lát{index} "
        f"trước khi đặt lên khay{index}. Sau đó gia vị{index} được nêm vào nồi{index} "
        f"để món ăn{index} có mùi thơm{index} và màu sắc{index} hài hòa."
        for index in range(50)
    )
    first = {"id": "same-id", "text": text, "metadata": {"keep": "original"}}
    (source / "one.jsonl").write_text(json.dumps(first, ensure_ascii=False) + "\nnot-json\n", encoding="utf-8")
    (source / "nested" / "two.jsonl").write_text(
        json.dumps(first, ensure_ascii=False) + "\n" + json.dumps({"text": ""}) + "\n",
        encoding="utf-8",
    )

    summary = run_folder_pipeline(source, output, tasks=2, workers=2)
    assert summary["input_files"] == 2
    assert summary["input_records"] == 4
    assert summary["survive"] == 1
    assert summary["eliminated"] == 3
    assert summary["stages"]["exact"] == 1
    assert _read(output / "eliminated" / "one.jsonl") or _read(output / "eliminated" / "nested" / "two.jsonl")
    assert len(_read(output / "survive" / "one.jsonl")) + len(_read(output / "survive" / "nested" / "two.jsonl")) == 1
    reasons = [
        row["metadata"]["filter_reason"]
        for path in (output / "eliminated").rglob("*.jsonl")
        for row in _read(path)
    ]
    assert set(reasons) == {"invalid_json", "missing_text", "exact_dedup"}
    assert _read(output / "survive" / "one.jsonl") or _read(output / "survive" / "nested" / "two.jsonl")
    assert (output / "summary.json").is_file()
    assert Path(summary["log_file"]).is_file()
    run_log = Path(summary["log_file"]).read_text(encoding="utf-8")
    assert "IHB-Trove run started" in run_log
    assert "START stage: repair and quality filters" in run_log
    assert "Run summary:" in run_log
    assert "IHB-Trove run finished" in run_log
    assert list((output / "logs" / "datatrove").rglob("task_*.log"))


def test_resume_reuses_completed_stages_and_checks_input_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    input_path = source / "book.jsonl"
    input_path.write_text(
        json.dumps({"id": "book-1", "text": "A book record."}) + "\nnot-json\n",
        encoding="utf-8",
    )
    calls = {"book": 0, "exact": 0, "sentence": 0, "minhash": 0}
    routed_once = False
    original_route_dropped = folder_pipeline._route_dropped

    class CopyExecutor:
        def __init__(self, source_path: Path, target_path: Path) -> None:
            self.source_path = source_path
            self.target_path = target_path

        def run(self) -> None:
            self.target_path.mkdir(parents=True, exist_ok=True)
            records = [record for path in self.source_path.rglob("*.jsonl") for record in _read(path)]
            target = self.target_path / "0.jsonl"
            with target.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def fake_book(source_path: Path, output_root: Path, **kwargs: object) -> CopyExecutor:
        calls["book"] += 1
        return CopyExecutor(source_path, output_root / "filtered")

    def fake_dedup(name: str):
        def build(source_path: Path, output_root: Path, **kwargs: object) -> CopyExecutor:
            calls[name] += 1
            if name == "sentence" and calls[name] == 1:
                raise RuntimeError("simulated interruption")
            return CopyExecutor(source_path, output_root / name)

        return build

    monkeypatch.setattr(folder_pipeline, "build_book_pipeline", fake_book)
    monkeypatch.setattr(folder_pipeline, "build_exact_dedup_pipeline", fake_dedup("exact"))
    monkeypatch.setattr(folder_pipeline, "build_sentence_dedup_pipeline", fake_dedup("sentence"))
    monkeypatch.setattr(folder_pipeline, "build_minhash_pipeline", fake_dedup("minhash"))

    def fail_once_during_routing(*args: object, **kwargs: object) -> int:
        nonlocal routed_once
        dropped = original_route_dropped(*args, **kwargs)
        if not routed_once:
            routed_once = True
            raise RuntimeError("simulated routing interruption")
        return dropped

    monkeypatch.setattr(folder_pipeline, "_route_dropped", fail_once_during_routing)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_folder_pipeline(source, output, tasks=1, workers=1)

    checkpoint = output / ".ihb_trove_cache" / "checkpoint.json"
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["completed_stages"] == ["input_prepared", "book_quality", "exact"]

    original_input = input_path.read_bytes()
    input_path.write_text(json.dumps({"id": "changed", "text": "Changed content."}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="content changed"):
        run_folder_pipeline(source, output, tasks=1, workers=1, resume=True)
    input_path.write_bytes(original_input)

    with pytest.raises(RuntimeError, match="routing interruption"):
        run_folder_pipeline(source, output, tasks=1, workers=1, resume=True)
    assert len(_read(output / "eliminated" / "book.jsonl")) == 1

    summary = run_folder_pipeline(source, output, tasks=1, workers=1, resume=True)
    assert summary["resumed_from_checkpoint"] is True
    assert calls == {"book": 1, "exact": 1, "sentence": 2, "minhash": 1}
    assert sum(len(_read(path)) for path in (output / "survive").rglob("*.jsonl")) == 1
    eliminated = [record for path in (output / "eliminated").rglob("*.jsonl") for record in _read(path)]
    assert len(eliminated) == 1
    assert eliminated[0]["metadata"]["filter_reason"] == "invalid_json"
    assert not (output / ".ihb_trove_cache").exists()
    assert run_folder_pipeline(source, output, tasks=1, workers=1, resume=True) == summary

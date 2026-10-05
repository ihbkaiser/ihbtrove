"""End-to-end routing across recursive sources and global exact dedup."""

import json
from pathlib import Path

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

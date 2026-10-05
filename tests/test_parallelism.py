from ihb_trove import folder_pipeline


def test_auto_parallelism_respects_cpu_limit_and_input_file_count(monkeypatch) -> None:
    monkeypatch.setattr(folder_pipeline, "_available_cpu_count", lambda: 224)
    assert folder_pipeline._resolve_parallelism(3, None, None) == (3, 3)
    assert folder_pipeline._resolve_parallelism(10_000, None, None) == (128, 32)
    assert folder_pipeline._resolve_parallelism(3, 8, 8) == (3, 3)

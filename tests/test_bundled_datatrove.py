"""Checks for the archive-specific DataTrove model lookup behavior."""

from pathlib import Path

import pytest

import datatrove
from datatrove.utils.lid import FT176LID


def test_datatrove_resolves_to_bundled_package() -> None:
    assert Path(datatrove.__file__).resolve().parents[1] == Path(__file__).resolve().parents[1]


def test_ft176_prefers_explicit_local_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = tmp_path / "lid.176.bin"
    model.touch()
    monkeypatch.setenv("IHB_TROVE_FT176_MODEL", str(model))
    assert FT176LID(["vi"]).MODEL_URL == str(model)


def test_ft176_rejects_missing_explicit_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("IHB_TROVE_FT176_MODEL", str(tmp_path / "missing.bin"))
    with pytest.raises(FileNotFoundError, match="IHB_TROVE_FT176_MODEL"):
        FT176LID(["vi"])


def test_ft176_retains_archive_path_or_public_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IHB_TROVE_FT176_MODEL", raising=False)
    original = FT176LID.MODEL_URL
    selected = FT176LID(["vi"]).MODEL_URL
    assert original == "/workspace/storage-shared/nlp/maitn4/code/data-processing/utils/lid.176.bin"
    assert selected == (original if Path(original).is_file() else "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin")

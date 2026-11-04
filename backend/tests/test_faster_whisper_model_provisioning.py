from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scripts import provision_faster_whisper_model as provisioning


def fake_snapshot_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_bin: bytes = b"fixture model",
    missing_file: str | None = None,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        destination = Path(str(kwargs["local_dir"]))
        for filename in provisioning.REQUIRED_MODEL_FILES:
            if filename != missing_file:
                content = model_bin if filename == "model.bin" else b"fixture"
                (destination / filename).write_bytes(content)
        return str(destination)

    fake_module = ModuleType("huggingface_hub")
    fake_module.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)
    return calls


def test_downloads_only_pinned_revision_verifies_and_installs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_bin = b"verified fixture"
    monkeypatch.setattr(
        provisioning,
        "EXPECTED_MODEL_BIN_SHA256",
        hashlib.sha256(model_bin).hexdigest(),
    )
    calls = fake_snapshot_module(monkeypatch, model_bin=model_bin)
    destination = tmp_path / "models" / "base"

    provisioning.provision_model(destination)

    assert len(calls) == 1
    assert calls[0]["repo_id"] == "Systran/faster-whisper-base"
    assert calls[0]["revision"] == "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"
    staging_path = Path(str(calls[0]["local_dir"]))
    assert staging_path.parent == destination.parent
    assert staging_path.name.startswith(".faster-whisper-base-")
    assert calls[0]["allow_patterns"] == list(provisioning.REQUIRED_MODEL_FILES)
    assert all((destination / filename).is_file() for filename in provisioning.REQUIRED_MODEL_FILES)
    assert (destination / "model.bin").read_bytes() == model_bin
    assert not list((tmp_path / "models").glob(".faster-whisper-base-*"))


def test_existing_destination_is_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "base"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("existing", encoding="utf-8")
    calls = fake_snapshot_module(monkeypatch)

    with pytest.raises(provisioning.ProvisioningError, match="already exists"):
        provisioning.provision_model(destination)

    assert marker.read_text(encoding="utf-8") == "existing"
    assert calls == []


def test_hash_mismatch_cleans_staging_and_does_not_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_snapshot_module(monkeypatch)
    destination = tmp_path / "base"

    with pytest.raises(provisioning.ProvisioningError, match="SHA-256"):
        provisioning.provision_model(destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".faster-whisper-base-*"))


def test_missing_required_file_cleans_staging_and_does_not_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_snapshot_module(monkeypatch, missing_file="tokenizer.json")
    destination = tmp_path / "base"

    with pytest.raises(provisioning.ProvisioningError, match="missing"):
        provisioning.provision_model(destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".faster-whisper-base-*"))

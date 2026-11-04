"""Manually provision the pinned Faster-Whisper base model."""

import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = REPOSITORY_ROOT / "models" / "faster-whisper" / "base"
MODEL_REPOSITORY = "Systran/faster-whisper-base"
MODEL_REVISION = "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"
EXPECTED_MODEL_BIN_SHA256 = (
    "d01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9"
)
REQUIRED_MODEL_FILES = (
    "config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.txt",
)


class ProvisioningError(Exception):
    """A safe, actionable model-provisioning failure."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for block in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provision_model(model_directory: Path = MODEL_DIRECTORY) -> None:
    if os.path.lexists(model_directory):
        raise ProvisioningError(
            "The destination already exists; move or inspect it manually before retrying."
        )

    model_directory.parent.mkdir(parents=True, exist_ok=True)
    staging_directory = Path(
        tempfile.mkdtemp(
            prefix=".faster-whisper-base-",
            dir=model_directory.parent,
        )
    )
    try:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ProvisioningError(
                "huggingface-hub is unavailable; install the pinned Faster-Whisper runtime first."
            ) from None

        snapshot_download(
            repo_id=MODEL_REPOSITORY,
            revision=MODEL_REVISION,
            local_dir=str(staging_directory),
            allow_patterns=list(REQUIRED_MODEL_FILES),
        )
        for filename in REQUIRED_MODEL_FILES:
            artifact = staging_directory / filename
            if artifact.is_symlink() or not artifact.is_file():
                raise ProvisioningError(
                    "The pinned model download is missing a required regular file."
                )

        if _sha256(staging_directory / "model.bin") != EXPECTED_MODEL_BIN_SHA256:
            raise ProvisioningError(
                "The downloaded model.bin failed SHA-256 integrity verification."
            )

        if os.path.lexists(model_directory):
            raise ProvisioningError(
                "The destination appeared during download; it was not overwritten."
            )
        staging_directory.rename(model_directory)
    except ProvisioningError:
        raise
    except Exception:
        raise ProvisioningError(
            "Pinned model provisioning failed; no verified model was installed."
        ) from None
    finally:
        if staging_directory.exists():
            shutil.rmtree(staging_directory)


def main() -> int:
    try:
        provision_model()
    except ProvisioningError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Model provisioned at: {MODEL_DIRECTORY}")
    print(f"Revision: {MODEL_REVISION}")
    print(f"model.bin SHA-256: {EXPECTED_MODEL_BIN_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

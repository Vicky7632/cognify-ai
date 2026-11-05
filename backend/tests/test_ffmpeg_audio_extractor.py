from io import BytesIO
import logging
import os
from pathlib import Path
import shutil
import struct
import subprocess
from subprocess import CompletedProcess
import wave

import pytest

from app.transcription.ffmpeg_audio_extractor import (
    FFmpegAudioExtractionError,
    FFmpegAudioExtractionErrorCode,
    FFmpegAudioExtractor,
)


def make_mp4(path: Path) -> Path:
    path.write_bytes(b"synthetic MP4 fixture")
    return path


def make_wav_bytes() -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 160)
    return output.getvalue()


def temporary_outputs(directory: Path) -> list[Path]:
    return list(directory.glob(".cognify-ffmpeg-*"))


def assert_extraction_error(
    code: FFmpegAudioExtractionErrorCode,
    operation,
) -> None:
    with pytest.raises(FFmpegAudioExtractionError) as raised:
        operation()

    assert raised.value.code is code
    assert str(raised.value) == f"Audio extraction failed: {code.value}."


def test_extracts_first_audio_stream_to_wav_with_paths_containing_spaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_video = make_mp4(tmp_path / "lesson video.mp4")
    output_audio = tmp_path / "extracted audio.wav"
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        calls.append((command, kwargs))
        output_path = Path(command[-1])
        output_path.write_bytes(make_wav_bytes())
        return CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "app.transcription.ffmpeg_audio_extractor.subprocess.run",
        run,
    )

    result = FFmpegAudioExtractor().extract(input_video, output_audio)

    assert result == output_audio
    assert output_audio.is_file()
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[0] == "ffmpeg"
    assert command[command.index("-i") + 1] == str(input_video)
    temporary_output = Path(command[-1])
    assert temporary_output.parent.parent == tmp_path
    assert temporary_output.name == "extracted.wav"
    assert temporary_output.suffix == ".wav"
    assert command[command.index("-map") + 1] == "0:a:0"
    assert command[command.index("-acodec") + 1] == "pcm_s16le"
    assert kwargs == {
        "check": False,
        "capture_output": True,
        "shell": False,
        "text": True,
        "timeout": 300,
    }
    assert output_audio.read_bytes() == make_wav_bytes()
    assert not temporary_outputs(tmp_path)


def test_missing_ffmpeg_is_reported_without_exposing_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_video = make_mp4(tmp_path / "private video.mp4")
    output_audio = tmp_path / "private output.wav"

    def run(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("sensitive path and environment details")

    monkeypatch.setattr(
        "app.transcription.ffmpeg_audio_extractor.subprocess.run",
        run,
    )

    assert_extraction_error(
        FFmpegAudioExtractionErrorCode.FFMPEG_UNAVAILABLE,
        lambda: FFmpegAudioExtractor().extract(input_video, output_audio),
    )
    assert not output_audio.exists()
    assert not temporary_outputs(tmp_path)


def test_nonzero_exit_cleans_partial_output_and_hides_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_video = make_mp4(tmp_path / "video.mp4")
    output_audio = tmp_path / "partial.wav"

    def run(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        Path(command[-1]).write_bytes(b"partial output")
        return CompletedProcess(
            command,
            1,
            stdout="",
            stderr="private paths and decoder diagnostics",
        )

    monkeypatch.setattr(
        "app.transcription.ffmpeg_audio_extractor.subprocess.run",
        run,
    )

    assert_extraction_error(
        FFmpegAudioExtractionErrorCode.EXTRACTION_FAILED,
        lambda: FFmpegAudioExtractor().extract(input_video, output_audio),
    )
    assert not output_audio.exists()
    assert not temporary_outputs(tmp_path)


@pytest.mark.parametrize(
    "input_kind",
    ["missing", "directory", "wrong_extension"],
)
def test_invalid_input_is_rejected_without_invoking_ffmpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_kind: str,
) -> None:
    if input_kind == "missing":
        input_video = tmp_path / "missing.mp4"
    elif input_kind == "directory":
        input_video = tmp_path / "video.mp4"
        input_video.mkdir()
    else:
        input_video = tmp_path / "video.mov"
        input_video.write_bytes(b"fixture")

    def unexpected_run(*_args: object, **_kwargs: object) -> None:
        pytest.fail("FFmpeg must not run for invalid input.")

    monkeypatch.setattr(
        "app.transcription.ffmpeg_audio_extractor.subprocess.run",
        unexpected_run,
    )

    assert_extraction_error(
        FFmpegAudioExtractionErrorCode.INVALID_INPUT,
        lambda: FFmpegAudioExtractor().extract(
            input_video,
            tmp_path / "output.wav",
        ),
    )


@pytest.mark.parametrize(
    ("output_name", "create_parent", "expected_code"),
    [
        ("output.mp3", True, FFmpegAudioExtractionErrorCode.INVALID_OUTPUT),
        (
            "output.wav",
            False,
            FFmpegAudioExtractionErrorCode.INVALID_OUTPUT,
        ),
    ],
)
def test_invalid_output_is_rejected_without_invoking_ffmpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_name: str,
    create_parent: bool,
    expected_code: FFmpegAudioExtractionErrorCode,
) -> None:
    input_video = make_mp4(tmp_path / "video.mp4")
    output_directory = tmp_path / "not-created"
    if create_parent:
        output_directory.mkdir()
    output_audio = output_directory / output_name

    monkeypatch.setattr(
        "app.transcription.ffmpeg_audio_extractor.subprocess.run",
        lambda *_args, **_kwargs: pytest.fail(
            "FFmpeg must not run for invalid output."
        ),
    )

    assert_extraction_error(
        expected_code,
        lambda: FFmpegAudioExtractor().extract(input_video, output_audio),
    )


def test_existing_output_is_not_overwritten_or_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_video = make_mp4(tmp_path / "video.mp4")
    output_audio = tmp_path / "existing.wav"
    output_audio.write_bytes(b"existing content")
    monkeypatch.setattr(
        "app.transcription.ffmpeg_audio_extractor.subprocess.run",
        lambda *_args, **_kwargs: pytest.fail(
            "FFmpeg must not run when output already exists."
        ),
    )

    assert_extraction_error(
        FFmpegAudioExtractionErrorCode.OUTPUT_ALREADY_EXISTS,
        lambda: FFmpegAudioExtractor().extract(input_video, output_audio),
    )
    assert output_audio.read_bytes() == b"existing content"


def test_timeout_is_sanitized_and_cleans_temporary_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_video = make_mp4(tmp_path / "timeout.mp4")

    def run(command: list[str], **kwargs: object) -> None:
        Path(command[-1]).write_bytes(b"partial output")
        assert kwargs["timeout"] == 2.5
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], "secret output")

    monkeypatch.setattr(
        "app.transcription.ffmpeg_audio_extractor.subprocess.run",
        run,
    )

    assert_extraction_error(
        FFmpegAudioExtractionErrorCode.TIMEOUT,
        lambda: FFmpegAudioExtractor(timeout_seconds=2.5).extract(
            input_video,
            tmp_path / "timeout.wav",
        ),
    )
    assert not temporary_outputs(tmp_path)


def test_concurrent_destination_creation_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_video = make_mp4(tmp_path / "race.mp4")
    output_audio = tmp_path / "race.wav"
    original_link = os.link

    def create_destination_then_link(source: Path, destination: Path) -> None:
        destination.write_bytes(b"created by another process")
        original_link(source, destination)

    monkeypatch.setattr(
        "app.transcription.ffmpeg_audio_extractor.os.link",
        create_destination_then_link,
    )
    def write_output(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        Path(command[-1]).write_bytes(make_wav_bytes())
        return CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "app.transcription.ffmpeg_audio_extractor.subprocess.run",
        write_output,
    )

    assert_extraction_error(
        FFmpegAudioExtractionErrorCode.OUTPUT_ALREADY_EXISTS,
        lambda: FFmpegAudioExtractor().extract(input_video, output_audio),
    )
    assert output_audio.read_bytes() == b"created by another process"
    assert not temporary_outputs(tmp_path)


def test_failed_publication_cleans_temporary_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_video = make_mp4(tmp_path / "publish.mp4")

    def fail_link(_source: Path, _destination: Path) -> None:
        raise OSError("private publication details")

    def write_output(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        Path(command[-1]).write_bytes(make_wav_bytes())
        return CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "app.transcription.ffmpeg_audio_extractor.os.link",
        fail_link,
    )
    monkeypatch.setattr(
        "app.transcription.ffmpeg_audio_extractor.subprocess.run",
        write_output,
    )

    assert_extraction_error(
        FFmpegAudioExtractionErrorCode.OUTPUT_PUBLISH_FAILED,
        lambda: FFmpegAudioExtractor().extract(
            input_video,
            tmp_path / "publish.wav",
        ),
    )
    assert not temporary_outputs(tmp_path)


def test_output_verification_oserror_is_sanitized_and_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_video = make_mp4(tmp_path / "verify.mp4")
    original_is_file = Path.is_file

    def run(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        Path(command[-1]).write_bytes(make_wav_bytes())
        return CompletedProcess(command, 0, stdout="", stderr="")

    def fail_temporary_is_file(path: Path) -> bool:
        if path.name == "extracted.wav":
            raise OSError("sensitive path details")
        return original_is_file(path)

    monkeypatch.setattr(
        "app.transcription.ffmpeg_audio_extractor.subprocess.run",
        run,
    )
    monkeypatch.setattr(Path, "is_file", fail_temporary_is_file)

    with pytest.raises(FFmpegAudioExtractionError) as raised:
        FFmpegAudioExtractor().extract(
            input_video,
            tmp_path / "verify.wav",
        )

    assert (
        raised.value.code
        is FFmpegAudioExtractionErrorCode.OUTPUT_VERIFICATION_FAILED
    )
    assert str(tmp_path) not in str(raised.value)
    assert "sensitive path details" not in str(raised.value)
    assert not (tmp_path / "verify.wav").exists()
    assert not temporary_outputs(tmp_path)


@pytest.mark.parametrize("removed_bytes", [1, 80])
def test_truncated_wav_payload_is_rejected_and_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    removed_bytes: int,
) -> None:
    input_video = make_mp4(tmp_path / "truncated.mp4")

    def run(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        Path(command[-1]).write_bytes(make_wav_bytes()[:-removed_bytes])
        return CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "app.transcription.ffmpeg_audio_extractor.subprocess.run",
        run,
    )

    assert_extraction_error(
        FFmpegAudioExtractionErrorCode.OUTPUT_VERIFICATION_FAILED,
        lambda: FFmpegAudioExtractor().extract(
            input_video,
            tmp_path / "truncated.wav",
        ),
    )
    assert not (tmp_path / "truncated.wav").exists()
    assert not temporary_outputs(tmp_path)


@pytest.mark.parametrize(
    ("offset", "format", "value"),
    [
        (22, "<H", 0),
        (24, "<I", 0),
        (34, "<H", 8),
        (40, "<I", 0),
    ],
)
def test_invalid_wav_format_metadata_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    offset: int,
    format: str,
    value: int,
) -> None:
    input_video = make_mp4(tmp_path / "invalid-wav.mp4")

    def run(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        wav_data = bytearray(make_wav_bytes())
        struct.pack_into(format, wav_data, offset, value)
        Path(command[-1]).write_bytes(wav_data)
        return CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "app.transcription.ffmpeg_audio_extractor.subprocess.run",
        run,
    )

    assert_extraction_error(
        FFmpegAudioExtractionErrorCode.OUTPUT_VERIFICATION_FAILED,
        lambda: FFmpegAudioExtractor().extract(
            input_video,
            tmp_path / "invalid-wav.wav",
        ),
    )
    assert not (tmp_path / "invalid-wav.wav").exists()
    assert not temporary_outputs(tmp_path)


def test_cleanup_failure_after_publication_keeps_success_and_logs_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    input_video = make_mp4(tmp_path / "cleanup.mp4")
    output_audio = tmp_path / "cleanup.wav"
    original_rmtree = shutil.rmtree

    def run(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        Path(command[-1]).write_bytes(make_wav_bytes())
        return CompletedProcess(command, 0, stdout="", stderr="")

    def fail_cleanup(_path: Path) -> None:
        raise OSError("private cleanup path details")

    monkeypatch.setattr(
        "app.transcription.ffmpeg_audio_extractor.subprocess.run",
        run,
    )
    monkeypatch.setattr(
        "app.transcription.ffmpeg_audio_extractor.shutil.rmtree",
        fail_cleanup,
    )

    with caplog.at_level(
        logging.WARNING,
        logger="app.transcription.ffmpeg_audio_extractor",
    ):
        result = FFmpegAudioExtractor().extract(input_video, output_audio)

    assert result == output_audio
    assert output_audio.read_bytes() == make_wav_bytes()
    assert "Temporary FFmpeg extraction files could not be removed." in caplog.text
    assert str(tmp_path) not in caplog.text
    assert "private cleanup path details" not in caplog.text

    monkeypatch.setattr(
        "app.transcription.ffmpeg_audio_extractor.shutil.rmtree",
        original_rmtree,
    )
    for temporary_directory in temporary_outputs(tmp_path):
        original_rmtree(temporary_directory)


@pytest.mark.parametrize("output_kind", ["missing", "empty"])
def test_success_without_valid_output_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_kind: str,
) -> None:
    input_video = make_mp4(tmp_path / "video.mp4")
    output_audio = tmp_path / "missing output.wav"

    def run(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        if output_kind == "empty":
            Path(command[-1]).touch()
        return CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "app.transcription.ffmpeg_audio_extractor.subprocess.run",
        run,
    )

    assert_extraction_error(
        FFmpegAudioExtractionErrorCode.OUTPUT_MISSING,
        lambda: FFmpegAudioExtractor().extract(input_video, output_audio),
    )
    assert not output_audio.exists()


def test_unexpected_subprocess_failure_cleans_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_video = make_mp4(tmp_path / "video.mp4")
    output_audio = tmp_path / "partial output.wav"

    def run(command: list[str], **_kwargs: object) -> None:
        Path(command[-1]).write_bytes(b"partial")
        raise RuntimeError("sensitive process details")

    monkeypatch.setattr(
        "app.transcription.ffmpeg_audio_extractor.subprocess.run",
        run,
    )

    assert_extraction_error(
        FFmpegAudioExtractionErrorCode.EXTRACTION_FAILED,
        lambda: FFmpegAudioExtractor().extract(input_video, output_audio),
    )
    assert not output_audio.exists()

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.models.document import DocumentFileType
from app.transcription import (
    FasterWhisperTranscriber,
    TranscriptionError,
    TranscriptionErrorCode,
    TranscriptionRequest,
)


MODEL_FILES = ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")


class FakeAudioContainer:
    def __init__(
        self,
        frames: list[object],
        *,
        failure: Exception | None = None,
    ) -> None:
        self.streams = SimpleNamespace(
            audio=[SimpleNamespace(index=0, rate=16_000)]
        )
        self._frames = frames
        self._failure = failure

    def __enter__(self) -> FakeAudioContainer:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def decode(self, *, audio: int) -> list[object]:
        assert audio == 0
        if self._failure is not None:
            raise self._failure
        return self._frames


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    path = tmp_path / "model"
    path.mkdir()
    for filename in MODEL_FILES:
        (path / filename).write_text("fixture", encoding="utf-8")
    return path


class FakeWhisperModel:
    def __init__(
        self,
        segments: object,
        *,
        failure: Exception | None = None,
    ) -> None:
        self.segments = segments
        self.failure = failure
        self.audio_path: str | None = None

    def transcribe(self, audio: str) -> tuple[object, object]:
        self.audio_path = audio
        if self.failure is not None:
            raise self.failure
        return self.segments, object()


def install_fake_runtime(
    monkeypatch,
    model_factory,
    *,
    audio_frames: list[object] | None = None,
    audio_failure: Exception | None = None,
    runtime_failure: Exception | None = None,
) -> list[tuple[str, dict[str, object]]]:
    calls: list[tuple[str, dict[str, object]]] = []
    frames = (
        audio_frames
        if audio_frames is not None
        else [SimpleNamespace(samples=16_000, sample_rate=16_000)]
    )

    def open_audio(_path: str, *, mode: str) -> FakeAudioContainer:
        assert mode == "r"
        return FakeAudioContainer(frames, failure=audio_failure)

    def import_module(name: str) -> object:
        if name == "av":
            return SimpleNamespace(open=open_audio)
        assert name == "faster_whisper"
        if runtime_failure is not None:
            raise runtime_failure

        def create_model(path: str, **kwargs: object) -> FakeWhisperModel:
            calls.append((path, kwargs))
            return model_factory(path, **kwargs)

        return SimpleNamespace(WhisperModel=create_model)

    monkeypatch.setattr(
        "app.transcription.faster_whisper_transcriber.importlib.import_module",
        import_module,
    )
    return calls


def make_request(
    audio_path: Path,
    file_type: DocumentFileType = DocumentFileType.WAV,
) -> TranscriptionRequest:
    return TranscriptionRequest(audio_path, file_type)


def test_missing_runtime_dependency_returns_safe_error(
    model_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    install_fake_runtime(
        monkeypatch,
        lambda *_args, **_kwargs: pytest.fail("Missing runtime cannot load."),
        runtime_failure=ModuleNotFoundError("sensitive runtime details"),
    )
    transcriber = FasterWhisperTranscriber(model_path)

    with pytest.raises(TranscriptionError) as raised:
        transcriber.transcribe(make_request(tmp_path / "audio.wav"))

    assert raised.value.code is TranscriptionErrorCode.RUNTIME_UNAVAILABLE
    assert str(raised.value) == "Transcription failed: runtime_unavailable."


def test_transcriber_construction_keeps_runtime_loading_lazy(
    model_path: Path,
    monkeypatch,
) -> None:
    imports: list[str] = []
    monkeypatch.setattr(
        "app.transcription.faster_whisper_transcriber.importlib.import_module",
        lambda name: imports.append(name),
    )

    FasterWhisperTranscriber(model_path)

    assert imports == []


def test_missing_model_directory_returns_model_unavailable(
    tmp_path: Path,
) -> None:
    transcriber = FasterWhisperTranscriber(tmp_path / "missing-model")

    with pytest.raises(TranscriptionError) as raised:
        transcriber.transcribe(make_request(tmp_path / "audio.wav"))

    assert raised.value.code is TranscriptionErrorCode.MODEL_UNAVAILABLE
    assert str(raised.value) == "Transcription failed: model_unavailable."


def test_incomplete_model_directory_returns_model_unavailable(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "incomplete-model"
    model_path.mkdir()
    (model_path / "model.bin").write_text("fixture", encoding="utf-8")

    with pytest.raises(TranscriptionError) as raised:
        FasterWhisperTranscriber(model_path).transcribe(
            make_request(tmp_path / "audio.wav")
        )

    assert raised.value.code is TranscriptionErrorCode.MODEL_UNAVAILABLE


def test_model_loading_failure_is_safe(
    model_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = install_fake_runtime(
        monkeypatch,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("local model path and internal details")
        ),
    )

    with pytest.raises(TranscriptionError) as raised:
        FasterWhisperTranscriber(model_path, cpu_threads=2).transcribe(
            make_request(tmp_path / "audio.wav")
        )

    assert raised.value.code is TranscriptionErrorCode.MODEL_UNAVAILABLE
    assert "local model path" not in str(raised.value)
    assert calls == [
        (
            str(model_path),
            {
                "device": "cpu",
                "compute_type": "int8",
                "cpu_threads": 2,
                "local_files_only": True,
            },
        )
    ]


def test_valid_output_preserves_text_timestamps_and_order(
    model_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime_model = FakeWhisperModel(
        [
            SimpleNamespace(text=" First ", start=0.25, end=1.5),
            SimpleNamespace(text="Second.", start=1.75, end=2.5),
        ]
    )
    calls = install_fake_runtime(
        monkeypatch,
        lambda *_args, **_kwargs: runtime_model,
    )
    audio_path = tmp_path / "audio.wav"

    result = FasterWhisperTranscriber(model_path).transcribe(
        make_request(audio_path)
    )

    assert [
        (segment.text, segment.start_seconds, segment.end_seconds)
        for segment in result.segments
    ] == [
        (" First ", 0.25, 1.5),
        ("Second.", 1.75, 2.5),
    ]
    assert runtime_model.audio_path == str(audio_path)
    assert len(calls) == 1


def test_custom_compute_type_is_passed_to_runtime(
    model_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime_model = FakeWhisperModel(
        [SimpleNamespace(text="Text", start=0.0, end=0.5)]
    )
    calls = install_fake_runtime(
        monkeypatch,
        lambda *_args, **_kwargs: runtime_model,
    )

    FasterWhisperTranscriber(
        model_path,
        compute_type="int8_float32",
    ).transcribe(make_request(tmp_path / "audio.wav"))

    assert calls[0][1]["compute_type"] == "int8_float32"


def test_invalid_compute_type_is_rejected_by_adapter(model_path: Path) -> None:
    with pytest.raises(ValueError, match="supported value"):
        FasterWhisperTranscriber(model_path, compute_type="cuda")


def test_exactly_maximum_audio_duration_is_accepted(
    model_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = install_fake_runtime(
        monkeypatch,
        lambda *_args, **_kwargs: FakeWhisperModel(
            [SimpleNamespace(text="At limit", start=0.0, end=300.0)]
        ),
        audio_frames=[
            SimpleNamespace(samples=300 * 16_000, sample_rate=16_000)
        ],
    )

    result = FasterWhisperTranscriber(model_path).transcribe(
        make_request(tmp_path / "audio.wav")
    )

    assert result.segments[0].text == "At limit"
    assert len(calls) == 1


def test_audio_over_maximum_is_rejected_before_model_loading(
    model_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = install_fake_runtime(
        monkeypatch,
        lambda *_args, **_kwargs: pytest.fail(
            "Model loading must not occur for over-limit audio."
        ),
        audio_frames=[
            SimpleNamespace(samples=300 * 16_000 + 1, sample_rate=16_000)
        ],
    )

    with pytest.raises(TranscriptionError) as raised:
        FasterWhisperTranscriber(model_path).transcribe(
            make_request(tmp_path / "too-long.wav")
        )

    assert raised.value.code is TranscriptionErrorCode.AUDIO_TOO_LONG
    assert calls == []


def test_custom_audio_duration_limit_is_respected(
    model_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = install_fake_runtime(
        monkeypatch,
        lambda *_args, **_kwargs: pytest.fail(
            "Model loading must not occur for over-limit audio."
        ),
        audio_frames=[
            SimpleNamespace(samples=6 * 16_000, sample_rate=16_000)
        ],
    )

    with pytest.raises(TranscriptionError) as raised:
        FasterWhisperTranscriber(
            model_path,
            max_audio_seconds=5,
        ).transcribe(make_request(tmp_path / "custom-limit.wav"))

    assert raised.value.code is TranscriptionErrorCode.AUDIO_TOO_LONG
    assert calls == []


@pytest.mark.parametrize("file_type", [DocumentFileType.MP3, DocumentFileType.WAV])
def test_unreadable_audio_fails_safely_before_model_loading(
    model_path: Path,
    tmp_path: Path,
    monkeypatch,
    file_type: DocumentFileType,
) -> None:
    calls = install_fake_runtime(
        monkeypatch,
        lambda *_args, **_kwargs: pytest.fail(
            "Model loading must not occur for unreadable audio."
        ),
        audio_failure=RuntimeError("private decoder detail"),
    )

    with pytest.raises(TranscriptionError) as raised:
        FasterWhisperTranscriber(model_path).transcribe(
            make_request(tmp_path / f"corrupt.{file_type.value}", file_type)
        )

    assert raised.value.code is TranscriptionErrorCode.INVALID_TRANSCRIPTION
    assert "private decoder detail" not in str(raised.value)
    assert calls == []


@pytest.mark.parametrize("file_type", [DocumentFileType.MP3, DocumentFileType.WAV])
def test_empty_audio_input_fails_before_model_loading(
    model_path: Path,
    tmp_path: Path,
    monkeypatch,
    file_type: DocumentFileType,
) -> None:
    inference_calls: list[str] = []

    class TrackingWhisperModel:
        def transcribe(self, audio: str) -> tuple[object, object]:
            inference_calls.append(audio)
            return [], object()

    model_construction_calls = install_fake_runtime(
        monkeypatch,
        lambda *_args, **_kwargs: TrackingWhisperModel(),
        audio_frames=[],
    )

    with pytest.raises(TranscriptionError) as raised:
        FasterWhisperTranscriber(model_path).transcribe(
            make_request(tmp_path / f"empty.{file_type.value}", file_type)
        )

    assert raised.value.code is TranscriptionErrorCode.INVALID_TRANSCRIPTION
    assert model_construction_calls == []
    assert inference_calls == []


@pytest.mark.parametrize("file_type", [DocumentFileType.MP3, DocumentFileType.WAV])
def test_empty_output_is_invalid_transcription(
    model_path: Path,
    tmp_path: Path,
    monkeypatch,
    file_type: DocumentFileType,
) -> None:
    install_fake_runtime(monkeypatch, lambda *_args, **_kwargs: FakeWhisperModel([]))

    with pytest.raises(TranscriptionError) as raised:
        FasterWhisperTranscriber(model_path).transcribe(
            make_request(tmp_path / f"audio.{file_type.value}", file_type)
        )

    assert raised.value.code is TranscriptionErrorCode.INVALID_TRANSCRIPTION


@pytest.mark.parametrize("file_type", [DocumentFileType.MP3, DocumentFileType.WAV])
def test_corrupt_audio_failure_is_safe(
    model_path: Path,
    tmp_path: Path,
    monkeypatch,
    file_type: DocumentFileType,
) -> None:
    install_fake_runtime(
        monkeypatch,
        lambda *_args, **_kwargs: FakeWhisperModel(
            [],
            failure=RuntimeError("decoder path and payload details"),
        ),
    )

    with pytest.raises(TranscriptionError) as raised:
        FasterWhisperTranscriber(model_path).transcribe(
            make_request(tmp_path / f"corrupt.{file_type.value}", file_type)
        )

    assert raised.value.code is TranscriptionErrorCode.INVALID_TRANSCRIPTION
    assert "decoder path" not in str(raised.value)


def test_malformed_segment_is_rejected(
    model_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    install_fake_runtime(
        monkeypatch,
        lambda *_args, **_kwargs: FakeWhisperModel([object()]),
    )

    with pytest.raises(TranscriptionError) as raised:
        FasterWhisperTranscriber(model_path).transcribe(
            make_request(tmp_path / "audio.wav")
        )

    assert raised.value.code is TranscriptionErrorCode.MALFORMED_OUTPUT


def test_malformed_runtime_return_shape_is_rejected(
    model_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    class MalformedModel:
        def transcribe(self, _audio: str) -> object:
            return object()

    install_fake_runtime(
        monkeypatch,
        lambda *_args, **_kwargs: MalformedModel(),
    )

    with pytest.raises(TranscriptionError) as raised:
        FasterWhisperTranscriber(model_path).transcribe(
            make_request(tmp_path / "audio.wav")
        )

    assert raised.value.code is TranscriptionErrorCode.MALFORMED_OUTPUT


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (float("nan"), 1.0),
        (0.0, float("inf")),
        (-0.1, 1.0),
        (1.0, 0.5),
    ],
)
def test_malformed_timestamps_are_rejected(
    model_path: Path,
    tmp_path: Path,
    monkeypatch,
    start: float,
    end: float,
) -> None:
    install_fake_runtime(
        monkeypatch,
        lambda *_args, **_kwargs: FakeWhisperModel(
            [SimpleNamespace(text="Text", start=start, end=end)]
        ),
    )

    with pytest.raises(TranscriptionError) as raised:
        FasterWhisperTranscriber(model_path).transcribe(
            make_request(tmp_path / "audio.wav")
        )

    assert raised.value.code is TranscriptionErrorCode.MALFORMED_OUTPUT


def test_out_of_order_segments_are_rejected(
    model_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    install_fake_runtime(
        monkeypatch,
        lambda *_args, **_kwargs: FakeWhisperModel(
            [
                SimpleNamespace(text="Later", start=2.0, end=3.0),
                SimpleNamespace(text="Earlier", start=1.0, end=1.5),
            ]
        ),
    )

    with pytest.raises(TranscriptionError) as raised:
        FasterWhisperTranscriber(model_path).transcribe(
            make_request(tmp_path / "audio.wav")
        )

    assert raised.value.code is TranscriptionErrorCode.MALFORMED_OUTPUT


def test_lazy_segment_decode_failure_is_safe(
    model_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    def failing_segments():
        raise RuntimeError("private decoder detail")
        yield

    install_fake_runtime(
        monkeypatch,
        lambda *_args, **_kwargs: FakeWhisperModel(failing_segments()),
    )

    with pytest.raises(TranscriptionError) as raised:
        FasterWhisperTranscriber(model_path).transcribe(
            make_request(tmp_path / "audio.wav")
        )

    assert raised.value.code is TranscriptionErrorCode.INVALID_TRANSCRIPTION
    assert "private decoder detail" not in str(raised.value)

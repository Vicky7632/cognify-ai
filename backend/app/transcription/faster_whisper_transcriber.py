import importlib
from fractions import Fraction
from math import isfinite
from pathlib import Path
from threading import Lock
from typing import Protocol

from app.core.config import SUPPORTED_TRANSCRIPTION_COMPUTE_TYPES
from app.models.document import DocumentFileType
from app.transcription.contracts import (
    Transcriber,
    TranscriptionError,
    TranscriptionErrorCode,
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptionSegment,
)

REQUIRED_MODEL_FILES = (
    "config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.txt",
)
class _RuntimeModel(Protocol):
    def transcribe(self, audio: str) -> tuple[object, object]: ...


class FasterWhisperTranscriber(Transcriber):
    """Lazy CPU adapter using an already provisioned local model."""

    def __init__(
        self,
        model_path: Path,
        *,
        compute_type: str = "int8",
        cpu_threads: int = 2,
        max_audio_seconds: float = 300,
    ) -> None:
        if not isinstance(model_path, Path):
            raise TypeError("model_path must be a Path.")
        if (
            not isinstance(compute_type, str)
            or compute_type.strip().lower()
            not in SUPPORTED_TRANSCRIPTION_COMPUTE_TYPES
        ):
            raise ValueError("compute_type must be a supported value.")
        if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int):
            raise TypeError("cpu_threads must be an integer.")
        if cpu_threads <= 0:
            raise ValueError("cpu_threads must be greater than zero.")
        if (
            isinstance(max_audio_seconds, bool)
            or not isinstance(max_audio_seconds, (int, float))
            or not isfinite(max_audio_seconds)
            or max_audio_seconds <= 0
        ):
            raise ValueError("max_audio_seconds must be greater than zero.")

        self._model_path = model_path
        self._compute_type = compute_type.strip().lower()
        self._cpu_threads = cpu_threads
        self._max_audio_seconds = max_audio_seconds
        self._model: _RuntimeModel | None = None
        self._model_lock = Lock()

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        if not isinstance(request, TranscriptionRequest):
            raise TypeError("request must be a TranscriptionRequest.")
        if request.file_type not in (DocumentFileType.MP3, DocumentFileType.WAV):
            raise TranscriptionError(TranscriptionErrorCode.UNSUPPORTED_MEDIA)

        if not self._has_required_model_files():
            raise TranscriptionError(TranscriptionErrorCode.MODEL_UNAVAILABLE)
        self._validate_audio_duration(request.audio_path)
        model = self._get_model()
        try:
            output = model.transcribe(str(request.audio_path))
        except Exception:
            raise TranscriptionError(
                TranscriptionErrorCode.INVALID_TRANSCRIPTION
            ) from None

        if not isinstance(output, tuple) or len(output) != 2:
            raise TranscriptionError(TranscriptionErrorCode.MALFORMED_OUTPUT)
        runtime_segments, _ = output
        try:
            iterator = iter(runtime_segments)
        except Exception:
            raise TranscriptionError(
                TranscriptionErrorCode.MALFORMED_OUTPUT
            ) from None

        segments: list[TranscriptionSegment] = []
        while True:
            try:
                runtime_segment = next(iterator)
            except StopIteration:
                break
            except Exception:
                raise TranscriptionError(
                    TranscriptionErrorCode.INVALID_TRANSCRIPTION
                ) from None

            try:
                text = runtime_segment.text
                start_seconds = runtime_segment.start
                end_seconds = runtime_segment.end
            except Exception:
                raise TranscriptionError(
                    TranscriptionErrorCode.MALFORMED_OUTPUT
                ) from None
            if not isinstance(text, str):
                raise TranscriptionError(
                    TranscriptionErrorCode.MALFORMED_OUTPUT
                )
            try:
                segments.append(
                    TranscriptionSegment(
                        text=text,
                        start_seconds=start_seconds,
                        end_seconds=end_seconds,
                    )
                )
            except TranscriptionError:
                raise
            except Exception:
                raise TranscriptionError(
                    TranscriptionErrorCode.MALFORMED_OUTPUT
                ) from None

        try:
            return TranscriptionResult(segments=tuple(segments))
        except TranscriptionError:
            raise
        except Exception:
            raise TranscriptionError(
                TranscriptionErrorCode.MALFORMED_OUTPUT
            ) from None

    def _get_model(self) -> _RuntimeModel:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model

            if not self._has_required_model_files():
                raise TranscriptionError(TranscriptionErrorCode.MODEL_UNAVAILABLE)
            try:
                runtime = importlib.import_module("faster_whisper")
            except Exception:
                raise TranscriptionError(
                    TranscriptionErrorCode.RUNTIME_UNAVAILABLE
                ) from None
            try:
                model = runtime.WhisperModel(
                    str(self._model_path),
                    device="cpu",
                    compute_type=self._compute_type,
                    cpu_threads=self._cpu_threads,
                    local_files_only=True,
                )
            except Exception:
                raise TranscriptionError(
                    TranscriptionErrorCode.MODEL_UNAVAILABLE
                ) from None

            self._model = model
            return model

    def _validate_audio_duration(self, audio_path: Path) -> None:
        try:
            audio_decoder = importlib.import_module("av")
        except Exception:
            raise TranscriptionError(
                TranscriptionErrorCode.RUNTIME_UNAVAILABLE
            ) from None

        try:
            with audio_decoder.open(str(audio_path), mode="r") as container:
                audio_streams = container.streams.audio
                if not audio_streams:
                    raise TranscriptionError(
                        TranscriptionErrorCode.INVALID_TRANSCRIPTION
                    )

                audio_stream = audio_streams[0]
                decoded_duration = Fraction(0)
                decoded_samples = 0
                for frame in container.decode(audio=audio_stream.index):
                    samples = frame.samples
                    sample_rate = frame.sample_rate or audio_stream.rate
                    if (
                        isinstance(samples, bool)
                        or not isinstance(samples, int)
                        or samples < 0
                        or isinstance(sample_rate, bool)
                        or not isinstance(sample_rate, int)
                        or sample_rate <= 0
                    ):
                        raise TranscriptionError(
                            TranscriptionErrorCode.INVALID_TRANSCRIPTION
                        )
                    decoded_samples += samples
                    decoded_duration += Fraction(samples, sample_rate)
                    if decoded_duration > Fraction(
                        str(self._max_audio_seconds)
                    ):
                        raise TranscriptionError(
                            TranscriptionErrorCode.AUDIO_TOO_LONG
                        )
                if decoded_samples == 0:
                    raise TranscriptionError(
                        TranscriptionErrorCode.INVALID_TRANSCRIPTION
                    )
        except TranscriptionError:
            raise
        except Exception:
            raise TranscriptionError(
                TranscriptionErrorCode.INVALID_TRANSCRIPTION
            ) from None

    def _has_required_model_files(self) -> bool:
        try:
            return self._model_path.is_dir() and all(
                (self._model_path / filename).is_file()
                for filename in REQUIRED_MODEL_FILES
            )
        except OSError:
            return False

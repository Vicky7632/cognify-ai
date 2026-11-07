import logging
import math
import os
import shutil
import struct
import subprocess
import tempfile
import wave
from enum import StrEnum
from pathlib import Path


logger = logging.getLogger(__name__)


class FFmpegAudioExtractionErrorCode(StrEnum):
    INVALID_INPUT = "invalid_input"
    INVALID_OUTPUT = "invalid_output"
    OUTPUT_ALREADY_EXISTS = "output_already_exists"
    FFMPEG_UNAVAILABLE = "ffmpeg_unavailable"
    EXTRACTION_FAILED = "extraction_failed"
    TIMEOUT = "timeout"
    OUTPUT_MISSING = "output_missing"
    OUTPUT_VERIFICATION_FAILED = "output_verification_failed"
    OUTPUT_PUBLISH_FAILED = "output_publish_failed"
    CLEANUP_FAILED = "cleanup_failed"


class FFmpegAudioExtractionError(RuntimeError):
    """Safe FFmpeg failure that exposes only a stable error category."""

    def __init__(self, code: FFmpegAudioExtractionErrorCode) -> None:
        self.code = FFmpegAudioExtractionErrorCode(code)
        super().__init__(f"Audio extraction failed: {self.code.value}.")


class FFmpegAudioExtractor:
    """Extract the first MP4 audio stream to uncompressed PCM WAV."""

    def __init__(
        self,
        executable: str = "ffmpeg",
        *,
        timeout_seconds: float = 300,
    ) -> None:
        if not isinstance(executable, str) or not executable.strip():
            raise ValueError("FFmpeg executable must not be empty.")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("FFmpeg timeout must be greater than zero.")
        self._executable = executable
        self._timeout_seconds = timeout_seconds

    def extract(self, input_video: Path, output_audio: Path) -> Path:
        self._validate_input(input_video)
        self._validate_output(output_audio)
        executable = self._find_executable()

        try:
            temporary_directory = Path(
                tempfile.mkdtemp(
                    prefix=".cognify-ffmpeg-",
                    dir=output_audio.parent,
                )
            )
        except OSError:
            raise FFmpegAudioExtractionError(
                FFmpegAudioExtractionErrorCode.OUTPUT_PUBLISH_FAILED
            ) from None

        temporary_audio = temporary_directory / "extracted.wav"
        command = [
            executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-i",
            str(input_video),
            "-map",
            "0:a:0",
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-f",
            "wav",
            str(temporary_audio),
        ]
        try:
            self._run_ffmpeg(command)
            self._validate_extracted_audio(temporary_audio)
            self._publish_without_overwrite(temporary_audio, output_audio)
            return output_audio
        finally:
            try:
                self._remove_temporary_directory(temporary_directory)
            except FFmpegAudioExtractionError:
                logger.warning(
                    "Temporary FFmpeg extraction files could not be removed."
                )

    def _run_ffmpeg(self, command: list[str]) -> None:
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                shell=False,
                text=True,
                timeout=self._timeout_seconds,
            )
        except FileNotFoundError:
            raise FFmpegAudioExtractionError(
                FFmpegAudioExtractionErrorCode.FFMPEG_UNAVAILABLE
            ) from None
        except subprocess.TimeoutExpired:
            raise FFmpegAudioExtractionError(
                FFmpegAudioExtractionErrorCode.TIMEOUT
            ) from None
        except OSError:
            raise FFmpegAudioExtractionError(
                FFmpegAudioExtractionErrorCode.EXTRACTION_FAILED
            ) from None
        except Exception:
            raise FFmpegAudioExtractionError(
                FFmpegAudioExtractionErrorCode.EXTRACTION_FAILED
            ) from None

        if result.returncode != 0:
            raise FFmpegAudioExtractionError(
                FFmpegAudioExtractionErrorCode.EXTRACTION_FAILED
            )

    @staticmethod
    def _validate_input(input_video: Path) -> None:
        try:
            is_valid = (
                isinstance(input_video, Path)
                and input_video.is_file()
                and input_video.suffix.lower() == ".mp4"
            )
            if is_valid:
                is_valid = FFmpegAudioExtractor._has_valid_mp4_container(
                    input_video
                )
        except OSError:
            is_valid = False
        if not is_valid:
            raise FFmpegAudioExtractionError(
                FFmpegAudioExtractionErrorCode.INVALID_INPUT
            )

    @staticmethod
    def _has_valid_mp4_container(input_video: Path) -> bool:
        with input_video.open("rb") as media:
            media.seek(0, os.SEEK_END)
            file_size = media.tell()
            offset = 0
            found_file_type = False

            while offset < file_size:
                if file_size - offset < 8:
                    return False
                media.seek(offset)
                header = media.read(8)
                if len(header) != 8:
                    return False
                box_size, box_type = struct.unpack(">I4s", header)
                header_size = 8
                if box_size == 1:
                    extended_size = media.read(8)
                    if len(extended_size) != 8:
                        return False
                    box_size = struct.unpack(">Q", extended_size)[0]
                    header_size = 16
                elif box_size == 0:
                    box_size = file_size - offset

                if (
                    box_size < header_size
                    or box_size > file_size - offset
                ):
                    return False

                if box_type == b"ftyp":
                    if box_size < 16 or (box_size - 16) % 4:
                        return False
                    found_file_type = True

                offset += box_size

            return found_file_type

    def _find_executable(self) -> str:
        try:
            executable = shutil.which(self._executable)
        except OSError:
            executable = None
        if executable is None:
            raise FFmpegAudioExtractionError(
                FFmpegAudioExtractionErrorCode.FFMPEG_UNAVAILABLE
            )
        return executable

    @staticmethod
    def _validate_output(output_audio: Path) -> None:
        if not isinstance(output_audio, Path):
            raise FFmpegAudioExtractionError(
                FFmpegAudioExtractionErrorCode.INVALID_OUTPUT
            )
        try:
            valid_path = (
                output_audio.suffix.lower() == ".wav"
                and output_audio.parent.is_dir()
            )
            destination_exists = os.path.lexists(output_audio)
        except OSError:
            raise FFmpegAudioExtractionError(
                FFmpegAudioExtractionErrorCode.INVALID_OUTPUT
            ) from None
        if not valid_path:
            raise FFmpegAudioExtractionError(
                FFmpegAudioExtractionErrorCode.INVALID_OUTPUT
            )
        if destination_exists:
            raise FFmpegAudioExtractionError(
                FFmpegAudioExtractionErrorCode.OUTPUT_ALREADY_EXISTS
            )

    @staticmethod
    def _validate_extracted_audio(temporary_audio: Path) -> None:
        try:
            if not temporary_audio.is_file() or temporary_audio.stat().st_size == 0:
                raise FFmpegAudioExtractionError(
                    FFmpegAudioExtractionErrorCode.OUTPUT_MISSING
                )
            with wave.open(str(temporary_audio), "rb") as extracted_audio:
                channel_count = extracted_audio.getnchannels()
                sample_width = extracted_audio.getsampwidth()
                sample_rate = extracted_audio.getframerate()
                frame_count = extracted_audio.getnframes()
                if (
                    extracted_audio.getcomptype() != "NONE"
                    or channel_count <= 0
                    or sample_width != 2
                    or sample_rate <= 0
                    or frame_count <= 0
                ):
                    raise FFmpegAudioExtractionError(
                        FFmpegAudioExtractionErrorCode.OUTPUT_VERIFICATION_FAILED
                    )
                frame_size = channel_count * sample_width
                frames_remaining = frame_count
                frames_per_read = max(1, 65_536 // frame_size)
                while frames_remaining:
                    frames_to_read = min(frames_remaining, frames_per_read)
                    frame_data = extracted_audio.readframes(frames_to_read)
                    if len(frame_data) != frames_to_read * frame_size:
                        raise FFmpegAudioExtractionError(
                            FFmpegAudioExtractionErrorCode.OUTPUT_VERIFICATION_FAILED
                        )
                    frames_remaining -= frames_to_read
                if extracted_audio.readframes(1):
                    raise FFmpegAudioExtractionError(
                        FFmpegAudioExtractionErrorCode.OUTPUT_VERIFICATION_FAILED
                    )
        except FFmpegAudioExtractionError:
            raise
        except (OSError, EOFError, wave.Error):
            raise FFmpegAudioExtractionError(
                FFmpegAudioExtractionErrorCode.OUTPUT_VERIFICATION_FAILED
            ) from None

    @staticmethod
    def _publish_without_overwrite(
        temporary_audio: Path,
        output_audio: Path,
    ) -> None:
        try:
            os.link(temporary_audio, output_audio)
        except FileExistsError:
            raise FFmpegAudioExtractionError(
                FFmpegAudioExtractionErrorCode.OUTPUT_ALREADY_EXISTS
            ) from None
        except OSError:
            raise FFmpegAudioExtractionError(
                FFmpegAudioExtractionErrorCode.OUTPUT_PUBLISH_FAILED
            ) from None

    @staticmethod
    def _remove_temporary_directory(temporary_directory: Path) -> None:
        try:
            shutil.rmtree(temporary_directory)
        except OSError:
            raise FFmpegAudioExtractionError(
                FFmpegAudioExtractionErrorCode.CLEANUP_FAILED
            ) from None

# Transcription

## Status

The typed MP3/WAV transcription contracts and a lazy Faster-Whisper adapter are implemented. The adapter loads only from a local model directory, uses configured CPU INT8 settings, and does not download model files during processing. The pinned base model has been provisioned with required-file and SHA-256 verification. Audio is decoded to enforce the configured maximum duration (300 seconds by default) before inference. Real CPU INT8 transcription through the adapter has been verified on two distinct local MP3 files; observations are recorded below and are not performance guarantees. MP3/WAV transcription can be injected into `LocalDocumentProcessor`, but the background worker is not integrated. MP4/video extraction and FFmpeg integration remain unimplemented.

The default model directory is `<repository-root>/models/faster-whisper/base` and can be overridden with `TRANSCRIPTION_MODEL_PATH`. Configure `TRANSCRIPTION_CPU_THREADS` to a positive integer; the default is `2`. `TRANSCRIPTION_COMPUTE_TYPE` defaults to `int8` and accepts CTranslate2 compute types. `TRANSCRIPTION_MAX_AUDIO_SECONDS` defaults to `300` and must be positive. The adapter uses PyAV (installed as a faster-whisper dependency) to decode the audio and sum decoded sample durations before loading the ASR model; it stops decoding as soon as the configured limit is exceeded. This duration cap is independent of the existing 25 MiB upload-size limit. Invalid or undecodable audio fails safely. Model loading fails safely with `model_unavailable` when the directory or required files are missing. A missing runtime or decoder dependency fails with `runtime_unavailable`. Document processing never downloads or provisions model files.

## Local runtime comparison

The selected runtime for this initial local evaluation is **faster-whisper**. It provides a Python API that returns segment text with start/end times, supports CPU INT8 and NVIDIA GPU execution, and CTranslate2 publishes Windows x86-64 wheels. Its documentation says audio decoding uses PyAV, so MP3/WAV transcription does not require a separate FFmpeg executable. GPU execution requires separately installed NVIDIA libraries; only CPU execution was used for the validation below.

| Runtime | CPU and RAM | GPU | Models and downloads | License and timestamps | Windows and integration |
|---|---|---|---|---|---|
| `faster-whisper` | CPU INT8 supported. Its README reports 1,477 MB RAM for its small-model INT8 benchmark on an Intel i7-12700K; this is a runtime-specific published result, not a Cognify benchmark or a guarantee. | NVIDIA CUDA supported; CTranslate2 documents CUDA/cuDNN requirements. | Converted Whisper model files are fetched/cached when selected. Size depends on model and quantization; compare base/small during evaluation. | faster-whisper and the cited converted small model card declare MIT. API segments include start/end seconds. | Lowest integration effort of these options: direct Python interface and Windows wheels. GPU library setup remains a separate concern. |
| `openai-whisper` | CPU inference is supported but the original PyTorch stack is comparatively heavy; its README publishes model GPU-VRAM guidance rather than a directly comparable CPU-RAM table. | PyTorch CUDA; documented approximate VRAM ranges from about 1 GB for tiny/base through 10 GB for large, depending on model. | Model weights download when first loaded. The project lists model parameter sizes from 39M (tiny) to 1,550M (large). | Code and model weights are MIT. The Python API exposes timestamped segments. | Python API is direct, but PyTorch installation and system FFmpeg are required. On Windows, its docs note possible Rust setup if a tiktoken wheel is unavailable. |
| `whisper.cpp` | CPU-only operation and quantized models are supported. Its README reports approximate memory/disk sizes: tiny 273 MB/75 MiB, base 388 MB/142 MiB, small 852 MB/466 MiB, medium 2.1 GB/1.5 GiB, and large 3.9 GB/2.9 GiB. | Supports multiple backends including NVIDIA; hardware-specific build/configuration applies. | Download a model converted to GGML format; quantization can further reduce disk and memory. | Runtime is MIT; timestamped segments are available through its API/CLI output. Verify the selected model artifact’s license as well. | Windows MSVC is supported, but a CMake/C++ build or maintained binding is needed. Its CLI audio path documents WAV input; conversion of MP3/MP4 may require FFmpeg. |

These figures come from different upstream test setups and are not directly comparable. The base model on CPU INT8 has now been exercised on the confirmed Windows development machine (Ryzen 5 5600H, 7.36 GB RAM), with one local sequential validation recorded below. This is not a quality evaluation or evidence that the configuration is suitable for broader use; low available memory was observed. Small remains only a possible comparison candidate if memory headroom permits. See the upstream [faster-whisper README](https://github.com/SYSTRAN/faster-whisper), [CTranslate2 installation guide](https://opennmt.net/CTranslate2/installation.html), [CTranslate2 quantization guide](https://opennmt.net/CTranslate2/quantization.html), [OpenAI Whisper README and model card](https://github.com/openai/whisper), and [whisper.cpp README](https://github.com/ggml-org/whisper.cpp).

## Contract

`Transcriber.transcribe(TranscriptionRequest) -> TranscriptionResult` is the replaceable runtime boundary. A request identifies an audio `Path` and MP3/WAV type. The faster-whisper adapter uses `device="cpu"`, the configured compute type and CPU thread count, and `local_files_only=True`. It requires `config.json`, `model.bin`, `tokenizer.json`, and `vocabulary.txt` in the configured model directory. MP4 is not accepted by this boundary; a future FFmpeg stage must extract audio before invoking a transcriber.

Results contain an immutable, non-empty tuple of `TranscriptionSegment` values. Each segment has non-blank text and start/end timestamps in seconds relative to the original media timeline. Timestamps must be finite, non-negative, and `start_seconds <= end_seconds`. The result preserves source order and rejects segments whose start times move backwards; overlapping segments are allowed. The boundary does not sort, merge, or otherwise rewrite runtime output.

`TranscriptionError.code` is a stable safe category: `unsupported_media`, `invalid_transcription` (including empty or undecodable audio), `runtime_unavailable`, `model_unavailable`, `timeout`, or `malformed_output`. Exception messages contain only the category, not paths, media contents, or provider details. Timeout remains reserved; the synchronous adapter does not claim hard timeout or memory-limit enforcement.

The processor maps validated transcription segments to `ProcessedSegment` while preserving text, timestamps, and order. The worker factory still uses the PDF/TXT processor without a transcriber, so audio job processing is not enabled end-to-end.

The worker singleton processes one queued job at a time, but the transcriber itself does not serialize concurrent calls. One-at-a-time transcription remains pending until worker integration uses that singleton path; no independent concurrency guarantee is made.

The faster-whisper, CTranslate2, and tested PyAV versions are pinned in `backend/requirements.txt`. Other ASR transitive dependencies are not locked. A generated, reviewed constraints file for the supported Windows/Python 3.12 environment remains a reproducibility follow-up; transitive versions are not guessed here.

The locally provisioned base model is `Systran/faster-whisper-base` at revision `ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66`. Its `model.bin` SHA-256 is `d01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9`; the expected hash was verified against the provisioned file. The successful smoke run verifies model loading and transcription on the tested machine, not transcription quality or suitability for other environments. Review redistribution terms before distributing the converted weights.

## Manual model provisioning

After installing the already-pinned backend requirements, run the following command from the repository root:

```powershell
backend\.venv\Scripts\python.exe -m scripts.provision_faster_whisper_model
```

The script downloads only the four runtime-required files from the pinned repository revision into a temporary sibling directory. It verifies all four files and the `model.bin` SHA-256 before moving the directory to `models/faster-whisper/base`. If the destination already exists, or if download or integrity checks fail, it stops without overwriting the destination. A failed staging directory is removed. The model directory is ignored by Git. Provisioning is an explicit operator action and is never triggered by application startup, upload, or processing.

## Local CPU INT8 smoke evaluation

On 2026-11-04, the approved base model at revision `ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66` was provisioned; all four required files were present and `model.bin` matched SHA-256 `d01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9`. With Faster-Whisper 1.2.1, CTranslate2 4.8.5, and PyAV 16.1.0, local CPU INT8 model loading and transcription succeeded. In a later sequential validation using two distinct MP3s and one adapter instance, the 71.851-second, 1,054,601-byte first file produced 10 segments in 6.976 seconds; the 46.656-second, 747,209-byte second file produced 6 segments in 3.626 seconds. Both outputs had nonblank text and valid, ordered timestamps; their transcription content differed. Peak process working set was 343.8 MiB. Available physical memory was 0.83 GiB before and 0.75 GiB after the two calls. These are observations from one local validation run, not performance guarantees. The transcript is not recorded here because the samples may contain personal speech.

The timings and memory figures above are single-run observations, not controlled benchmarks, resource guarantees, or estimates of steady-state memory use. Low available system memory was observed. The adapter does not guarantee a hard transcription timeout, memory limit, or concurrency safety. Worker integration and MP4/video extraction remain pending; this validation does not establish production readiness or transcription quality.

## Test evidence

The following checks were run during Day 34 implementation and validation; they were not rerun as part of this documentation update:

- Full backend suite: 533 passed, with one third-party deprecation warning.
- Focused transcription contract and adapter tests: 37 passed.
- `pip check`: passed with no broken requirements.
- `git diff --check`: passed.

## Testing

Contract tests use a deterministic fake and require no model, downloads, FFmpeg, Gemini, Pinecone, or network access:

```powershell
.\backend\.venv\Scripts\python.exe -m pytest backend\tests\test_transcription_contracts.py
```

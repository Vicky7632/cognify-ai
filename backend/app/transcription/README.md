# Transcription

## Status

The provider-neutral contracts in `contracts.py` are implemented and covered by deterministic fake-based tests. No ASR runtime, model, FFmpeg integration, or worker wiring is implemented. No runtime dependency has been installed and no model has been downloaded.

## Local runtime comparison

The recommendation is **faster-whisper** for the first runtime evaluation. It provides a Python API that returns segment text with start/end times, supports CPU INT8 and NVIDIA GPU execution, and CTranslate2 publishes Windows x86-64 wheels. Its documentation says audio decoding uses PyAV, so MP3/WAV transcription does not require a separate FFmpeg executable. GPU execution requires separately installed NVIDIA libraries; CPU quantized inference is the practical initial path to evaluate on developer machines.

| Runtime | CPU and RAM | GPU | Models and downloads | License and timestamps | Windows and integration |
|---|---|---|---|---|---|
| `faster-whisper` | CPU INT8 supported. Its README reports 1,477 MB RAM for its small-model INT8 benchmark on an Intel i7-12700K; this is a runtime-specific published result, not a Cognify benchmark or a guarantee. | NVIDIA CUDA supported; CTranslate2 documents CUDA/cuDNN requirements. | Converted Whisper model files are fetched/cached when selected. Size depends on model and quantization; compare base/small during evaluation. | faster-whisper and the cited converted small model card declare MIT. API segments include start/end seconds. | Lowest integration effort of these options: direct Python interface and Windows wheels. GPU library setup remains a separate concern. |
| `openai-whisper` | CPU inference is supported but the original PyTorch stack is comparatively heavy; its README publishes model GPU-VRAM guidance rather than a directly comparable CPU-RAM table. | PyTorch CUDA; documented approximate VRAM ranges from about 1 GB for tiny/base through 10 GB for large, depending on model. | Model weights download when first loaded. The project lists model parameter sizes from 39M (tiny) to 1,550M (large). | Code and model weights are MIT. The Python API exposes timestamped segments. | Python API is direct, but PyTorch installation and system FFmpeg are required. On Windows, its docs note possible Rust setup if a tiktoken wheel is unavailable. |
| `whisper.cpp` | CPU-only operation and quantized models are supported. Its README reports approximate memory/disk sizes: tiny 273 MB/75 MiB, base 388 MB/142 MiB, small 852 MB/466 MiB, medium 2.1 GB/1.5 GiB, and large 3.9 GB/2.9 GiB. | Supports multiple backends including NVIDIA; hardware-specific build/configuration applies. | Download a model converted to GGML format; quantization can further reduce disk and memory. | Runtime is MIT; timestamped segments are available through its API/CLI output. Verify the selected model artifact’s license as well. | Windows MSVC is supported, but a CMake/C++ build or maintained binding is needed. Its CLI audio path documents WAV input; conversion of MP3/MP4 may require FFmpeg. |

These figures come from different upstream test setups and are not directly comparable. Before adopting a model, benchmark representative Cognify audio for transcription quality, peak RAM, latency, license/provenance, and cold-start/model-download behavior. The exact model size remains undecided. See the upstream [faster-whisper README](https://github.com/SYSTRAN/faster-whisper), [CTranslate2 installation guide](https://opennmt.net/CTranslate2/installation.html), [CTranslate2 quantization guide](https://opennmt.net/CTranslate2/quantization.html), [OpenAI Whisper README and model card](https://github.com/openai/whisper), and [whisper.cpp README](https://github.com/ggml-org/whisper.cpp).

## Contract

`Transcriber.transcribe(TranscriptionRequest) -> TranscriptionResult` is the replaceable runtime boundary. A request identifies an audio `Path` and MP3/WAV type. MP4 is not accepted by this boundary; a future FFmpeg stage must extract audio before invoking a transcriber.

Results contain an immutable, non-empty tuple of `TranscriptionSegment` values. Each segment has non-blank text and start/end timestamps in seconds relative to the original media timeline. Timestamps must be finite, non-negative, and `start_seconds <= end_seconds`. The result preserves source order and rejects segments whose start times move backwards; overlapping segments are allowed. The boundary does not sort, merge, or otherwise rewrite runtime output.

`TranscriptionError.code` is a stable safe category: `unsupported_media`, `invalid_transcription` (including empty output), `runtime_unavailable`, `timeout`, or `malformed_output`. Exception messages contain only the category, not paths, media contents, or provider details. Runtime-unavailable and timeout categories are defined for future adapters; no runtime currently raises them.

The contracts are independent of `ProcessedSegment` and are not yet wired into document processing. A later integration push should explicitly map validated transcription segments to the existing timestamp-capable processing result and preserve its order.

## Testing

Contract tests use a deterministic fake and require no model, downloads, FFmpeg, Gemini, Pinecone, or network access:

```powershell
.\backend\.venv\Scripts\python.exe -m pytest backend\tests\test_transcription_contracts.py
```

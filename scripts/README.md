# Scripts

## Faster-Whisper model provisioning

`provision_faster_whisper_model.py` manually downloads only the approved `Systran/faster-whisper-base` revision and required runtime files. It stages the files, validates the manifest and pinned `model.bin` SHA-256, refuses an existing destination, and then installs to `models/faster-whisper/base`.

Run it from the repository root after installing the backend requirements:

```powershell
backend\.venv\Scripts\python.exe -m scripts.provision_faster_whisper_model
```

The script is not called by application startup, document upload, or processing. It has no destination override and does not replace an existing model directory.

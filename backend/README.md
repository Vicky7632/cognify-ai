# Backend

The backend is planned as a standalone Python/FastAPI service using Pydantic Settings, SQLAlchemy, and PostgreSQL. Python **3.12.x** is the project target; Python 3.12.10 was available when this setup was verified. The four direct runtime/test dependencies and their resolved transitive dependencies are pinned in [`requirements.txt`](requirements.txt) for repeatable pip installs. Pip manages a backend-local virtual environment; no additional dependency manager is required.

No database schema or migrations are implemented yet. PostgreSQL will hold relational application metadata; Pinecone will hold vectors and searchable metadata; original files use a separate storage abstraction, initially local filesystem. Do not create database tables or migrations until their roadmap phase.

## Windows PowerShell setup

Run from the repository root:

```powershell
cd E:\Cognify\backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation under the current execution policy, use the environment's interpreter directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

To verify the installed direct dependencies:

```powershell
python -c "import fastapi, pydantic_settings, pytest, uvicorn; print('FastAPI', fastapi.__version__); print('Pydantic Settings', pydantic_settings.__version__); print('pytest', pytest.__version__); print('Uvicorn', uvicorn.__version__)"
```

Run backend tests from `E:\Cognify\backend`:

```powershell
python -m pytest
```

Start the development server from `E:\Cognify\backend`:

```powershell
python -m uvicorn app.main:app --reload
```

The health check is available at `http://127.0.0.1:8000/api/v1/health`; FastAPI's interactive API documentation remains available at `http://127.0.0.1:8000/docs`.

The current tests cover settings behavior and the health endpoint. Extend them as features are implemented.

## Logging

The application uses Python's standard logging module for fixed startup and shutdown lifecycle messages. It does not log configuration values, request bodies, uploaded content, credentials, or API keys. Configure logging output through the server/runtime environment; no additional logging package is required.

## Configuration and safety

The minimal settings module reads `COGNIFY_APP_NAME` and `COGNIFY_API_V1_PREFIX` from process environment variables or the repository-root `.env`; environment variables take precedence. Names are case-insensitive, and unrelated variables are ignored. Defaults are `Cognify AI` and `/api/v1`; configured API prefixes must start with `/` and must not end with `/`. The repository-root path is resolved from the settings module, so loading does not depend on the current working directory. Runtime settings are cached once.

The root `.env.example` exists and contains placeholder values for planned settings; it is not loaded automatically. For a temporary PowerShell override, set `$env:COGNIFY_APP_NAME` or `$env:COGNIFY_API_V1_PREFIX` before starting the server; the override applies to that PowerShell session and its child processes. Alternatively, add only non-secret settings to the repository-root `.env`. Do not commit `.env`, print credentials, or put backend secrets in frontend variables. The root `.gitignore` excludes `.env` and `.venv` while allowing `.env.example` to be tracked. PostgreSQL, Pinecone, and LLM credentials are not required to import settings, start the health endpoint, or run tests.

The initial MVP is unauthenticated and must not be exposed publicly without suitable access controls. No PostgreSQL/Pinecone integration, paid service, or migrations are required to run the health endpoint.

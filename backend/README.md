# Backend

The backend is a standalone Python/FastAPI service using Pydantic Settings, SQLAlchemy, and PostgreSQL. Python **3.12.x** is the project target; Python 3.12.10 was available when this setup was verified. Direct runtime/test dependencies and their resolved transitive dependencies are pinned in [`requirements.txt`](requirements.txt) for repeatable pip installs. Pip manages a backend-local virtual environment; no additional dependency manager is required.

The backend now has a lazy PostgreSQL engine and SQLAlchemy session dependency foundation, but no relational models, tables, or migrations. PostgreSQL will hold relational application metadata; Pinecone will hold vectors and searchable metadata; original files use a separate storage abstraction, initially local filesystem.

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

The current tests cover settings behavior, lazy database/session behavior, lifecycle logging, and the health endpoint. Extend them as features are implemented.

## Logging

The application uses Python's standard logging module for fixed startup and shutdown lifecycle messages. It does not log configuration values, request bodies, uploaded content, credentials, or API keys. Configure logging output through the server/runtime environment; no additional logging package is required.

## Configuration and safety

The minimal settings module reads `COGNIFY_APP_NAME`, `COGNIFY_API_V1_PREFIX`, and optional `DATABASE_URL` from process environment variables or the repository-root `.env`; environment variables take precedence. Names are case-insensitive, and unrelated variables are ignored. Defaults are `Cognify AI`, `/api/v1`, and no database URL. Configured API prefixes must start with `/` and must not end with `/`. The repository-root path is resolved from the settings module, so loading does not depend on the current working directory. Runtime settings are cached once.

`DATABASE_URL` is optional for application startup and the health endpoint. When a route later requests a database session, the cached SQLAlchemy engine is created lazily from `DATABASE_URL`; if it is absent, the database dependency raises an explicit configuration error. Session cleanup is handled by the `get_db` dependency. The engine uses `pool_pre_ping`; no connection is opened merely by importing the app or constructing the engine.

The root `.env.example` contains placeholder values, including a sample local PostgreSQL URL, and is not loaded automatically. For a temporary PowerShell override, set `$env:COGNIFY_APP_NAME`, `$env:COGNIFY_API_V1_PREFIX`, or `$env:DATABASE_URL` before starting the server; the override applies to that PowerShell session and its child processes. Alternatively, add local settings to the repository-root `.env`. Do not commit `.env`, print credentials, or put backend secrets in frontend variables. The root `.gitignore` excludes `.env` and `.venv` while allowing `.env.example` to be tracked. Pinecone and LLM credentials are not needed to import settings, start the health endpoint, or run tests.

The current tests mock the engine/session boundary and do not require or connect to a PostgreSQL server. A live database connection has not been verified.

The initial MVP is unauthenticated and must not be exposed publicly without suitable access controls. No live PostgreSQL connection, Pinecone service, paid service, or migrations are required to run the health endpoint.

## Week 1 milestone review

### Implemented

- Python 3.12 backend tooling with pinned FastAPI, Uvicorn, Pydantic Settings, pytest, and TestClient dependencies.
- FastAPI application factory with a configurable application title and versioned API prefix.
- `GET /api/v1/health`, returning HTTP 200 and exactly `{"status":"ok"}`; default FastAPI `/docs` remains enabled.
- Pydantic Settings defaults, case-insensitive environment loading, unrelated-variable ignoring, repository-root `.env` path resolution, and cached runtime settings.
- API prefix validation and fixed startup/shutdown lifecycle logs that do not include configuration or request data.
- Automated tests for settings defaults/overrides/dotenv loading, valid and invalid prefixes, health behavior, and lifecycle logging.

### Not implemented

At the Week 1 review, PostgreSQL connection/session management, SQLAlchemy models, migrations, document persistence, file upload or extraction, Pinecone, embeddings, retrieval, generation, transcription, frontend screens, authentication, and deployment infrastructure were not yet implemented.

### Next milestone

The planned PostgreSQL connection/session foundation is now in place. Next, define and test the first document-related relational model and decide the migration workflow. Keep that work separate from Pinecone vector storage and original-file storage; no model or migration is included yet.

# Backend

The backend is planned as a standalone Python/FastAPI service using Pydantic Settings, SQLAlchemy, and PostgreSQL. Python **3.12.x** is the project target; Python 3.12.10 was available when this setup was verified. The four direct runtime/test dependencies and their resolved transitive dependencies are pinned in [`requirements.txt`](requirements.txt) for repeatable pip installs. Pip manages a backend-local virtual environment; no additional dependency manager is required.

No application code, routes, database schema, migrations, or tests are implemented yet. PostgreSQL will hold relational application metadata; Pinecone will hold vectors and searchable metadata; original files use a separate storage abstraction, initially local filesystem. Do not create database tables or migrations until their roadmap phase.

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

When tests have been added, run them from `E:\Cognify\backend`:

```powershell
python -m pytest
```

There are currently no backend test cases, so pytest may report that no tests were collected until the backend skeleton phase adds tests.

## Configuration and safety

Application settings are not implemented yet. The root `.env.example` exists and contains placeholder values for planned settings; it is a template, not a validated runtime configuration. Keep real credentials in an untracked `.env` or an appropriate secret store, never in source control or frontend variables. The root `.gitignore` excludes `.env` and `.venv` while allowing `.env.example` to be tracked.

The initial MVP is unauthenticated and must not be exposed publicly without suitable access controls. No PostgreSQL/Pinecone integration, paid service, application behavior, or migrations are required to create the local Python environment.

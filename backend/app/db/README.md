# Database

`base.py` defines the shared SQLAlchemy declarative base. Models imported through `app.models` register their tables with `Base.metadata` without creating an engine or connecting to PostgreSQL.

`Document` in `app/models/document.py` describes the `documents` table with a UUID primary key, original filename, supported file type, local storage path, processing status, and timezone-aware creation/update timestamps. Supported initial file types are PDF, TXT, MP3, WAV, and MP4. Processing statuses are `pending`, `processing`, `completed`, and `failed`; new records default to `pending`.

This is an ORM model definition only. No table has been created, and this phase does not add migrations or run schema DDL. Table creation and migration workflow remain future work.

`session.py` provides lazy PostgreSQL engine creation, a cached SQLAlchemy session factory, and the `get_db` FastAPI dependency. Configure the optional `DATABASE_URL` before a route requests a database session. Importing the application and using the health endpoint do not create an engine or require a database URL.

The yielded session is closed after request handling. Transaction commit/rollback policy belongs to the future service-of-work layer.

# Backend

The planned backend is a standalone Python/FastAPI service with Pydantic settings, SQLAlchemy, and PostgreSQL. HTTPX may be used where outbound HTTP calls are needed. No application code, routes, dependency manifest, database schema, or migrations exist yet.

PostgreSQL is intended for relational document/processing/chat metadata; Pinecone is for vector embeddings and searchable metadata; original files use a separate storage abstraction (local filesystem is acceptable initially). Do not create database tables or migrations until the relevant roadmap phase.

The initial MVP is unauthenticated and must not be exposed publicly without appropriate access controls. Setup instructions will follow once Python version and package tooling are selected.

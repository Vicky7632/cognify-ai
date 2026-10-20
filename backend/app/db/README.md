# Database

`session.py` provides lazy PostgreSQL engine creation, a cached SQLAlchemy session factory, and the `get_db` FastAPI dependency. Configure the optional `DATABASE_URL` before a route requests a database session. Importing the application and using the health endpoint do not create an engine or require a database URL.

The yielded session is closed after request handling. Transaction commit/rollback policy belongs to the future service-of-work layer. No relational models, tables, or migrations are defined in this phase.

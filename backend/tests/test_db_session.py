from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db import session as db_session


@pytest.fixture(autouse=True)
def clear_database_caches() -> Generator[None, None, None]:
    db_session.get_engine.cache_clear()
    db_session.get_session_factory.cache_clear()
    yield
    db_session.get_session_factory.cache_clear()
    db_session.get_engine.cache_clear()


def test_engine_is_created_lazily_and_cached(monkeypatch) -> None:
    settings = Settings(
        database_url="postgresql+psycopg://test:test@localhost/test_db",
        _env_file=None,
    )
    create_engine = MagicMock()
    engine = MagicMock(spec=Engine)
    create_engine.return_value = engine
    monkeypatch.setattr(db_session, "get_settings", lambda: settings)
    monkeypatch.setattr(db_session, "create_engine", create_engine)

    first_engine = db_session.get_engine()
    second_engine = db_session.get_engine()

    assert first_engine is engine
    assert second_engine is engine
    create_engine.assert_called_once_with(
        "postgresql+psycopg://test:test@localhost/test_db",
        pool_pre_ping=True,
    )
    engine.connect.assert_not_called()


def test_engine_requires_database_url_only_when_requested(monkeypatch) -> None:
    settings = Settings(_env_file=None)
    create_engine = MagicMock()
    monkeypatch.setattr(db_session, "get_settings", lambda: settings)
    monkeypatch.setattr(db_session, "create_engine", create_engine)

    with pytest.raises(RuntimeError, match="DATABASE_URL must be configured"):
        db_session.get_engine()

    create_engine.assert_not_called()


def test_database_dependency_closes_session(monkeypatch) -> None:
    session = MagicMock(spec=Session)
    factory = MagicMock(return_value=session)
    monkeypatch.setattr(db_session, "get_session_factory", lambda: factory)

    dependency = db_session.get_db()
    yielded_session = next(dependency)
    dependency.close()

    assert yielded_session is session
    session.close.assert_called_once_with()

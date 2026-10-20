from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.application import create_app
from app.core.config import Settings
from app.db import session as db_session


def test_health_returns_ok_without_creating_database_engine(monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    create_engine = MagicMock(wraps=db_session.create_engine)
    monkeypatch.setattr(db_session, "create_engine", create_engine)
    app = create_app(Settings(_env_file=None))
    client = TestClient(app)

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    create_engine.assert_not_called()

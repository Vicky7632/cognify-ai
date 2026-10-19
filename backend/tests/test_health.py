from fastapi.testclient import TestClient

from app.application import create_app
from app.core.config import Settings


def test_health_returns_ok() -> None:
    app = create_app(Settings(_env_file=None))
    client = TestClient(app)

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

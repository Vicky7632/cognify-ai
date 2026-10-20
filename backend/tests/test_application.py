import logging

from fastapi.testclient import TestClient

from app.application import create_app
from app.core.config import Settings


def test_app_configuration_and_lifecycle_logging(caplog) -> None:
    app_name = "Private Test Application Name"
    api_prefix = "/private-test-prefix"
    app = create_app(
        Settings(
            app_name=app_name,
            api_v1_prefix=api_prefix,
            _env_file=None,
        )
    )

    assert app.title == app_name

    with caplog.at_level(logging.INFO, logger="app.application"):
        with TestClient(app) as client:
            response = client.get(f"{api_prefix}/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}

    messages = [record.getMessage() for record in caplog.records]
    assert "Application startup complete" in messages
    assert "Application shutdown complete" in messages
    assert all(app_name not in message for message in messages)
    assert all(api_prefix not in message for message in messages)

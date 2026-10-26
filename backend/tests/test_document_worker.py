from __future__ import annotations

from collections.abc import Callable
from threading import enumerate as enumerate_threads

import pytest

from app import document_worker
from app.core.config import Settings
from app.services.document_indexing_service import DocumentWorkerLockLostError


class FakeTransaction:
    def __enter__(self) -> FakeTransaction:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class FakeSession:
    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def begin(self) -> FakeTransaction:
        return FakeTransaction()


class FakeLockConnection:
    def __init__(
        self,
        *,
        acquire: bool = True,
        valid: bool = True,
        fail_acquire: bool = False,
        fail_backend_pid: bool = False,
        fail_unlock: bool = False,
        fail_invalidate: bool = False,
    ) -> None:
        self.acquire = acquire
        self.is_valid = valid
        self.fail_acquire = fail_acquire
        self.fail_backend_pid = fail_backend_pid
        self.fail_unlock = fail_unlock
        self.fail_invalidate = fail_invalidate
        self.closed = False
        self.exited = False
        self.unlock_attempts = 0
        self.invalidated = False
        self.detach_attempts = 0
        self.close_attempts = 0

    @property
    def connection(self) -> FakeLockConnection:
        return self

    @property
    def driver_connection(self) -> FakeLockConnection:
        return self

    def __enter__(self) -> FakeLockConnection:
        return self

    def __exit__(self, *_args: object) -> None:
        self.exited = True
        self.closed = True

    def begin(self) -> FakeTransaction:
        return FakeTransaction()

    def scalar(
        self,
        statement: object,
        _parameters: dict[str, int] | None = None,
    ) -> bool | int:
        sql = str(statement)
        if "pg_try_advisory_lock" in sql:
            if self.fail_acquire:
                raise RuntimeError("synthetic acquisition response failure")
            return self.acquire
        if "pg_backend_pid" in sql:
            if self.fail_backend_pid:
                raise RuntimeError("synthetic backend PID lookup failure")
            return 42
        if "pg_advisory_unlock" in sql:
            self.unlock_attempts += 1
            if self.fail_unlock:
                raise RuntimeError("synthetic connection failure")
            return True
        raise AssertionError(f"Unexpected SQL statement: {sql}")

    def invalidate(self) -> None:
        if self.fail_invalidate:
            raise RuntimeError("synthetic invalidation failure")
        self.invalidated = True
        self.is_valid = False

    def detach(self) -> None:
        self.detach_attempts += 1

    def close(self) -> None:
        self.close_attempts += 1
        self.closed = True


class FakeEngine:
    def __init__(self, connection: FakeLockConnection) -> None:
        self.connection = connection

    def connect(self) -> FakeLockConnection:
        return self.connection


def patch_startup_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        document_worker.document_processing_job_repository,
        "recover_interrupted_document_processing_jobs",
        lambda _session: 0,
    )
    monkeypatch.setattr(
        document_worker.document_processing_job_repository,
        "enqueue_unqueued_uploaded_documents",
        lambda _session: 0,
    )
    monkeypatch.setattr(
        document_worker,
        "get_settings",
        lambda: Settings(_env_file=None),
    )


def test_worker_closes_lock_connection_when_acquisition_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = FakeLockConnection(acquire=False)
    monkeypatch.setattr(
        document_worker,
        "_advisory_lock_is_held",
        lambda _engine, _pid: pytest.fail("ownership must not be checked"),
    )

    with pytest.raises(RuntimeError, match="Another document processing worker"):
        document_worker.run_worker(FakeEngine(connection), lambda: FakeSession())

    assert connection.unlock_attempts == 0
    assert connection.exited is True
    assert connection.closed is True


def test_worker_invalidates_connection_when_lock_acquisition_is_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = FakeLockConnection(fail_acquire=True)

    with pytest.raises(RuntimeError, match="synthetic acquisition response failure"):
        document_worker.run_worker(FakeEngine(connection), lambda: FakeSession())

    assert connection.invalidated is True
    assert connection.unlock_attempts == 0
    assert connection.exited is True


def test_worker_invalidates_connection_when_backend_pid_lookup_fails() -> None:
    connection = FakeLockConnection(fail_backend_pid=True)

    with pytest.raises(RuntimeError, match="synthetic backend PID lookup failure"):
        document_worker.run_worker(FakeEngine(connection), lambda: FakeSession())

    assert connection.invalidated is True
    assert connection.unlock_attempts == 0
    assert connection.exited is True


def test_worker_discards_connection_when_invalidation_fails() -> None:
    connection = FakeLockConnection(
        fail_acquire=True,
        fail_invalidate=True,
    )

    with pytest.raises(RuntimeError, match="synthetic acquisition response failure"):
        document_worker.run_worker(FakeEngine(connection), lambda: FakeSession())

    assert connection.detach_attempts == 1
    assert connection.close_attempts == 1
    assert connection.unlock_attempts == 0
    assert connection.exited is True


def test_worker_stops_without_claiming_after_lock_connection_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_startup_dependencies(monkeypatch)
    connection = FakeLockConnection()
    claim_attempts = 0

    def lock_is_held(_engine: FakeEngine, _pid: int) -> bool:
        return True

    def process_next(
        _session_factory: Callable[[], FakeSession],
        _service_factory: Callable[..., object],
        *,
        ensure_worker_lock: Callable[[], None],
    ) -> bool:
        nonlocal claim_attempts
        connection.invalidated = True
        ensure_worker_lock()
        claim_attempts += 1
        return False

    monkeypatch.setattr(document_worker, "_advisory_lock_is_held", lock_is_held)
    monkeypatch.setattr(document_worker, "process_next_document_job", process_next)

    with pytest.raises(DocumentWorkerLockLostError):
        document_worker.run_worker(
            FakeEngine(connection),
            lambda: FakeSession(),
            stop_requested=lambda: False,
        )

    assert claim_attempts == 0
    assert connection.exited is True
    assert connection.closed is True
    assert not any(
        thread.name == "document-worker-lock-monitor"
        for thread in enumerate_threads()
    )


def test_worker_releases_lock_connection_on_graceful_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_startup_dependencies(monkeypatch)
    connection = FakeLockConnection()
    process_calls = 0
    monkeypatch.setattr(
        document_worker,
        "_advisory_lock_is_held",
        lambda _engine, _pid: True,
    )

    def process_next(
        _session_factory: Callable[[], FakeSession],
        _service_factory: Callable[..., object],
        *,
        ensure_worker_lock: Callable[[], None],
    ) -> bool:
        nonlocal process_calls
        ensure_worker_lock()
        process_calls += 1
        return False

    monkeypatch.setattr(document_worker, "process_next_document_job", process_next)

    document_worker.run_worker(
        FakeEngine(connection),
        lambda: FakeSession(),
        stop_requested=lambda: True,
    )

    assert process_calls == 0
    assert connection.unlock_attempts == 1
    assert connection.invalidated is False
    assert connection.exited is True
    assert connection.closed is True
    assert not any(
        thread.name == "document-worker-lock-monitor"
        for thread in enumerate_threads()
    )


def test_worker_invalidates_connection_when_advisory_unlock_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_startup_dependencies(monkeypatch)
    connection = FakeLockConnection(fail_unlock=True)
    monkeypatch.setattr(
        document_worker,
        "_advisory_lock_is_held",
        lambda _engine, _pid: True,
    )

    document_worker.run_worker(
        FakeEngine(connection),
        lambda: FakeSession(),
        stop_requested=lambda: True,
    )

    assert connection.unlock_attempts == 1
    assert connection.invalidated is True
    assert connection.exited is True

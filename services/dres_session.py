"""In-memory application sessions that keep DRES tokens out of browser JavaScript."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import secrets
from threading import RLock
from typing import Any


@dataclass(slots=True)
class DresSession:
    local_id: str
    dres_session_id: str
    user: dict[str, Any]
    evaluations: list[dict[str, Any]]
    selected_evaluation_id: str | None = None
    submitted_fingerprints: set[str] = field(default_factory=set)
    last_used: datetime = field(default_factory=lambda: datetime.now(UTC))


class DresSessionStore:
    def __init__(self, ttl_hours: int = 8) -> None:
        self._sessions: dict[str, DresSession] = {}
        self._lock = RLock()
        self._ttl = timedelta(hours=ttl_hours)

    def create(
        self,
        dres_session_id: str,
        user: dict[str, Any],
        evaluations: list[dict[str, Any]],
    ) -> DresSession:
        local_id = secrets.token_urlsafe(32)
        session = DresSession(local_id, dres_session_id, user, evaluations)
        with self._lock:
            self._sessions[local_id] = session
        return session

    def get(self, local_id: str | None) -> DresSession | None:
        if not local_id:
            return None
        now = datetime.now(UTC)
        with self._lock:
            session = self._sessions.get(local_id)
            if session is None:
                return None
            if now - session.last_used > self._ttl:
                self._sessions.pop(local_id, None)
                return None
            session.last_used = now
            return session

    def delete(self, local_id: str | None) -> DresSession | None:
        if not local_id:
            return None
        with self._lock:
            return self._sessions.pop(local_id, None)

    def select_evaluation(self, local_id: str, evaluation_id: str) -> DresSession | None:
        session = self.get(local_id)
        if session is None:
            return None
        with self._lock:
            session.selected_evaluation_id = evaluation_id
        return session

    def has_fingerprint(self, local_id: str, fingerprint: str) -> bool:
        session = self.get(local_id)
        return bool(session and fingerprint in session.submitted_fingerprints)

    def record_fingerprint(self, local_id: str, fingerprint: str) -> None:
        session = self.get(local_id)
        if session is None:
            return
        with self._lock:
            session.submitted_fingerprints.add(fingerprint)

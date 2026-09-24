"""Small asynchronous client for the DRES v2 participant API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx


@dataclass(slots=True)
class DresApiError(Exception):
    status_code: int
    message: str
    details: Any = None

    def __str__(self) -> str:
        return self.message


class DresClient:
    """Calls only the DRES endpoints required by the competition UI."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        normalized = base_url.strip().rstrip("/")
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("DRES_BASE_URL must be an absolute HTTP(S) URL")
        if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1"}:
            raise ValueError("Non-local DRES_BASE_URL must use HTTPS")
        if timeout_seconds <= 0:
            raise ValueError("DRES timeout must be greater than zero")

        self.base_url = normalized
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def _request(
        self,
        method: str,
        path: str,
        *,
        expected_statuses: set[int] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        expected = expected_statuses or {200}
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
                follow_redirects=False,
            ) as client:
                response = await client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise DresApiError(504, "DRES không phản hồi trong thời gian cho phép") from exc
        except httpx.RequestError as exc:
            raise DresApiError(502, "Không thể kết nối tới DRES") from exc

        if response.status_code not in expected:
            details = self._response_body(response)
            message = self._error_message(details) or f"DRES trả về HTTP {response.status_code}"
            raise DresApiError(response.status_code, message, details)
        return response

    @staticmethod
    def _response_body(response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return response.text.strip() or None

    @classmethod
    def _error_message(cls, details: Any) -> str | None:
        if isinstance(details, str):
            return details[:500]
        if isinstance(details, dict):
            for key in ("description", "message", "error", "detail"):
                value = details.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()[:500]
                if isinstance(value, dict):
                    nested = cls._error_message(value)
                    if nested:
                        return nested
        return None

    async def login(self, username: str, password: str) -> dict[str, Any]:
        response = await self._request(
            "POST",
            "/api/v2/login",
            json={"username": username, "password": password},
        )
        data = self._response_body(response)
        if not isinstance(data, dict) or not data.get("sessionId"):
            raise DresApiError(502, "Phản hồi đăng nhập DRES không có sessionId")
        return data

    async def list_evaluations(self, session_id: str) -> list[dict[str, Any]]:
        response = await self._request(
            "GET",
            "/api/v2/client/evaluation/list",
            params={"session": session_id},
        )
        data = self._response_body(response)
        if not isinstance(data, list):
            raise DresApiError(502, "Phản hồi danh sách evaluation của DRES không hợp lệ")
        return [item for item in data if isinstance(item, dict)]

    async def current_task(self, session_id: str, evaluation_id: str) -> dict[str, Any]:
        response = await self._request(
            "GET",
            f"/api/v2/client/evaluation/currentTask/{evaluation_id}",
            params={"session": session_id},
        )
        data = self._response_body(response)
        if not isinstance(data, dict):
            raise DresApiError(502, "Phản hồi current task của DRES không hợp lệ")
        return data

    async def submit(
        self,
        session_id: str,
        evaluation_id: str,
        payload: dict[str, Any],
    ) -> tuple[int, Any]:
        response = await self._request(
            "POST",
            f"/api/v2/submit/{evaluation_id}",
            params={"session": session_id},
            json=payload,
            expected_statuses={200, 202},
        )
        return response.status_code, self._response_body(response)

    async def logout(self, session_id: str) -> None:
        await self._request(
            "GET",
            "/api/v2/logout",
            params={"session": session_id},
        )

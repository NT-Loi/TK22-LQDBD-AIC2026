"""Local-only DRES v2 mock for safely testing the browser submission flow."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from html import escape
import json
import re
import secrets
from threading import RLock
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel


app = FastAPI(title="AIC 2026 Mock DRES", docs_url="/docs")


class LoginData(BaseModel):
    username: str
    password: str


class NextResponseData(BaseModel):
    outcome: str


MOCK_EVALUATION_ID = "mock-final"
EVALUATIONS = [
    {
        "id": MOCK_EVALUATION_ID,
        "name": "MOCK • Chung kết AIC 2026",
        "type": "SYNCHRONOUS",
        "status": "ACTIVE",
        "templateId": "mock-template",
        "teams": ["team_197"],
        "taskTemplates": [],
    }
]

MOCK_TASK = {
    "name": "Mock Final • Chọn chế độ trên giao diện",
    "taskGroup": "AIC_FINAL",
    "taskType": "MANUAL",
    "duration": 300,
}

MOCK_OUTCOMES = {
    "CORRECT",
    "WRONG",
    "INDETERMINATE",
    "UNDECIDABLE",
    "PENDING",
    "REJECTED",
    "SESSION_EXPIRED",
}

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
VIDEO_ID_RE = r"[A-Za-z0-9][A-Za-z0-9_.]*"

_sessions: dict[str, str] = {}
_submissions: list[dict[str, Any]] = []
_next_response = "CORRECT"
_lock = RLock()


def reset_mock_state() -> None:
    """Reset in-memory state; primarily useful for automated tests."""
    global _next_response
    with _lock:
        _sessions.clear()
        _submissions.clear()
        _next_response = "CORRECT"


def set_next_mock_response(outcome: str) -> None:
    normalized = outcome.strip().upper()
    if normalized not in MOCK_OUTCOMES:
        raise ValueError("Phản hồi mock không được hỗ trợ")
    global _next_response
    with _lock:
        _next_response = normalized


@app.exception_handler(HTTPException)
async def http_error_handler(_: Request, error: HTTPException) -> JSONResponse:
    detail = error.detail
    if isinstance(detail, dict):
        description = detail.get("description") or str(detail)
    else:
        description = str(detail)
    return JSONResponse(
        status_code=error.status_code,
        content={"status": False, "description": description},
    )


def _require_session(session: str) -> str:
    with _lock:
        username = _sessions.get(session)
    if username is None:
        raise HTTPException(status_code=401, detail={"description": "Mock session không hợp lệ"})
    return username


def _extract_single_answer(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        answer_sets = payload["answerSets"]
        answers = answer_sets[0]["answers"]
        answer = answers[0]
    except (KeyError, IndexError, TypeError):
        raise ValueError("Payload phải có answerSets[0].answers[0]") from None
    if len(answer_sets) != 1 or len(answers) != 1 or not isinstance(answer, dict):
        raise ValueError("Mock chỉ chấp nhận đúng một answer trong một answerSet")
    return answer


def _validate_kis(answer: dict[str, Any]) -> None:
    video_id = answer.get("mediaItemName")
    start = answer.get("start")
    end = answer.get("end")
    if not isinstance(video_id, str) or not re.fullmatch(VIDEO_ID_RE, video_id):
        raise ValueError("KIS mediaItemName không hợp lệ")
    if video_id.lower().endswith(VIDEO_EXTENSIONS):
        raise ValueError("KIS mediaItemName không được chứa phần mở rộng")
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError("KIS start phải là millisecond nguyên không âm")
    if end != start:
        raise ValueError("KIS end phải bằng start")


def _validate_qa(answer: dict[str, Any]) -> None:
    text = answer.get("text")
    if not isinstance(text, str) or not re.fullmatch(
        rf"QA-.+-{VIDEO_ID_RE}-\d+",
        text,
    ):
        raise ValueError("Q&A phải có dạng QA-ANSWER-VIDEO_ID-TIME_MS")


def _validate_trake(answer: dict[str, Any]) -> None:
    text = answer.get("text")
    if not isinstance(text, str):
        raise ValueError("TRAKE answer phải là text")
    matched = re.fullmatch(rf"TR-{VIDEO_ID_RE}-(\d+(?:,\d+)*)", text)
    if matched is None:
        raise ValueError("TRAKE phải có dạng TR-VIDEO_ID-FRAME_ID1,FRAME_ID2,...")
    frames = [int(value) for value in matched.group(1).split(",")]
    if len(set(frames)) != len(frames):
        raise ValueError("TRAKE không chấp nhận frame trùng nhau")
    if any(left >= right for left, right in zip(frames, frames[1:])):
        raise ValueError("Frame TRAKE phải tăng nghiêm ngặt")


def _validate_submission(evaluation_id: str, payload: dict[str, Any]) -> str:
    if evaluation_id != MOCK_EVALUATION_ID:
        raise ValueError("Evaluation mock không tồn tại")
    answer = _extract_single_answer(payload)
    if "mediaItemName" in answer:
        _validate_kis(answer)
        return "kis"
    text = answer.get("text")
    if isinstance(text, str) and text.startswith("QA-"):
        _validate_qa(answer)
        return "qa"
    if isinstance(text, str) and text.startswith("TR-"):
        _validate_trake(answer)
        return "trake"
    raise ValueError("Không nhận diện được payload KIS, Q&A hoặc TRAKE")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return """
    <html lang="vi"><head><meta charset="utf-8"><title>Mock DRES</title></head>
    <body style="font-family:system-ui;max-width:900px;margin:40px auto;padding:0 20px">
      <h1>Mock DRES đang hoạt động</h1>
      <p>Server này chỉ nhận submission trên localhost và không chuyển tiếp ra DRES thật.</p>
      <p><a href="/debug">Xem các gói tin đã nhận</a> · <a href="/docs">API docs</a></p>
    </body></html>
    """


@app.post("/api/v2/login")
async def login(data: LoginData) -> dict[str, Any]:
    username = data.username.strip()
    if not username or not data.password:
        raise HTTPException(status_code=401, detail={"description": "Thiếu mock credential"})
    session_id = f"mock-{secrets.token_urlsafe(18)}"
    with _lock:
        _sessions[session_id] = username
    return {
        "sessionId": session_id,
        "username": username,
        "role": "PARTICIPANT",
    }


@app.get("/api/v2/client/evaluation/list")
async def evaluation_list(session: str = Query(...)) -> list[dict[str, Any]]:
    _require_session(session)
    return deepcopy(EVALUATIONS)


@app.get("/api/v2/client/evaluation/currentTask/{evaluation_id}")
async def current_task(
    evaluation_id: str,
    session: str = Query(...),
) -> dict[str, Any]:
    _require_session(session)
    if evaluation_id != MOCK_EVALUATION_ID:
        raise HTTPException(status_code=404, detail={"description": "Không có mock task"})
    return deepcopy(MOCK_TASK)


@app.post("/api/v2/submit/{evaluation_id}")
async def submit(
    evaluation_id: str,
    payload: dict[str, Any],
    request: Request,
    session: str = Query(...),
) -> JSONResponse:
    username = _require_session(session)
    try:
        mode = _validate_submission(evaluation_id, payload)
    except ValueError as error:
        raise HTTPException(status_code=412, detail={"description": str(error)}) from error

    global _next_response
    with _lock:
        outcome = _next_response
        _next_response = "CORRECT"
        submission_number = len(_submissions) + 1
        record = {
            "number": submission_number,
            "receivedAt": datetime.now(UTC).isoformat(),
            "username": username,
            "mode": mode,
            "method": request.method,
            "path": request.url.path,
            "query": {"session": "<mock-session>"},
            "evaluationId": evaluation_id,
            "payload": deepcopy(payload),
            "mockOutcome": outcome,
        }
        _submissions.append(record)

    if outcome == "SESSION_EXPIRED":
        return JSONResponse(
            status_code=401,
            content={"status": False, "description": "Mock session đã hết hạn"},
        )
    if outcome == "REJECTED":
        return JSONResponse(
            status_code=412,
            content={"status": False, "description": "Mock DRES từ chối submission"},
        )
    if outcome == "PENDING":
        return JSONResponse(
            status_code=202,
            content={
                "status": True,
                "submission": "INDETERMINATE",
                "description": "Mock DRES đã nhận bài và đang chờ verdict",
            },
        )
    return JSONResponse(
        status_code=200,
        content={
            "status": True,
            "submission": outcome,
            "description": f"Mock verdict: {outcome}",
        },
    )


@app.get("/api/v2/logout")
async def logout(session: str = Query(...)) -> dict[str, Any]:
    _require_session(session)
    with _lock:
        _sessions.pop(session, None)
    return {"status": True, "description": "Mock logout thành công"}


@app.get("/debug/submissions")
async def debug_submissions() -> dict[str, Any]:
    with _lock:
        items = deepcopy(_submissions)
        next_response = _next_response
    return {
        "count": len(items),
        "nextResponse": next_response,
        "submissions": items,
    }


@app.post("/debug/next-response")
async def debug_next_response(data: NextResponseData) -> dict[str, Any]:
    try:
        set_next_mock_response(data.outcome)
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"description": str(error)}) from error
    return {"status": True, "nextResponse": data.outcome.strip().upper()}


@app.post("/debug/reset", response_class=RedirectResponse)
async def debug_reset() -> RedirectResponse:
    with _lock:
        _submissions.clear()
    return RedirectResponse(url="/debug", status_code=303)


@app.get("/debug", response_class=HTMLResponse)
async def debug_page() -> str:
    with _lock:
        items = deepcopy(_submissions)
        next_response = _next_response
    formatted = escape(json.dumps(items, ensure_ascii=False, indent=2))
    return f"""
    <html lang="vi"><head><meta charset="utf-8"><title>Mock DRES Packets</title>
    </head>
    <body style="font-family:system-ui;max-width:1100px;margin:30px auto;padding:0 20px;background:#0d1117;color:#e6edf3">
      <h1>Gói tin Mock DRES đã nhận: {len(items)}</h1>
      <p>Session đã được che. Phản hồi tự trở về CORRECT sau mỗi lần submit.</p>
      <label for="next-response">Phản hồi cho lần nộp kế tiếp:</label>
      <select id="next-response">
        {''.join(f'<option value="{item}" {"selected" if item == next_response else ""}>{item}</option>' for item in sorted(MOCK_OUTCOMES))}
      </select>
      <button type="button" onclick="setNextResponse()">Áp dụng</button>
      <strong id="next-response-status">Hiện tại: {next_response}</strong>
      <form method="post" action="/debug/reset"><button type="submit">Xóa lịch sử</button></form>
      <pre style="padding:16px;background:#161b22;border:1px solid #30363d;border-radius:8px;white-space:pre-wrap">{formatted or "Chưa có submission"}</pre>
      <script>
        async function setNextResponse() {{
          const outcome = document.getElementById('next-response').value;
          const response = await fetch('/debug/next-response', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{outcome}})
          }});
          const data = await response.json();
          document.getElementById('next-response-status').textContent = response.ok
            ? `Hiện tại: ${{data.nextResponse}}`
            : `Lỗi: ${{data.description || 'không xác định'}}`;
        }}
      </script>
    </body></html>
    """

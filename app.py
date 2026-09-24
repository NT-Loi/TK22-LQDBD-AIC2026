from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Any, List, Literal, Optional
import uvicorn
import os
import hashlib
import json
from contextlib import asynccontextmanager


from dotenv import load_dotenv
from retrieval_system import RetrievalSystem
from utils.video_metadata import load_video_metadata
from config import VISION_EMBEDDING_DIM, KEYFRAME_SEARCH_WEIGHT, CAPTION_SEARCH_WEIGHT, DEFAULT_VISION_MODEL

from services.dres_client import DresApiError, DresClient
from services.dres_session import DresSession, DresSessionStore
from services.dres_submission import build_dres_submission_payload

load_dotenv()
system = None
video_metadata = {}

SUPPORTED_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")

DRES_COOKIE_NAME = "aic_dres_session"
DRES_BASE_URL = os.getenv("DRES_BASE_URL", "https://eventretrieval.one")
DRES_DEFAULT_USERNAME = os.getenv("DRES_DEFAULT_USERNAME", "team_197")
DRES_DEFAULT_PASSWORD = os.getenv("DRES_DEFAULT_PASSWORD", "")
DRES_COOKIE_SECURE = os.getenv("DRES_COOKIE_SECURE", "false").lower() in {"1", "true", "yes"}

try:
    DRES_REQUEST_TIMEOUT_SECONDS = float(os.getenv("DRES_REQUEST_TIMEOUT_SECONDS", "10"))
except ValueError:
    DRES_REQUEST_TIMEOUT_SECONDS = 10.0

dres_client = DresClient(DRES_BASE_URL, DRES_REQUEST_TIMEOUT_SECONDS)
dres_sessions = DresSessionStore(ttl_hours=8)

def get_safe_video_path(video_name: str) -> Optional[str]:
    """Resolves video path securely, matching either exact file or base ID with supported extensions."""
    video_dir = os.path.abspath("data/video")
    # Check exact file request
    direct_path = os.path.abspath(os.path.join(video_dir, video_name))
    if os.path.commonpath([video_dir, direct_path]) == video_dir and os.path.isfile(direct_path):
        return direct_path

    # Check without extension or with alternative extensions (.mp4, .mov, etc.)
    base_id = os.path.splitext(os.path.basename(video_name))[0]
    for ext in SUPPORTED_VIDEO_EXTS + (".MP4", ".MOV"):
        cand = os.path.join(video_dir, f"{base_id}{ext}")
        if os.path.isfile(cand):
            return cand
    return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global system, video_metadata
    # Initialize the retrieval system when the app starts
    system = RetrievalSystem(re_ingest=False)
    video_metadata = load_video_metadata()
    yield
    system = None
    video_metadata = {}

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure media directories exist to prevent errors
os.makedirs("data/video", exist_ok=True)
os.makedirs("data/keyframe", exist_ok=True)

# Video serving endpoint supporting both .mp4 and .mov with range requests
@app.get("/video/{video_name:path}")
async def serve_video(video_name: str):
    file_path = get_safe_video_path(video_name)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Video '{video_name}' not found")
    media_type = "video/mp4" if file_path.lower().endswith((".mp4", ".mov")) else None
    return FileResponse(file_path, media_type=media_type)

# Mount static and keyframe directories
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/keyframes", StaticFiles(directory="data/keyframe"), name="keyframes")

templates = Jinja2Templates(directory="templates")


class SearchQuery(BaseModel):
    text_queries: Optional[List[Any]] = []
    text_filters: Optional[Any] = None  # Non-temporal text filters: list of {text: str, level: "frame"|"video"}
    anchor_index: int = 0
    models: Optional[List[str]] = None
    objects: Optional[List[Any]] = None
    audio: Optional[Any] = None
    ocr_query: Optional[Any] = None
    group_by_shot: bool = False
    group_by_video: bool = False
    score_threshold: float = 0.0
    limit: int = 100
    search_mode: str = "keyframe"  # "keyframe", "caption", "both"
    caption_weight: Optional[float] = None  # weight for caption in fused search

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})

@app.get("/api/models")
async def get_models():
    # Return available model filters
    models = ["score"] + list(VISION_EMBEDDING_DIM.keys())
    return {"models": models}

@app.post("/search")
async def search(query: SearchQuery):
    valid_text_queries = []
    if isinstance(query.text_queries, list):
        for item in query.text_queries:
            if isinstance(item, str) and item.strip():
                valid_text_queries.append({"text": item.strip()})
            elif isinstance(item, dict):
                text_val = str(item.get("text", "")).strip()
                if text_val:
                    valid_text_queries.append(item)
    
    # Resolve audio query (can be string, dict, or list of strings/dicts)
    valid_audio_queries = []
    if isinstance(query.audio, list):
        for item in query.audio:
            if isinstance(item, str) and item.strip():
                valid_audio_queries.append({"text": item.strip(), "level": "frame"})
            elif isinstance(item, dict):
                text_val = str(item.get("text", "")).strip()
                if text_val:
                    lvl = str(item.get("level", "frame")).strip().lower()
                    valid_audio_queries.append({"text": text_val, "level": lvl if lvl in ("frame", "video") else "frame"})
    elif isinstance(query.audio, str) and query.audio.strip():
        valid_audio_queries.append({"text": query.audio.strip(), "level": "frame"})
    elif isinstance(query.audio, dict):
        text_val = str(query.audio.get("text", "")).strip()
        if text_val:
            lvl = str(query.audio.get("level", "frame")).strip().lower()
            valid_audio_queries.append({"text": text_val, "level": lvl if lvl in ("frame", "video") else "frame"})

    # Resolve OCR query (can be string, dict, or list of strings/dicts)
    valid_ocr_queries = []
    if isinstance(query.ocr_query, list):
        for item in query.ocr_query:
            if isinstance(item, str) and item.strip():
                valid_ocr_queries.append({"text": item.strip(), "level": "frame"})
            elif isinstance(item, dict):
                text_val = str(item.get("text", "")).strip()
                if text_val:
                    lvl = str(item.get("level", "frame")).strip().lower()
                    valid_ocr_queries.append({"text": text_val, "level": lvl if lvl in ("frame", "video") else "frame"})
    elif isinstance(query.ocr_query, str) and query.ocr_query.strip():
        valid_ocr_queries.append({"text": query.ocr_query.strip(), "level": "frame"})
    elif isinstance(query.ocr_query, dict):
        text_val = str(query.ocr_query.get("text", "")).strip()
        if text_val:
            lvl = str(query.ocr_query.get("level", "frame")).strip().lower()
            valid_ocr_queries.append({"text": text_val, "level": lvl if lvl in ("frame", "video") else "frame"})

    # Resolve text filters (non-temporal frame/video level visual filters)
    valid_text_filters = []
    if isinstance(query.text_filters, list):
        for item in query.text_filters:
            if isinstance(item, str) and item.strip():
                valid_text_filters.append({"text": item.strip(), "level": "frame"})
            elif isinstance(item, dict):
                text_val = str(item.get("text", "")).strip()
                if text_val:
                    lvl = str(item.get("level", "frame")).strip().lower()
                    valid_text_filters.append({"text": text_val, "level": lvl if lvl in ("frame", "video") else "frame"})
    elif isinstance(query.text_filters, str) and query.text_filters.strip():
        valid_text_filters.append({"text": query.text_filters.strip(), "level": "frame"})
    elif isinstance(query.text_filters, dict):
        text_val = str(query.text_filters.get("text", "")).strip()
        if text_val:
            lvl = str(query.text_filters.get("level", "frame")).strip().lower()
            valid_text_filters.append({"text": text_val, "level": lvl if lvl in ("frame", "video") else "frame"})

    audio_query = valid_audio_queries if valid_audio_queries else None
    ocr_query = valid_ocr_queries if valid_ocr_queries else None
    text_filters = valid_text_filters if valid_text_filters else None
    
    # Determine model_names based on selected models (Direction 3: default to SigLIP2)
    model_names = None
    if query.models and "all" not in query.models:
        model_names = query.models
    elif query.models and "all" in query.models:
        model_names = ["all"]
    else:
        model_names = [DEFAULT_VISION_MODEL]
    
    if len(valid_text_queries) > 1:
        # Perform temporal search
        results = system.temporal_search(
            valid_text_queries, 
            model_names=model_names, 
            objects=query.objects,
            group_by_video=query.group_by_video, 
            score_threshold=query.score_threshold,
            limit=query.limit,
            ocr_query=ocr_query,
            audio_query=audio_query,
            text_filters=text_filters
        )
    elif len(valid_text_queries) == 1:
        # Perform search using the single event query
        q_item = valid_text_queries[0]
        q_text = q_item["text"] if isinstance(q_item, dict) else str(q_item)
        q_ocr = q_item.get("ocr") if (isinstance(q_item, dict) and q_item.get("ocr") is not None) else ocr_query
        q_audio = q_item.get("audio") if (isinstance(q_item, dict) and q_item.get("audio") is not None) else audio_query
        q_objects = q_item.get("objects") if (isinstance(q_item, dict) and q_item.get("objects") is not None) else query.objects

        search_mode = query.search_mode or "keyframe"

        if search_mode == "caption":
            results = system.caption_search(
                q_text,
                score_threshold=query.score_threshold,
                limit=query.limit,
                ocr_query=q_ocr,
                audio_query=q_audio,
                objects=q_objects,
                text_filters=text_filters
            )
        elif search_mode == "both":
            # Compute weights: caption_weight from request, keyframe_weight = 1 - caption_weight
            cap_w = query.caption_weight if query.caption_weight is not None else CAPTION_SEARCH_WEIGHT
            kf_w = 1.0 - cap_w
            results = system.fused_search(
                q_text,
                model_names=model_names,
                objects=q_objects,
                score_threshold=query.score_threshold,
                limit=query.limit,
                ocr_query=q_ocr,
                audio_query=q_audio,
                keyframe_weight=kf_w,
                caption_weight=cap_w,
                text_filters=text_filters,
                group_by_shot=query.group_by_shot
            )
        else:
            results = system.semantic_search(
                q_text, 
                model_names=model_names, 
                objects=q_objects,
                score_threshold=query.score_threshold,
                group_by_shot=query.group_by_shot,
                limit=query.limit,
                ocr_query=q_ocr,
                audio_query=q_audio,
                text_filters=text_filters
            )
    else:
        # No text queries -> Filter search (OCR, audio transcript, object detection, text query filter)
        results = system.filter_search(
            ocr_query=ocr_query,
            audio_query=audio_query,
            objects=query.objects,
            text_filters=text_filters,
            model_names=model_names,
            score_threshold=query.score_threshold,
            group_by_shot=query.group_by_shot,
            limit=query.limit
        )
    
    # Map results and add accurate FPS for the frontend
    for item in results:
        vid = item.get("video_id")
        fps = 25.0
        if vid and vid in video_metadata and "fps" in video_metadata[vid]:
            fps = video_metadata[vid]["fps"]
            
        item["fps"] = fps
        if "frames" in item:
            for frame in item["frames"]:
                frame["fps"] = fps
        if "display_frames" in item:
            for frame in item["display_frames"]:
                frame["fps"] = fps
            
    return results

class OCRSearchQuery(BaseModel):
    query: str
    fuzziness: str = "AUTO"
    limit: int = 200

@app.post("/ocr-search")
async def ocr_search(query: OCRSearchQuery):
    if not query.query:
        return []
    
    results = system.ocr_search(query.query, fuzziness=query.fuzziness, size=query.limit)
    
    # Add FPS metadata
    for item in results:
        vid = item.get("video_id")
        fps = 25.0
        if vid and vid in video_metadata and "fps" in video_metadata[vid]:
            fps = video_metadata[vid]["fps"]
        item["fps"] = fps
    
    return results

@app.get("/api/video_info/{video_id}")
async def get_video_info(video_id: str):
    raw_vid = video_id.strip()
    base_vid = os.path.splitext(raw_vid)[0]
    fps = 25.0
    for key in (base_vid, raw_vid):
        if key in video_metadata and "fps" in video_metadata[key]:
            fps = video_metadata[key]["fps"]
            break
    
    video_path = get_safe_video_path(raw_vid)
    return {
        "video_id": base_vid,
        "fps": fps,
        "exists": video_path is not None
    }

@app.get("/api/video_keyframes/{video_id}")
async def get_video_keyframes(video_id: str):
    raw_vid = video_id.strip()
    vid = os.path.splitext(raw_vid)[0]
    fps = 25.0
    for key in (vid, raw_vid):
        if key in video_metadata and "fps" in video_metadata[key]:
            fps = video_metadata[key]["fps"]
            break

    kf_dir = os.path.join("data", "keyframe", vid)
    keyframes = []
    if os.path.exists(kf_dir) and os.path.isdir(kf_dir):
        for fname in os.listdir(kf_dir):
            if fname.startswith("keyframe_") and fname.endswith(".webp"):
                try:
                    idx = int(fname.replace("keyframe_", "").replace(".webp", ""))
                    keyframes.append({
                        "video_id": vid,
                        "keyframe_index": idx,
                        "fps": fps
                    })
                except ValueError:
                    pass
        keyframes.sort(key=lambda x: x["keyframe_index"])
        
    return {
        "video_id": vid,
        "fps": fps,
        "keyframes": keyframes
    }

@app.get("/api/shot_keyframes/{video_id}")
async def get_shot_keyframes(video_id: str, start_frame: int = 0, end_frame: int = 999999999):
    """Return all keyframes in a given shot range [start_frame, end_frame] for a video."""
    raw_vid = video_id.strip()
    vid = os.path.splitext(raw_vid)[0]
    fps = 25.0
    for key in (vid, raw_vid):
        if key in video_metadata and "fps" in video_metadata[key]:
            fps = video_metadata[key]["fps"]
            break

    kf_dir = os.path.join("data", "keyframe", vid)
    keyframes = []
    if os.path.exists(kf_dir) and os.path.isdir(kf_dir):
        for fname in os.listdir(kf_dir):
            if fname.startswith("keyframe_") and fname.endswith(".webp"):
                try:
                    idx = int(fname.replace("keyframe_", "").replace(".webp", ""))
                    if start_frame <= idx <= end_frame:
                        keyframes.append({
                            "video_id": vid,
                            "keyframe_index": idx,
                            "fps": fps
                        })
                except ValueError:
                    pass
        keyframes.sort(key=lambda x: x["keyframe_index"])

    return {
        "video_id": vid,
        "fps": fps,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "keyframes": keyframes
    }

videos_cache = []

def scan_all_videos():
    global videos_cache
    kf_dir = os.path.join("data", "keyframe")
    videos = []
    if os.path.exists(kf_dir) and os.path.isdir(kf_dir):
        for v_name in sorted(os.listdir(kf_dir)):
            v_path = os.path.join(kf_dir, v_name)
            if os.path.isdir(v_path):
                files = [f for f in os.listdir(v_path) if f.endswith(".webp")]
                fps = 25.0
                if v_name in video_metadata and "fps" in video_metadata[v_name]:
                    fps = video_metadata[v_name]["fps"]
                
                sorted_files = sorted(files, key=lambda x: int(x.replace("keyframe_", "").replace(".webp", "")) if x.replace("keyframe_", "").replace(".webp", "").isdigit() else 0)
                first_kf = sorted_files[0] if sorted_files else None
                
                videos.append({
                    "video_id": v_name,
                    "fps": fps,
                    "keyframe_count": len(files),
                    "first_keyframe": first_kf
                })
    videos_cache = videos
    return videos

@app.get("/api/videos")
async def get_videos():
    global videos_cache
    if not videos_cache:
        scan_all_videos()
    return {"videos": videos_cache}

class DresLoginData(BaseModel):
    username: str
    password: str


class DresEvaluationSelection(BaseModel):
    evaluationId: str


class DresSubmissionData(BaseModel):
    evaluationId: Optional[str] = None
    mode: Literal["kis", "qa", "trake"]
    videoId: str
    timeMs: Optional[int] = None
    answer: Optional[str] = None
    frameIds: Optional[List[int]] = None


def _public_user(user: dict[str, Any]) -> dict[str, Any]:
    return {
        key: user.get(key)
        for key in ("id", "username", "role")
        if user.get(key) is not None
    }


def _active_evaluations(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        evaluation
        for evaluation in evaluations
        if str(evaluation.get("status", "")).upper() == "ACTIVE"
    ]


def _dres_http_error(error: DresApiError) -> HTTPException:
    status_code = error.status_code if 400 <= error.status_code <= 599 else 502
    return HTTPException(status_code=status_code, detail=error.message)


def _require_dres_session(request: Request) -> tuple[str, DresSession]:
    local_id = request.cookies.get(DRES_COOKIE_NAME)
    session = dres_sessions.get(local_id)
    if session is None:
        raise HTTPException(status_code=401, detail="Phiên DRES không tồn tại hoặc đã hết hạn")
    return local_id, session


def _session_response(session: DresSession) -> dict[str, Any]:
    return {
        "connected": True,
        "user": _public_user(session.user),
        "evaluations": session.evaluations,
        "selectedEvaluationId": session.selected_evaluation_id,
        "defaultUsername": DRES_DEFAULT_USERNAME,
        "dresBaseUrl": DRES_BASE_URL,
    }


async def _login_to_dres(
    request: Request,
    response: Response,
    username: str,
    password: str,
) -> dict[str, Any]:
    username = username.strip()
    if not username or not password:
        raise HTTPException(status_code=422, detail="Username và password không được để trống")

    old_local_id = request.cookies.get(DRES_COOKIE_NAME)
    old_session = dres_sessions.delete(old_local_id)
    if old_session is not None:
        try:
            await dres_client.logout(old_session.dres_session_id)
        except DresApiError:
            pass

    try:
        dres_user = await dres_client.login(username, password)
        evaluations = _active_evaluations(
            await dres_client.list_evaluations(dres_user["sessionId"])
        )
    except DresApiError as error:
        raise _dres_http_error(error) from error

    session = dres_sessions.create(
        dres_user["sessionId"],
        _public_user(dres_user),
        evaluations,
    )
    response.set_cookie(
        DRES_COOKIE_NAME,
        session.local_id,
        max_age=8 * 60 * 60,
        httponly=True,
        secure=DRES_COOKIE_SECURE,
        samesite="strict",
        path="/",
    )
    return _session_response(session)


@app.post("/api/dres/login/default")
async def dres_default_login(request: Request, response: Response):
    if not DRES_DEFAULT_PASSWORD:
        raise HTTPException(
            status_code=503,
            detail="DRES_DEFAULT_PASSWORD chưa được cấu hình trong .env",
        )
    return await _login_to_dres(
        request,
        response,
        DRES_DEFAULT_USERNAME,
        DRES_DEFAULT_PASSWORD,
    )


@app.post("/api/dres/login")
async def dres_custom_login(data: DresLoginData, request: Request, response: Response):
    return await _login_to_dres(request, response, data.username, data.password)


@app.get("/api/dres/session")
async def dres_session_status(request: Request):
    local_id = request.cookies.get(DRES_COOKIE_NAME)
    session = dres_sessions.get(local_id)
    if session is None:
        return {
            "connected": False,
            "defaultUsername": DRES_DEFAULT_USERNAME,
            "dresBaseUrl": DRES_BASE_URL,
        }
    return _session_response(session)


@app.get("/api/dres/evaluations")
async def dres_evaluation_list(request: Request):
    _, session = _require_dres_session(request)
    try:
        session.evaluations = _active_evaluations(
            await dres_client.list_evaluations(session.dres_session_id)
        )
    except DresApiError as error:
        if error.status_code == 401:
            dres_sessions.delete(request.cookies.get(DRES_COOKIE_NAME))
        raise _dres_http_error(error) from error
    return {"evaluations": session.evaluations}


@app.post("/api/dres/evaluation")
async def dres_select_evaluation(data: DresEvaluationSelection, request: Request):
    local_id, session = _require_dres_session(request)
    visible_ids = {str(evaluation.get("id")) for evaluation in session.evaluations}
    if data.evaluationId not in visible_ids:
        raise HTTPException(status_code=404, detail="Evaluation không tồn tại hoặc không ACTIVE")
    dres_sessions.select_evaluation(local_id, data.evaluationId)
    return {"selectedEvaluationId": data.evaluationId}


@app.get("/api/dres/current-task/{evaluation_id}")
async def dres_current_task(evaluation_id: str, request: Request):
    _, session = _require_dres_session(request)
    visible_ids = {str(evaluation.get("id")) for evaluation in session.evaluations}
    if evaluation_id not in visible_ids:
        raise HTTPException(status_code=404, detail="Evaluation không tồn tại hoặc không ACTIVE")
    try:
        task = await dres_client.current_task(session.dres_session_id, evaluation_id)
    except DresApiError as error:
        if error.status_code == 401:
            dres_sessions.delete(request.cookies.get(DRES_COOKIE_NAME))
        raise _dres_http_error(error) from error
    return {"task": task}


def _submission_fingerprint(
    evaluation_id: str,
    task: dict[str, Any],
    payload: dict[str, Any],
) -> str:
    task_identity = {
        "id": task.get("id"),
        "taskId": task.get("taskId"),
        "name": task.get("name"),
        "taskGroup": task.get("taskGroup"),
        "taskType": task.get("taskType"),
    }
    serialized = json.dumps(
        {"evaluationId": evaluation_id, "task": task_identity, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@app.post("/api/dres/submit")
async def dres_submit(data: DresSubmissionData, request: Request):
    local_id, session = _require_dres_session(request)
    evaluation_id = data.evaluationId or session.selected_evaluation_id
    if not evaluation_id:
        raise HTTPException(status_code=422, detail="Chưa chọn evaluation")
    visible_ids = {str(evaluation.get("id")) for evaluation in session.evaluations}
    if evaluation_id not in visible_ids:
        raise HTTPException(status_code=404, detail="Evaluation không tồn tại hoặc không ACTIVE")

    try:
        payload = build_dres_submission_payload(
            mode=data.mode,
            video_id=data.videoId,
            time_ms=data.timeMs,
            answer=data.answer,
            frame_ids=data.frameIds,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    task: dict[str, Any] | None = None
    fingerprint: str | None = None
    try:
        task = await dres_client.current_task(session.dres_session_id, evaluation_id)
    except DresApiError as error:
        if error.status_code == 401:
            dres_sessions.delete(local_id)
            raise _dres_http_error(error) from error
        if error.status_code != 404:
            raise _dres_http_error(error) from error

    if task:
        fingerprint = _submission_fingerprint(evaluation_id, task, payload)
        if dres_sessions.has_fingerprint(local_id, fingerprint):
            raise HTTPException(
                status_code=409,
                detail="Payload này đã được nộp cho task DRES hiện tại",
            )

    try:
        status_code, dres_response = await dres_client.submit(
            session.dres_session_id,
            evaluation_id,
            payload,
        )
    except DresApiError as error:
        if error.status_code == 401:
            dres_sessions.delete(local_id)
        raise _dres_http_error(error) from error

    if fingerprint:
        dres_sessions.record_fingerprint(local_id, fingerprint)
    session.selected_evaluation_id = evaluation_id
    return {
        "accepted": True,
        "pending": status_code == 202,
        "dresStatus": status_code,
        "result": dres_response,
        "payload": payload,
        "task": task,
    }


@app.post("/api/dres/logout")
async def dres_logout(request: Request, response: Response):
    local_id = request.cookies.get(DRES_COOKIE_NAME)
    session = dres_sessions.delete(local_id)
    if session is not None:
        try:
            await dres_client.logout(session.dres_session_id)
        except DresApiError:
            pass
    response.delete_cookie(DRES_COOKIE_NAME, path="/")
    return {"status": "success"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

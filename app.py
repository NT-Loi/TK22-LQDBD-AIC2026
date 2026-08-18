from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Union
import uvicorn
import os
from contextlib import asynccontextmanager

from retrieval_system import RetrievalSystem
from utils.video_metadata import load_video_metadata
from config import VECTOR_SIZES

system = None
video_metadata = {}

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

# Ensure media directories exist to prevent StaticFiles from crashing
os.makedirs("data/video", exist_ok=True)
os.makedirs("data/keyframe", exist_ok=True)

# Mount static and media directories
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/video", StaticFiles(directory="data/video"), name="video")
app.mount("/keyframes", StaticFiles(directory="data/keyframe"), name="keyframes")

templates = Jinja2Templates(directory="templates")

from typing import List, Optional, Union, Any

class SearchQuery(BaseModel):
    text_queries: Optional[List[Any]] = []
    anchor_index: int = 0
    models: Optional[List[str]] = None
    objects: Optional[List[Any]] = None
    audio: Optional[Any] = None
    ocr_query: Optional[Any] = None
    group_by_shot: bool = False
    group_by_video: bool = False
    score_threshold: float = 0.0
    limit: int = 100

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})

@app.get("/api/models")
async def get_models():
    # Return available model filters
    models = ["score"] + list(VECTOR_SIZES.keys())
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

    audio_query = valid_audio_queries if valid_audio_queries else None
    ocr_query = valid_ocr_queries if valid_ocr_queries else None
    
    # Determine model_names based on selected models
    model_names = None
    if query.models and "all" not in query.models:
        model_names = query.models
    
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
            audio_query=audio_query
        )
    elif len(valid_text_queries) == 1:
        # Perform search using the single event query
        q_item = valid_text_queries[0]
        q_text = q_item["text"] if isinstance(q_item, dict) else str(q_item)
        q_ocr = q_item.get("ocr") if (isinstance(q_item, dict) and q_item.get("ocr") is not None) else ocr_query
        q_audio = q_item.get("audio") if (isinstance(q_item, dict) and q_item.get("audio") is not None) else audio_query
        q_objects = q_item.get("objects") if (isinstance(q_item, dict) and q_item.get("objects") is not None) else query.objects

        results = system.semantic_search(
            q_text, 
            model_names=model_names, 
            objects=q_objects,
            score_threshold=query.score_threshold,
            group_by_shot=query.group_by_shot,
            limit=query.limit,
            ocr_query=q_ocr,
            audio_query=q_audio
        )
    else:
        # No text queries -> Filter search (OCR, audio transcript, object detection)
        results = system.filter_search(
            ocr_query=ocr_query,
            audio_query=audio_query,
            objects=query.objects,
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

@app.post("/api/login")
async def login(request: Request):
    # Stub login endpoint
    return {
        "sessionId": "session-1234",
        "evaluations": [
            {"id": "eval-1", "name": "AIC 2026 Test Evaluation", "status": "active"}
        ]
    }

class SubmitData(BaseModel):
    sessionId: str
    evaluationId: str
    videoId: str
    timeMs: int

@app.post("/api/submit")
async def submit(data: SubmitData):
    print(f"Submitted result: Video={data.videoId}, Time={data.timeMs}ms (Session: {data.sessionId})")
    return {"status": "success"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

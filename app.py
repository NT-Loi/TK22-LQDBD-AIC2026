from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
from contextlib import asynccontextmanager

from retrieval_system import RetrievalSystem
from utils.video_metadata import load_video_metadata
from config import VECTOR_SIZES

from fastapi.responses import FileResponse, Response
from pathlib import Path

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

# Ensure media directories exist
os.makedirs("data/video", exist_ok=True)
os.makedirs("data/keyframe", exist_ok=True)

from fastapi.responses import FileResponse, Response
from pathlib import Path

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/keyframes/{path:path}")
async def get_keyframe(path: str):
    file_path = Path("data/keyframe") / path
    if file_path.is_file():
        return FileResponse(file_path)

    matches = list(Path("data/keyframe").glob(f"**/{path}"))
    if matches:
        return FileResponse(matches[0])

    return FileResponse("static/placeholder.png")

@app.get("/video/{path:path}")
async def get_video(path: str):
    file_path = Path("data/video") / path
    if file_path.is_file():
        return FileResponse(file_path)

    matches = list(Path("data/video").glob(f"**/{path}"))
    if matches:
        return FileResponse(matches[0])

    return Response(status_code=404)

templates = Jinja2Templates(directory="templates")

class SearchQuery(BaseModel):
    text_queries: List[str]
    anchor_index: int
    models: List[str]
    objects: list
    audio: str
    group_by_shot: bool = False
    score_threshold: float = 0.3
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
    if not query.text_queries or query.anchor_index >= len(query.text_queries):
        return []
    
    primary_query = query.text_queries[query.anchor_index]
    
    # Determine model_names based on selected models
    model_names = None
    if query.models and "all" not in query.models:
        model_names = query.models
    
    # Use audio field as text query for whisper/ocr text filter
    text_query = query.audio.strip() if query.audio else None

    if len(query.text_queries) > 1:
        # Perform temporal search
        results = system.temporal_search(
            query.text_queries, 
            model_names=model_names, 
            objects=query.objects,
            text_query=text_query,
            group_by_shot=query.group_by_shot, 
            score_threshold=query.score_threshold,
            limit=query.limit
        )
    else:
        # Perform search using the primary event query
        results = system.semantic_search(
            primary_query, 
            model_names=model_names, 
            objects=query.objects,
            text_query=text_query,
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
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

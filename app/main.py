from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

from .config import settings
from .reid import ReIDEngine
from .search import SearchSession, detect_query_person
from .utils import decode_data_url, encode_jpeg

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

queries: dict[str, np.ndarray] = {}
reid = ReIDEngine(settings)
query_detector: YOLO | None = None
model_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global query_detector
    log.info("Loading %s...", settings.yolo_model)
    query_detector = YOLO(settings.yolo_model)
    # Re-ID is lazy-loaded on first target upload so the UI opens immediately.
    yield
    queries.clear()


app = FastAPI(title="Person Search", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(settings.static_dir / "index.html")


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "online",
        "device": "GPU" if torch.cuda.is_available() else "CPU",
        "detector": settings.yolo_model,
        "tracker": settings.tracker,
        "reid": "OSNet-AIN x1.0 (MSMT17)",
    }


@app.post("/api/target")
async def upload_target(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(415, "Vui lòng chọn một tệp ảnh")
    raw = await file.read()
    if len(raw) > 12 * 1024 * 1024:
        raise HTTPException(413, "Ảnh tối đa 12 MB")
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Ảnh không hợp lệ")
    assert query_detector is not None
    async with model_lock:
        crop, box = await asyncio.to_thread(detect_query_person, query_detector, image, settings)
        # Query-time flip augmentation is worth the one-off cost and improves robustness.
        feature = await asyncio.to_thread(reid.embed, [crop], True)
    session_id = uuid.uuid4().hex
    queries[session_id] = feature
    return {
        "session_id": session_id,
        "preview": encode_jpeg(crop, 90),
        "person_detected": box is not None,
        "bbox": box,
    }


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str) -> dict:
    queries.pop(session_id, None)
    return {"ok": True}


@app.websocket("/ws/search/{session_id}")
async def search_socket(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    query = queries.get(session_id)
    if query is None:
        await websocket.send_json({"type": "error", "message": "Hãy tải ảnh người cần tìm trước"})
        await websocket.close(code=4404)
        return
    try:
        session = await asyncio.to_thread(SearchSession, settings, reid, query)
        await websocket.send_json({"type": "ready"})
        while True:
            payload = await websocket.receive_json()
            if payload.get("type") == "reset":
                session = await asyncio.to_thread(SearchSession, settings, reid, query)
                await websocket.send_json({"type": "ready"})
                continue
            if payload.get("type") != "frame":
                continue
            try:
                frame = decode_data_url(payload.get("data", ""))
                threshold = float(payload.get("threshold", settings.match_threshold))
                threshold = min(max(threshold, 0.1), 0.95)
                result = await asyncio.to_thread(session.process, frame, threshold)
                await websocket.send_json(result)
            except (ValueError, TypeError) as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.exception("Search websocket failed")
        try:
            await websocket.send_json({"type": "error", "message": f"Lỗi xử lý: {exc}"})
        except Exception:
            pass


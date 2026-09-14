from __future__ import annotations

import base64
from datetime import datetime

import cv2
import numpy as np


def decode_data_url(value: str) -> np.ndarray:
    """Decode a browser data URL (or raw base64) into an OpenCV BGR image."""
    encoded = value.split(",", 1)[1] if "," in value else value
    raw = base64.b64decode(encoded, validate=True)
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Không thể giải mã ảnh")
    return frame


def encode_jpeg(image: np.ndarray, quality: int = 82) -> str:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_crop(frame: np.ndarray, box: tuple[int, int, int, int], pad: float = 0.025) -> np.ndarray:
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    return frame[y1:y2, x1:x2].copy()


def crop_quality(crop: np.ndarray, confidence: float) -> float:
    """Prefer large, sharp, high-confidence crops for a track's feature gallery."""
    if crop.size == 0:
        return 0.0
    h, w = crop.shape[:2]
    sharpness = min(float(cv2.Laplacian(crop, cv2.CV_64F).var()) / 450.0, 1.0)
    area = min((h * w) / 110_000.0, 1.0)
    aspect = w / max(h, 1)
    aspect_score = max(0.0, 1.0 - abs(aspect - 0.42) / 0.75)
    return 0.42 * confidence + 0.28 * sharpness + 0.2 * area + 0.1 * aspect_score


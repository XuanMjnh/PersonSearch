from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .config import Settings
from .reid import ReIDEngine
from .utils import crop_quality, encode_jpeg, iso_now, safe_crop


@dataclass
class GalleryItem:
    quality: float
    feature: np.ndarray


class SearchSession:
    """One independent tracker and temporal Re-ID gallery per browser session."""

    def __init__(self, config: Settings, reid: ReIDEngine, query: np.ndarray):
        self.config = config
        self.reid = reid
        self.query = query
        self.detector = YOLO(config.yolo_model)
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.frame_number = 0
        self.seen_ids: set[int] = set()
        self.gallery: dict[int, list[GalleryItem]] = defaultdict(list)
        self.scores: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=7))
        self.streaks: dict[int, int] = defaultdict(int)
        self.best: dict | None = None
        self.started = time.perf_counter()
        self.last_seen: dict[int, int] = {}

    def _update_gallery(self, track_id: int, feature: np.ndarray, quality: float) -> None:
        items = self.gallery[track_id]
        items.append(GalleryItem(quality, feature))
        items.sort(key=lambda x: x.quality, reverse=True)
        del items[self.config.max_gallery_features:]

    def _stable_score(self, track_id: int, current: float) -> float:
        history = self.scores[track_id]
        history.append(current)
        weights = np.linspace(0.65, 1.0, len(history), dtype=np.float32)
        return float(np.average(np.asarray(history), weights=weights))

    def process(self, frame: np.ndarray, threshold: float) -> dict:
        started = time.perf_counter()
        self.frame_number += 1
        output = self.detector.track(
            frame,
            persist=True,
            tracker=self.config.tracker,
            classes=[0],
            conf=self.config.detection_confidence,
            iou=self.config.detection_iou,
            imgsz=self.config.image_size,
            device=self.device,
            half=torch.cuda.is_available(),
            verbose=False,
        )[0]

        detections: list[dict] = []
        pending: list[tuple[int, np.ndarray, float]] = []
        if output.boxes is not None and len(output.boxes):
            xyxy = output.boxes.xyxy.detach().cpu().numpy().astype(int)
            confidences = output.boxes.conf.detach().cpu().numpy()
            ids_tensor = output.boxes.id
            ids = ids_tensor.detach().cpu().numpy().astype(int) if ids_tensor is not None else np.arange(len(xyxy)) + 1
            for box, conf, track_id in zip(xyxy, confidences, ids):
                x1, y1, x2, y2 = map(int, box)
                crop = safe_crop(frame, (x1, y1, x2, y2))
                self.seen_ids.add(int(track_id))
                self.last_seen[int(track_id)] = self.frame_number
                quality = crop_quality(crop, float(conf))
                if (
                    crop.shape[0] >= self.config.min_person_height
                    and (self.frame_number % self.config.reid_every_n_frames == 0 or track_id not in self.gallery)
                    and quality >= 0.28
                ):
                    pending.append((int(track_id), crop, quality))
                detections.append({
                    "track_id": int(track_id),
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(float(conf), 4),
                    "crop": crop,
                })

        if pending:
            features = self.reid.embed([item[1] for item in pending])
            for (track_id, _, quality), feature in zip(pending, features):
                self._update_gallery(track_id, feature, quality)

        best_in_frame: dict | None = None
        response_boxes: list[dict] = []
        for det in detections:
            track_id = det["track_id"]
            items = self.gallery.get(track_id, [])
            gallery = np.stack([x.feature for x in items]) if items else np.empty((0, self.query.shape[1]))
            raw = self.reid.similarity(self.query, gallery)
            score = self._stable_score(track_id, raw)
            if score >= threshold:
                self.streaks[track_id] += 1
            else:
                self.streaks[track_id] = max(0, self.streaks[track_id] - 1)
            matched = self.streaks[track_id] >= 2
            item = {
                "track_id": track_id,
                "bbox": det["bbox"],
                "confidence": det["confidence"],
                "similarity": round(score, 4),
                "matched": matched,
            }
            response_boxes.append(item)
            if best_in_frame is None or score > best_in_frame["similarity"]:
                best_in_frame = {**item, "crop": det["crop"]}

        promote_to_match = bool(
            best_in_frame
            and self.best
            and best_in_frame["track_id"] == self.best["track_id"]
            and best_in_frame["matched"]
            and not self.best["matched"]
        )
        if best_in_frame and (
            self.best is None
            or best_in_frame["similarity"] > self.best["similarity"]
            or promote_to_match
        ):
            self.best = {
                "track_id": best_in_frame["track_id"],
                "similarity": best_in_frame["similarity"],
                "matched": best_in_frame["similarity"] >= threshold,
                "time": iso_now(),
                "frame": self.frame_number,
                "crop": encode_jpeg(best_in_frame["crop"], 80),
            }

        # Expire feature galleries from IDs absent for a long time.
        stale = [track_id for track_id, seen in self.last_seen.items() if self.frame_number - seen > 300]
        for track_id in stale:
            self.gallery.pop(track_id, None)
            self.scores.pop(track_id, None)
            self.streaks.pop(track_id, None)
            self.last_seen.pop(track_id, None)

        elapsed = max(time.perf_counter() - started, 1e-6)
        return {
            "type": "result",
            "frame": self.frame_number,
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "boxes": response_boxes,
            "best": self.best,
            "stats": {
                "people": len(response_boxes),
                "tracks": len(self.last_seen),
                "total_people": len(self.seen_ids),
                "fps": round(1.0 / elapsed, 1),
                "uptime": round(time.perf_counter() - self.started, 1),
                "device": "GPU" if torch.cuda.is_available() else "CPU",
            },
        }


def detect_query_person(detector: YOLO, image: np.ndarray, config: Settings) -> tuple[np.ndarray, list[int] | None]:
    result = detector.predict(
        image, classes=[0], conf=0.2, iou=0.65, imgsz=config.image_size, verbose=False
    )[0]
    if result.boxes is None or not len(result.boxes):
        return image, None
    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(int)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    box = boxes[int(np.argmax(areas))].tolist()
    return safe_crop(image, tuple(box), pad=0.04), box

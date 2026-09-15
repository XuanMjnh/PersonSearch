from __future__ import annotations

import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .config import Settings
from .reid import ReIDEngine
from .utils import encode_jpeg, iso_now, safe_crop


class SearchSession:
    """One independent tracker with per-frame Re-ID scores per browser session."""

    def __init__(self, config: Settings, reid: ReIDEngine, query: np.ndarray):
        self.config = config
        self.reid = reid
        self.query = query
        self.detector = YOLO(config.yolo_model)
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.frame_number = 0
        self.seen_ids: set[int] = set()
        self.streaks: dict[int, int] = defaultdict(int)
        self.best: dict | None = None
        self.started = time.perf_counter()
        self.last_seen: dict[int, int] = {}

    def _realtime_scores(self, track_ids: list[int], features: np.ndarray) -> dict[int, float]:
        """Score each track from its crop in the current frame only."""
        return {
            track_id: self.reid.similarity(self.query, feature[None, :])
            for track_id, feature in zip(track_ids, features)
        }

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
        pending: list[tuple[int, np.ndarray]] = []
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
                if crop.shape[0] >= self.config.min_person_height:
                    pending.append((int(track_id), crop))
                detections.append({
                    "track_id": int(track_id),
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(float(conf), 4),
                    "crop": crop,
                })

        current_scores: dict[int, float] = {}
        if pending:
            features = self.reid.embed([item[1] for item in pending])
            current_scores = self._realtime_scores([item[0] for item in pending], features)

        best_in_frame: dict | None = None
        response_boxes: list[dict] = []
        for det in detections:
            track_id = det["track_id"]
            score = current_scores.get(track_id, 0.0)
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
                "matched": best_in_frame["matched"],
                "time": iso_now(),
                "frame": self.frame_number,
                "crop": encode_jpeg(best_in_frame["crop"], 80),
            }

        visible_ids = {det["track_id"] for det in detections}
        for track_id in list(self.streaks):
            if track_id not in visible_ids:
                self.streaks[track_id] = 0

        # Expire state from IDs absent for a long time.
        stale = [track_id for track_id, seen in self.last_seen.items() if self.frame_number - seen > 300]
        for track_id in stale:
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
                "best_similarity": best_in_frame["similarity"] if best_in_frame else None,
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

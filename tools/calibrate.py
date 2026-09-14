from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.calibration import choose_threshold
from app.config import settings
from app.reid import ReIDEngine
from app.search import detect_query_person


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def images_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def load_person(path: Path, detector: YOLO) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Không đọc được: {path}")
    crop, _ = detect_query_person(detector, image, settings)
    return crop


def main() -> None:
    parser = argparse.ArgumentParser(description="Hiệu chỉnh ngưỡng Person Search bằng validation set")
    parser.add_argument("--target", type=Path, required=True, help="Ảnh mục tiêu")
    parser.add_argument("--same", type=Path, required=True, help="Thư mục ảnh cùng người")
    parser.add_argument("--different", type=Path, required=True, help="Thư mục ảnh người khác")
    args = parser.parse_args()

    same_paths, different_paths = images_in(args.same), images_in(args.different)
    if not same_paths or not different_paths:
        raise SystemExit("Mỗi thư mục --same và --different phải có ít nhất một ảnh")

    detector = YOLO(settings.yolo_model)
    reid = ReIDEngine(settings)
    query = reid.embed([load_person(args.target, detector)], augment=True)

    all_paths = same_paths + different_paths
    crops = [load_person(path, detector) for path in all_paths]
    features = reid.embed(crops)
    scores = [reid.similarity(query, feature[None, :]) for feature in features]
    positive_scores = scores[: len(same_paths)]
    negative_scores = scores[len(same_paths) :]
    result = choose_threshold(positive_scores, negative_scores)
    result["same_samples"] = len(same_paths)
    result["different_samples"] = len(different_paths)
    result["positive_score_mean"] = round(float(np.mean(positive_scores)), 4)
    result["negative_score_mean"] = round(float(np.mean(negative_scores)), 4)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


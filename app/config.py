from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def resolve_model_path(value: str) -> str:
    """Keep relative model weights under the project's models directory."""
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if path.parent == Path("."):
        return str(MODEL_DIR / path.name)
    return str(ROOT / path)


@dataclass(frozen=True)
class Settings:
    yolo_model: str = resolve_model_path(os.getenv("YOLO_MODEL", "yolo11m.pt"))
    tracker: str = os.getenv("TRACKER", "botsort.yaml")
    detection_confidence: float = float(os.getenv("DETECTION_CONFIDENCE", "0.28"))
    detection_iou: float = float(os.getenv("DETECTION_IOU", "0.65"))
    image_size: int = int(os.getenv("IMAGE_SIZE", "960"))
    match_threshold: float = float(os.getenv("MATCH_THRESHOLD", "0.62"))
    min_person_height: int = int(os.getenv("MIN_PERSON_HEIGHT", "100"))
    reid_repo: str = "kaiyangzhou/osnet"
    reid_filename: str = (
        "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_"
        "coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
    )
    static_dir: Path = ROOT / "static"


settings = Settings()

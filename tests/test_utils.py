import base64

import cv2
import numpy as np

from app.reid import ReIDEngine
from app.calibration import choose_threshold
from app.config import MODEL_DIR, settings
from app.utils import crop_quality, decode_data_url, safe_crop


def test_decode_data_url_round_trip():
    image = np.full((30, 20, 3), 127, np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    url = "data:image/jpeg;base64," + base64.b64encode(encoded).decode()
    result = decode_data_url(url)
    assert result.shape == image.shape


def test_safe_crop_never_wraps_negative_coordinates():
    image = np.zeros((100, 100, 3), np.uint8)
    crop = safe_crop(image, (0, 0, 50, 60), pad=0.2)
    assert crop.shape[:2] == (72, 60)


def test_quality_rewards_clear_image():
    clear = np.zeros((200, 90, 3), np.uint8)
    clear[::4] = 255
    blurry = np.full_like(clear, 100)
    assert crop_quality(clear, 0.9) > crop_quality(blurry, 0.9)


def test_similarity_uses_best_query_and_gallery_views():
    query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    gallery = np.array([[0.99, 0.01], [-1.0, 0.0]], dtype=np.float32)
    assert ReIDEngine.similarity(query, gallery) > 0.65


def test_threshold_calibration_separates_validation_groups():
    result = choose_threshold([0.82, 0.76, 0.71], [0.22, 0.38, 0.55])
    assert result["f1"] == 1.0
    assert 0.55 < result["threshold"] <= 0.71


def test_default_yolo_weight_is_kept_in_models_directory():
    assert settings.yolo_model == str(MODEL_DIR / "yolo11m.pt")

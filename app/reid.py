from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from .config import MODEL_DIR, Settings

log = logging.getLogger(__name__)


class ReIDEngine:
    """OSNet-AIN person Re-ID with test-time flip augmentation."""

    def __init__(self, config: Settings):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractor = None
        self.dimension = 512

    def load(self) -> None:
        if self.extractor is not None:
            return
        # PyPI 0.2.5 keeps the implementation under torchreid.reid, while the
        # upstream repository exposes torchreid.utils. Supporting both avoids a
        # fragile install-from-Git requirement on Windows.
        try:
            from torchreid.utils import FeatureExtractor
        except ModuleNotFoundError:
            from torchreid.reid.utils import FeatureExtractor

        weights = hf_hub_download(
            repo_id=self.config.reid_repo,
            filename=self.config.reid_filename,
            local_dir=MODEL_DIR,
        )
        self.extractor = FeatureExtractor(
            model_name="osnet_ain_x1_0",
            model_path=weights,
            device=self.device,
            image_size=(256, 128),
        )
        log.info("Re-ID ready on %s", self.device)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)

    def embed(self, crops: list[np.ndarray], augment: bool = False) -> np.ndarray:
        if not crops:
            return np.empty((0, self.dimension), dtype=np.float32)
        self.load()
        assert self.extractor is not None
        # Torchreid accepts BGR numpy arrays and performs the resize/normalization.
        images = crops + ([cv2.flip(c, 1) for c in crops] if augment else [])
        with torch.inference_mode():
            features = self.extractor(images).detach().float().cpu().numpy()
        features = self._normalize(features)
        if augment:
            n = len(crops)
            features = self._normalize((features[:n] + features[n:]) * 0.5)
        return features.astype(np.float32)

    @staticmethod
    def similarity(query: np.ndarray, gallery: np.ndarray) -> float:
        if query.size == 0 or gallery.size == 0:
            return 0.0
        sims = gallery @ query.T
        # Robust top-k aggregation: one bad/occluded reference does not ruin a match.
        flattened = np.sort(sims.reshape(-1))
        top = flattened[-min(3, len(flattened)):]
        return float(np.clip(0.7 * top[-1] + 0.3 * top.mean(), 0.0, 1.0))

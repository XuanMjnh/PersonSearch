from collections import defaultdict, deque
from types import SimpleNamespace

import numpy as np

from app.reid import ReIDEngine
from app.search import SearchSession, SearchState


class FakeReID:
    def __init__(self, features: list[np.ndarray] | None = None):
        self.features = list(features or [])
        self.embed_calls: list[list[np.ndarray]] = []

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        self.embed_calls.append(crops)
        return np.stack([self.features.pop(0) for _ in crops])

    similarity = staticmethod(ReIDEngine.similarity)


def make_session(reid: FakeReID | None = None) -> SearchSession:
    session = SearchSession.__new__(SearchSession)
    session.config = SimpleNamespace(
        max_gallery_features=12,
        match_confirm_frames=2,
        reid_every_n_frames=1,
        target_lost_tolerance_frames=2,
        revalidate_every_n_frames=30,
        revalidate_threshold=0.5,
        revalidate_fail_limit=2,
        min_person_height=100,
    )
    session.reid = reid or FakeReID()
    session.query = np.array([[1.0, 0.0]], dtype=np.float32)
    session.frame_number = 1
    session.seen_ids = set()
    session.gallery = defaultdict(list)
    session.scores = defaultdict(lambda: deque(maxlen=7))
    session.match_streaks = defaultdict(int)
    session.best = None
    session.last_seen = {}
    session.state = SearchState.SEARCHING
    session.target_track_id = None
    session.target_score = None
    session.target_lost_frames = 0
    session.last_revalidation_frame = 0
    session.revalidate_fail_count = 0
    session.last_revalidation_score = None
    session.force_revalidation = False
    return session


def detection(track_id: int, marker: int = 0) -> dict:
    crop = np.zeros((160, 70, 3), dtype=np.uint8)
    crop[0, 0, 0] = marker
    return {
        "track_id": track_id,
        "bbox": [0, 0, 70, 160],
        "confidence": 0.95,
        "crop": crop,
    }


def test_search_requires_two_fresh_reid_confirmations_before_locking():
    reid = FakeReID([
        np.array([0.8, 0.6], dtype=np.float32),
        np.array([0.82, 0.57], dtype=np.float32),
    ])
    session = make_session(reid)
    target = detection(5)

    first = session._process_searching([target], 0.7)
    assert session.state == SearchState.SEARCHING
    assert first[0]["matched"] is False

    session.frame_number += 1
    second = session._process_searching([target], 0.7)
    assert session.state == SearchState.TRACKING
    assert session.target_track_id == 5
    assert second[0]["matched"] is True
    assert len(reid.embed_calls) == 2
    assert not session.gallery


def test_missing_search_track_breaks_confirmation_streak():
    reid = FakeReID([
        np.array([0.8, 0.6], dtype=np.float32),
        np.array([0.82, 0.57], dtype=np.float32),
    ])
    session = make_session(reid)
    target = detection(5)

    session._process_searching([target], 0.7)
    assert session.match_streaks[5] == 1
    session._process_searching([], 0.7)
    assert session.match_streaks[5] == 0

    session.frame_number += 1
    session._process_searching([target], 0.7)
    assert session.state == SearchState.SEARCHING
    assert session.match_streaks[5] == 1


def test_tracking_keeps_match_and_score_without_running_reid_each_frame():
    reid = FakeReID()
    session = make_session(reid)
    target = detection(5)
    session._lock_target(target, 0.81)
    session.frame_number += 1

    result = session._process_tracking([detection(5, marker=99)])

    assert result[0]["matched"] is True
    assert result[0]["similarity"] == 0.81
    assert result[0]["state"] == "TRACKING"
    assert reid.embed_calls == []


def test_periodic_revalidation_embeds_only_the_locked_target():
    reid = FakeReID([np.array([0.75, 0.66], dtype=np.float32)])
    session = make_session(reid)
    target = detection(5, marker=5)
    other = detection(9, marker=9)
    session._lock_target(target, 0.81)
    session.frame_number += session.config.revalidate_every_n_frames

    result = session._process_tracking([target, other])

    assert len(reid.embed_calls) == 1
    assert len(reid.embed_calls[0]) == 1
    assert reid.embed_calls[0][0][0, 0, 0] == 5
    assert result[0]["matched"] is True
    assert result[1]["matched"] is False
    assert "revalidation_score" in result[0]


def test_short_occlusion_keeps_tracking_and_revalidates_on_recovery():
    reid = FakeReID([np.array([0.8, 0.6], dtype=np.float32)])
    session = make_session(reid)
    target = detection(5)
    session._lock_target(target, 0.81)

    session._process_tracking([])
    session._process_tracking([])
    assert session.state == SearchState.TRACKING

    result = session._process_tracking([target])
    assert session.state == SearchState.TRACKING
    assert session.target_lost_frames == 0
    assert result[0]["matched"] is True
    assert len(reid.embed_calls) == 1


def test_long_absence_enters_lost_then_release_clears_track_data_only():
    session = make_session()
    query = session.query
    session._lock_target(detection(5), 0.81)
    confirmed_best = session.best

    for _ in range(session.config.target_lost_tolerance_frames + 1):
        session._process_tracking([])

    assert session.state == SearchState.LOST
    assert session.target_track_id == 5

    session._release_target()
    assert session.state == SearchState.SEARCHING
    assert session.target_track_id is None
    assert session.query is query
    assert not session.gallery
    assert not session.scores
    assert not session.match_streaks
    assert session.best is confirmed_best
    assert session.best["confirmed"] is True
    assert session.best["matched"] is True
    assert session.best["similarity"] == 0.81
    assert session.best["active"] is False


def test_best_match_keeps_highest_confirmed_result_in_history():
    session = make_session()
    session._lock_target(detection(5, marker=5), 0.81)
    highest = session.best

    session._set_best(detection(8, marker=8), 0.74, True)
    assert session.best is highest
    assert session.best["track_id"] == 5
    assert session.best["similarity"] == 0.81

    session._set_best(detection(9, marker=9), 0.88, True)
    assert session.best is not highest
    assert session.best["track_id"] == 9
    assert session.best["similarity"] == 0.88


def test_two_failed_revalidations_mark_target_lost():
    reid = FakeReID([
        np.array([0.2, 0.98], dtype=np.float32),
        np.array([0.1, 0.99], dtype=np.float32),
    ])
    session = make_session(reid)
    target = detection(5)
    session._lock_target(target, 0.81)

    session.frame_number += session.config.revalidate_every_n_frames
    first = session._process_tracking([target])
    assert session.state == SearchState.TRACKING
    assert first[0]["matched"] is True

    session.frame_number += session.config.revalidate_every_n_frames
    second = session._process_tracking([target])
    assert session.state == SearchState.LOST
    assert second[0]["matched"] is False

"""Shared helpers for the Real-Time Lip Reading Assistant."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
import platform
from typing import Iterable

import cv2
import numpy as np


VOCABULARY = [
    "idle",
    "hello",
    "yes",
    "no",
    "help",
    "stop",
    "water",
    "food",
    "medicine",
    "doctor",
    "pain",
    "emergency",
    "call",
    "thank_you",
    "please",
    "bathroom",
    "goodbye",
    "repeat",
    "slow_down",
    "more",
    "finished",
    "where",
    "when",
    "who",
    "family",
]

LABEL_TO_INDEX = {label: index for index, label in enumerate(VOCABULARY)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}

# MediaPipe Face Mesh landmark ids around the outer and inner lips.
MOUTH_LANDMARK_IDS = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    185,
    40,
    39,
    37,
    0,
    267,
    269,
    270,
    409,
    78,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
]


def get_label_maps() -> tuple[dict[str, int], dict[int, str]]:
    """Return stable label/index maps used by training and inference."""
    return LABEL_TO_INDEX.copy(), INDEX_TO_LABEL.copy()


def get_face_mesh_solution():
    """Return MediaPipe's legacy Face Mesh solution module."""
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe failed to import. If this mentions an incompatible "
            "NumPy architecture, recreate the virtual environment with the "
            "same Terminal architecture you will use to run the app, then "
            "run `python3 -m pip install -r requirements.txt`."
        ) from exc

    try:
        return mp.solutions.face_mesh
    except AttributeError as exc:
        raise RuntimeError(
            "This project uses the legacy `mp.solutions.face_mesh` API, but "
            "the installed MediaPipe package does not expose it. Reinstall "
            "the pinned dependency with `python3 -m pip install --force-reinstall "
            "-r requirements.txt`."
        ) from exc


def labels_to_maps(labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Build label maps from a runtime vocabulary."""
    label_to_index = {label: index for index, label in enumerate(labels)}
    index_to_label = {index: label for label, index in label_to_index.items()}
    return label_to_index, index_to_label


def validate_label(label: str) -> str:
    """Normalize and validate a vocabulary label."""
    normalized = label.strip().lower()
    if normalized not in LABEL_TO_INDEX:
        allowed = ", ".join(VOCABULARY)
        raise ValueError(f"Unknown label '{label}'. Allowed labels: {allowed}")
    return normalized


def extract_mouth_bbox(
    face_landmarks,
    frame_width: int,
    frame_height: int,
    padding: float = 0.35,
) -> tuple[int, int, int, int] | None:
    """Return a padded mouth bbox from Face Mesh landmarks, or None if invalid."""
    if face_landmarks is None:
        return None

    points = []
    for landmark_id in MOUTH_LANDMARK_IDS:
        landmark = face_landmarks.landmark[landmark_id]
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        points.append((x, y))

    if not points:
        return None

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    box_width = x_max - x_min
    box_height = y_max - y_min
    if box_width <= 0 or box_height <= 0:
        return None

    pad_x = int(box_width * padding)
    pad_y = int(box_height * padding)
    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(frame_width, x_max + pad_x)
    y2 = min(frame_height, y_max + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_and_resize_mouth(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    img_size: int = 96,
) -> np.ndarray | None:
    """Crop mouth ROI, convert to RGB, resize, and return uint8 image."""
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    return cv2.resize(roi_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)


def check_camera_permissions(camera_id: int = 0) -> bool:
    """Basic camera permission/availability check, useful on macOS."""
    if platform.system() != "Darwin":
        return True

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        cap.release()
        print("Error: camera permission denied or camera not available.")
        print("On macOS, enable Camera permission for Terminal, iTerm, or VS Code.")
        return False
    cap.release()
    return True


def open_camera(camera_id: int = 0, max_camera_id: int = 4) -> tuple[cv2.VideoCapture, int]:
    """Open a webcam and verify it can return frames.

    Some macOS setups expose camera id 0 but return no frames. If the requested
    id fails to read, probe nearby ids and return the first usable camera.
    """
    candidate_ids = [camera_id] + [idx for idx in range(max_camera_id) if idx != camera_id]

    for candidate_id in candidate_ids:
        cap = cv2.VideoCapture(candidate_id)
        if not cap.isOpened():
            cap.release()
            continue

        ok, _ = cap.read()
        if ok:
            return cap, candidate_id

        cap.release()

    raise RuntimeError(
        "Could not read frames from any camera id. "
        "Check macOS Camera permission and close other apps using the webcam."
    )


def normalize_clip(clip: np.ndarray) -> np.ndarray:
    """Convert clip to float32 channels-first data in [0, 1]."""
    clip = clip.astype(np.float32) / 255.0
    if clip.ndim != 4:
        raise ValueError(f"Expected clip shape (frames, height, width, channels), got {clip.shape}")
    return np.transpose(clip, (0, 3, 1, 2))


def ensure_dir(path: str | Path) -> Path:
    """Create directory if needed and return Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def next_sample_path(label_dir: Path, label: str) -> Path:
    """Return next numbered .npy path for a label directory."""
    existing = sorted(label_dir.glob(f"{label}_*.npy"))
    return label_dir / f"{label}_{len(existing):04d}.npy"


class FPSCounter:
    """Small rolling FPS calculator."""

    def __init__(self, window_size: int = 30) -> None:
        self.times: deque[float] = deque(maxlen=window_size)
        self.last_time = time.perf_counter()

    def update(self) -> float:
        now = time.perf_counter()
        delta = now - self.last_time
        self.last_time = now
        if delta > 0:
            self.times.append(delta)
        if not self.times:
            return 0.0
        return 1.0 / (sum(self.times) / len(self.times))


def most_common_prediction(predictions: Iterable[tuple[int, float]]) -> tuple[int | None, float]:
    """Return majority class and averaged confidence from recent predictions."""
    predictions = list(predictions)
    if not predictions:
        return None, 0.0

    counts: dict[int, int] = {}
    confidences: dict[int, list[float]] = {}
    for class_id, confidence in predictions:
        counts[class_id] = counts.get(class_id, 0) + 1
        confidences.setdefault(class_id, []).append(confidence)

    best_class = max(counts, key=counts.get)
    avg_confidence = float(np.mean(confidences[best_class]))
    return best_class, avg_confidence

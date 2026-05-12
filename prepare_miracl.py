"""Download/convert MIRACL-VC1 image sequences into .npy mouth clips.

Expected input is the Kaggle MIRACL-VC1 tree, usually speaker folders like
F01, F02, M01, etc. The script scans image-sequence directories, infers labels
from MIRACL word/phrase ids in path names, extracts mouth crops with Face Mesh,
resamples to fixed length, and writes data/raw/<label>/*.npy.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from utils import crop_and_resize_mouth, ensure_dir, extract_mouth_bbox, next_sample_path


MIRACL_WORDS = {
    "01": "begin",
    "02": "choose",
    "03": "connection",
    "04": "navigation",
    "05": "next",
    "06": "previous",
    "07": "start",
    "08": "stop",
    "09": "hello",
    "10": "web",
}

MIRACL_PHRASES = {
    "01": "stop_navigation",
    "02": "excuse_me",
    "03": "i_am_sorry",
    "04": "thank_you",
    "05": "goodbye",
    "06": "i_love_this_game",
    "07": "nice_to_meet_you",
    "08": "you_are_welcome",
    "09": "how_are_you",
    "10": "have_a_good_time",
}

DEFAULT_KEEP_LABELS = ["hello", "stop", "thank_you", "goodbye"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MIRACL-VC1 for this lip-reading project.")
    parser.add_argument("--dataset_dir", default=None, help="Local MIRACL-VC1 root. If omitted with --download, uses Kaggle cache.")
    parser.add_argument("--download", action="store_true", help="Download MIRACL-VC1 from Kaggle with kagglehub.")
    parser.add_argument("--kaggle_slug", default="blueguydeez8974/miracl-vc1", help="Kaggle dataset slug.")
    parser.add_argument("--output_dir", default="data/raw", help="Output directory for .npy clips.")
    parser.add_argument("--frames_per_clip", type=int, default=30)
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--max_samples_per_label", type=int, default=200)
    parser.add_argument("--labels", nargs="*", default=DEFAULT_KEEP_LABELS, help="Labels to keep after MIRACL id mapping.")
    return parser.parse_args()


def download_dataset(slug: str) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("Install kagglehub first: pip install kagglehub") from exc
    return Path(kagglehub.dataset_download(slug))


def list_image_sequence_dirs(root: Path) -> list[Path]:
    sequence_dirs = []
    for directory, _, files in os.walk(root):
        image_count = sum(1 for name in files if Path(name).suffix.lower() in IMAGE_EXTS)
        if image_count >= 4:
            sequence_dirs.append(Path(directory))
    return sequence_dirs


def infer_label(path: Path) -> str | None:
    parts = [part.lower() for part in path.parts]
    joined = "/".join(parts)

    phrase_mode = any(token in joined for token in ["phrase", "phrases", "/p"])
    word_mode = any(token in joined for token in ["word", "words", "/w"])

    numeric_tokens = []
    for part in parts:
        numeric_tokens.extend(re.findall(r"\d+", part))

    for token in reversed(numeric_tokens):
        key = token.zfill(2)[-2:]
        if phrase_mode and key in MIRACL_PHRASES:
            return MIRACL_PHRASES[key]
        if word_mode and key in MIRACL_WORDS:
            return MIRACL_WORDS[key]

    for part in parts:
        normalized = part.replace(" ", "_").replace("-", "_")
        if normalized in set(MIRACL_WORDS.values()) | set(MIRACL_PHRASES.values()):
            return normalized

    return None


def resample_frames(frames: list[np.ndarray], target_count: int) -> list[np.ndarray]:
    if not frames:
        return []
    indices = np.linspace(0, len(frames) - 1, target_count).round().astype(int)
    return [frames[index] for index in indices]


def convert_sequence(
    sequence_dir: Path,
    face_mesh,
    frames_per_clip: int,
    img_size: int,
) -> np.ndarray | None:
    image_paths = sorted(path for path in sequence_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    mouth_frames: list[np.ndarray] = []

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            continue
        bbox = extract_mouth_bbox(results.multi_face_landmarks[0], width, height)
        crop = crop_and_resize_mouth(frame, bbox, img_size)
        if crop is not None:
            mouth_frames.append(crop)

    if len(mouth_frames) < 4:
        return None
    return np.stack(resample_frames(mouth_frames, frames_per_clip))


def main() -> None:
    args = parse_args()
    dataset_dir = download_dataset(args.kaggle_slug) if args.download else Path(args.dataset_dir or "")
    if not dataset_dir.exists():
        raise RuntimeError("Dataset directory not found. Pass --dataset_dir or use --download.")

    keep_labels = set(args.labels)
    sequence_dirs = list_image_sequence_dirs(dataset_dir)
    if not sequence_dirs:
        raise RuntimeError(f"No image sequence folders found under {dataset_dir}")

    counts = {label: 0 for label in keep_labels}
    output_dir = Path(args.output_dir)

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        for sequence_dir in tqdm(sequence_dirs, desc="Converting MIRACL"):
            label = infer_label(sequence_dir)
            if label not in keep_labels:
                continue
            if counts[label] >= args.max_samples_per_label:
                continue

            clip = convert_sequence(sequence_dir, face_mesh, args.frames_per_clip, args.img_size)
            if clip is None:
                continue

            label_dir = ensure_dir(output_dir / label)
            np.save(next_sample_path(label_dir, label), clip)
            counts[label] += 1

    print("Converted samples:")
    for label in sorted(counts):
        print(f"  {label}: {counts[label]}")


if __name__ == "__main__":
    main()

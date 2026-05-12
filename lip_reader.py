"""Real-time webcam lip-reading demo."""

from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import mediapipe as mp
import numpy as np
import torch

from model import LipReadingModel
from utils import (
    FPSCounter,
    check_camera_permissions,
    VOCABULARY,
    crop_and_resize_mouth,
    extract_mouth_bbox,
    most_common_prediction,
    normalize_clip,
    open_camera,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time lip-reading demo.")
    parser.add_argument("--checkpoint", default="models/lip_reader.pt", help="Model checkpoint path.")
    parser.add_argument("--frames_per_clip", type=int, default=30, help="Rolling frames per inference.")
    parser.add_argument("--camera_id", type=int, default=0, help="OpenCV camera id.")
    parser.add_argument("--img_size", type=int, default=96, help="Mouth crop size.")
    parser.add_argument("--smooth_window", type=int, default=5, help="Number of predictions to smooth.")
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: Path, device: torch.device, img_size: int) -> tuple[LipReadingModel | None, int, int, list[str]]:
    if not checkpoint_path.exists():
        return None, 30, img_size, VOCABULARY

    checkpoint = torch.load(checkpoint_path, map_location=device)
    frames_per_clip = int(checkpoint.get("frames_per_clip", 30))
    checkpoint_img_size = int(checkpoint.get("img_size", img_size))
    vocabulary = checkpoint.get("vocabulary", VOCABULARY)
    model = LipReadingModel(num_classes=len(vocabulary), img_size=checkpoint_img_size)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, frames_per_clip, checkpoint_img_size, vocabulary


def predict_clip(model: LipReadingModel, clip_frames: deque[np.ndarray], device: torch.device) -> tuple[int, float]:
    clip = np.stack(list(clip_frames))
    clip = normalize_clip(clip)
    tensor = torch.from_numpy(clip).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        confidence, class_id = torch.max(probs, dim=0)
    return int(class_id.item()), float(confidence.item())


def main() -> None:
    args = parse_args()
    device = get_device()
    model, checkpoint_frames, checkpoint_img_size, runtime_vocab = load_model(Path(args.checkpoint), device, args.img_size)

    if model is None:
        print(f"Warning: checkpoint not found at {args.checkpoint}. Running mouth-detection demo mode.")
        frames_per_clip = args.frames_per_clip
        img_size = args.img_size
    else:
        frames_per_clip = checkpoint_frames
        img_size = checkpoint_img_size
        print(f"Loaded model from {args.checkpoint} on {device}")

    if not check_camera_permissions(args.camera_id):
        return

    cap, active_camera_id = open_camera(args.camera_id)
    if active_camera_id != args.camera_id:
        print(f"Camera id {args.camera_id} opened but returned no frames. Using camera id {active_camera_id}.")

    mp_face_mesh = mp.solutions.face_mesh
    frame_buffer: deque[np.ndarray] = deque(maxlen=frames_per_clip)
    prediction_buffer: deque[tuple[int, float]] = deque(maxlen=args.smooth_window)
    fps_counter = FPSCounter()

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: camera frame not available.")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            bbox = None
            mouth_crop = None
            if results.multi_face_landmarks:
                bbox = extract_mouth_bbox(results.multi_face_landmarks[0], width, height)
                mouth_crop = crop_and_resize_mouth(frame, bbox, img_size)

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if mouth_crop is not None:
                frame_buffer.append(mouth_crop)
                if model is not None and len(frame_buffer) == frames_per_clip:
                    prediction_buffer.append(predict_clip(model, frame_buffer, device))

            smoothed_class, smoothed_conf = most_common_prediction(prediction_buffer)
            predicted_word = runtime_vocab[smoothed_class] if smoothed_class is not None else "..."
            fps = fps_counter.update()

            cv2.putText(frame, f"Prediction: {predicted_word}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {smoothed_conf:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
            cv2.putText(frame, f"Buffer: {len(frame_buffer)}/{frames_per_clip}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

            if mouth_crop is None:
                cv2.putText(frame, "No face/mouth detected", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
            if model is None:
                cv2.putText(frame, "Demo mode: train model first", (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

            cv2.imshow("Real-Time Lip Reading Assistant", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""Collect labeled mouth-movement clips from webcam."""

from __future__ import annotations

import argparse
import os
from collections import deque

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import mediapipe as mp
import numpy as np

from utils import (
    FPSCounter,
    check_camera_permissions,
    crop_and_resize_mouth,
    ensure_dir,
    extract_mouth_bbox,
    next_sample_path,
    open_camera,
    validate_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record labeled lip-reading clips.")
    parser.add_argument("--label", required=True, help="Vocabulary label to record.")
    parser.add_argument("--num_samples", type=int, default=30, help="Number of samples to collect.")
    parser.add_argument("--frames_per_clip", type=int, default=30, help="Frames per recorded clip.")
    parser.add_argument("--camera_id", type=int, default=0, help="OpenCV camera id.")
    parser.add_argument("--img_size", type=int, default=96, help="Mouth crop size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label = validate_label(args.label)
    label_dir = ensure_dir(f"data/raw/{label}")

    if not check_camera_permissions(args.camera_id):
        return

    cap, active_camera_id = open_camera(args.camera_id)
    if active_camera_id != args.camera_id:
        print(f"Camera id {args.camera_id} opened but returned no frames. Using camera id {active_camera_id}.")

    mp_face_mesh = mp.solutions.face_mesh
    fps_counter = FPSCounter()
    recording = False
    recorded = 0
    clip_frames: deque[np.ndarray] = deque(maxlen=args.frames_per_clip)

    print("Press SPACE to record one sample. Press Q to quit.")
    print(f"Recording label '{label}' into {label_dir}")

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while recorded < args.num_samples:
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
                mouth_crop = crop_and_resize_mouth(frame, bbox, args.img_size)

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if recording:
                if mouth_crop is not None:
                    clip_frames.append(mouth_crop)

                if len(clip_frames) == args.frames_per_clip:
                    save_path = next_sample_path(label_dir, label)
                    np.save(save_path, np.stack(clip_frames))
                    recorded += 1
                    recording = False
                    clip_frames.clear()
                    print(f"Saved {save_path} ({recorded}/{args.num_samples})")

            fps = fps_counter.update()
            status = "RECORDING" if recording else "READY"
            if mouth_crop is None:
                status = "NO FACE"

            cv2.putText(frame, f"Label: {label}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Samples: {recorded}/{args.num_samples}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Status: {status}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Collect Lip Reading Data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32 and not recording:
                if mouth_crop is None:
                    print("No face/mouth detected. Move into frame before recording.")
                else:
                    recording = True
                    clip_frames.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


LANDMARK_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def landmark_to_dict(lm):
    return {
        "x": float(lm.x),
        "y": float(lm.y),
        "z": float(lm.z),
        "visibility": float(getattr(lm, "visibility", 1.0)),
        "presence": float(getattr(lm, "presence", 1.0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parent / "models" / "pose_landmarker_full.task"),
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    out_path = Path(args.out)
    model_path = Path(args.model)

    if not image_path.exists():
        raise RuntimeError(f"Image does not exist: {image_path}")

    if not model_path.exists():
        raise RuntimeError(
            "Pose Landmarker model file not found:\n"
            f"{model_path}\n\n"
            "Put pose_landmarker_full.task inside the models folder."
        )

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    height, width = image_bgr.shape[:2]

    # MediaPipe Image wants RGB.
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    with vision.PoseLandmarker.create_from_options(options) as detector:
        result = detector.detect(mp_image)

    if not result.pose_landmarks:
        data = {
            "ok": False,
            "error": "No pose detected",
            "image": str(image_path),
            "width": width,
            "height": height,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print("No pose detected.")
        return

    pose_landmarks = result.pose_landmarks[0]

    landmarks = {}
    visibility_values = []

    for i, lm in enumerate(pose_landmarks):
        name = LANDMARK_NAMES[i]
        item = landmark_to_dict(lm)
        landmarks[name] = item
        visibility_values.append(min(item["visibility"], item["presence"]))

    world_landmarks = {}

    if result.pose_world_landmarks:
        for i, lm in enumerate(result.pose_world_landmarks[0]):
            name = LANDMARK_NAMES[i]
            world_landmarks[name] = landmark_to_dict(lm)

    avg_visibility = float(np.mean(visibility_values)) if visibility_values else 0.0

    data = {
        "ok": True,
        "source_type": "image",
        "image": str(image_path),
        "width": width,
        "height": height,
        "average_visibility": avg_visibility,
        "landmarks": landmarks,
        "world_landmarks": world_landmarks,
        "model": str(model_path),
        "api": "mediapipe_tasks_pose_landmarker",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print(f"Wrote pose JSON: {out_path}")
    print(f"Average visibility: {avg_visibility:.3f}")


if __name__ == "__main__":
    main()
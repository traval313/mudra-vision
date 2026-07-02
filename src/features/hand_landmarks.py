from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LANDMARK_COUNT = 21
FEATURE_COLUMNS = [
    f"lm{landmark_id}_{axis}"
    for landmark_id in range(LANDMARK_COUNT)
    for axis in ("x", "y", "z")
]


def iter_image_paths(dataset_dir):
    dataset_dir = Path(dataset_dir)
    for image_path in sorted(dataset_dir.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield image_path


def create_hands_detector(static_image_mode=True, max_num_hands=1):
    return mp.solutions.hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=0.5,
    )


def extract_landmark_features_from_rgb_image(rgb_image, hands):
    result = hands.process(rgb_image)

    if not result.multi_hand_landmarks:
        return None

    landmarks = result.multi_hand_landmarks[0].landmark
    points = np.array([[point.x, point.y, point.z] for point in landmarks], dtype=float)

    wrist = points[0].copy()
    points -= wrist

    scale = np.max(np.linalg.norm(points[:, :2], axis=1))
    if scale > 0:
        points /= scale

    return points.reshape(-1).tolist()


def extract_landmark_features_from_image(image, hands):
    if isinstance(image, Image.Image):
        rgb_image = np.array(image.convert("RGB"))
    else:
        rgb_image = np.asarray(image)

    return extract_landmark_features_from_rgb_image(rgb_image, hands)


def extract_landmark_features(image_path, hands):
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return extract_landmark_features_from_rgb_image(rgb_image, hands)

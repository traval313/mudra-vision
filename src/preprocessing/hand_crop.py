from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from PIL import ImageOps 

def crop_hand_from_image(
    image_path: str | Path,
    padding_ratio: float = 0.35,
) -> Image.Image | None:
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    height, width, _ = image_np.shape

    mp_hands = mp.solutions.hands

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.4,
    ) as hands:
        results = hands.process(image_np)

    if not results.multi_hand_landmarks:
        return None

    landmarks = results.multi_hand_landmarks[0].landmark

    x_coords = [int(lm.x * width) for lm in landmarks]
    y_coords = [int(lm.y * height) for lm in landmarks]

    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    box_width = x_max - x_min
    box_height = y_max - y_min

    padding = int(max(box_width, box_height) * padding_ratio)

    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(width, x_max + padding)
    y_max = min(height, y_max + padding)

    cropped = image_np[y_min:y_max, x_min:x_max]

    if cropped.size == 0:
        return None

    return Image.fromarray(cropped)



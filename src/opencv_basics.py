import cv2
from pathlib import Path

IMAGE_PATH = "images/train/Bramaram(1)_Bramaram_308_b1.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None: 
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

print("Image loaded successfully!")
print("Type:", type(image))
print("Shape:", image.shape)
print("Data type:", image.dtype)

height, width, channels = image.shape
print("Height:", height)
print("Width:", width)
print("Channels:", channels)


row = height // 2
col = width // 2
pixel = image[row, col]

print(f"Center pixel at row={row}, col={col}:", pixel)
print("Remember: OpenCV uses BGR, not RGB.")


annotated = image.copy()

cv2.rectangle(
    annotated,
    (50, 50),
    (width - 50, height - 50),
    (0, 255, 0),
    3
)

cv2.circle(
    annotated,
    (width // 2, height // 2),
    20,
    (0, 0, 255),
    -1
)

cv2.putText(
    annotated,
    "MudraVision OpenCV Basics",
    (50, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 0, 0),
    2
)

Path("outputs").mkdir(exist_ok=True)
cv2.imwrite("outputs/annotated_mudra.jpg", annotated)

print("Saved annotated image to outputs/annotated_mudra.jpg")

cv2.imshow("Original Image", image)
cv2.imshow("Annotated Image", annotated)

cv2.waitKey(0)
cv2.destroyAllWindows()
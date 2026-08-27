import cv2
import numpy as np
from PIL import Image

# Must match the Kaggle dataset generation script exactly - DO NOT change these values.
MAX_DIM = 640
JPEG_QUALITY = 85


def simulate_kaggle_capture_pipeline(
    image: Image.Image,
    max_dim: int = MAX_DIM,
    jpeg_quality: int = JPEG_QUALITY,
) -> Image.Image:

    img_rgb = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ok, encoded_buffer = cv2.imencode(
        ".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    )
    if not ok:
        raise RuntimeError("Failed to encode the image as JPEG while simulating the Kaggle pipeline.")

    decoded_bgr = cv2.imdecode(encoded_buffer, cv2.IMREAD_COLOR)
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(decoded_rgb)
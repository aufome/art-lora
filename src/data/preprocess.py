import os
import cv2
from src.utils import get_logger

logger = get_logger("art_lora.preprocess")

def preprocess_images(input_dir: str, output_dir: str, target_size: int = 512):
    """
    Detects faces and crops images into target_size x target_size squares for LoRA fine-tuning.
    Each image is processed using OpenCV's Haar cascade face detector.
    """
    os.makedirs(output_dir, exist_ok=True)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    logger.info(f"Starting preprocessing: input_dir='{input_dir}', output_dir='{output_dir}'")

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        logger.warning(f"No valid image files found in '{input_dir}'.")
        return

    processed_count = 0

    for filename in image_files:
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            logger.warning(f"Skipping '{filename}': cannot read file.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            logger.warning(f"No face detected in '{filename}', skipping.")
            continue

        (x, y, w, h) = faces[0]
        center_x, center_y = x + w // 2, y + h // 2
        side_length = int(max(w, h) * 1.2)
        x1, y1 = max(0, center_x - side_length // 2), max(0, center_y - side_length // 2)
        x2, y2 = x1 + side_length, y1 + side_length

        cropped = img[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (target_size, target_size))

        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, resized)
        processed_count += 1

        logger.info(f"Processed '{filename}' --> '{output_path}'")

    logger.info(f"Preprocessing complete. {processed_count} images saved to '{output_dir}'.")

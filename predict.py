from ultralytics import YOLO
from utils import available_hardware
import numpy as np
import cv2
import os

# Load the YOLOv8 model
try:
    model = YOLO(
        "runs/classify/train/weights/best.pt"
    )  # If you have freshly trained, the model will be in this directory...
except Exception as e:
    model = YOLO(
        "models/best.pt"
    )  # ...otherwise you can already use the one I trained.

device = available_hardware(log=False)


def process_and_annotate_images(content_folder="content", output_folder="result"):

    print("-" * 40)
    print(f"Processing images from '{content_folder}' folder")
    print("-" * 40)

    # Check if content folder exists
    if not os.path.exists(content_folder):
        print(f"Error: '{content_folder}' folder not found!")
        return

    # Create output directory for annotated images
    os.makedirs(output_folder, exist_ok=True)
    # print(f"Saving annotated images to '{output_folder}' folder...")

    # Get list of image files
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
    image_files = [
        f for f in os.listdir(content_folder) if f.lower().endswith(image_extensions)
    ]

    if not image_files:
        print(f"No image files found in '{content_folder}' folder!")
        return

    # print(f"Found {len(image_files)} images to process...")

    # Make predictions on the content folder
    results = model.predict(
        source=content_folder, conf=0.25, device=device, verbose=False
    )

    # Process and annotate each result
    for result in results:
        try:
            img_path = result.path  # Get the image path
            img = cv2.imread(img_path)  # Load the image
            img_name = os.path.basename(img_path)

            if img is None:
                print(f"Warning: Could not load image {img_path}")
                continue

            print(f"-" * 40)
            print(f"Processing | {img_name} | ...")

            # Handle segmentation model results
            if (
                hasattr(result, "boxes")
                and result.boxes is not None
                and len(result.boxes) > 0
            ):
                # This is a segmentation/detection model
                for box in result.boxes:
                    # Get class and confidence
                    class_id = int(box.cls.item())
                    confidence = box.conf.item()
                    predicted_class = result.names[class_id]

                    # Since this model only detects cancer (segmentation model with 1 class)
                    print(
                        f"Cancerous tissue detected.\nConfidence: {confidence*100:.1f}%"
                    )

                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    # Create label text
                    label = f"cancer: {confidence:.2f}"

                    # Draw label background (red)
                    label_size = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )[0]
                    cv2.rectangle(
                        img,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1),
                        (0, 0, 255),
                        -1,
                    )

                    # Draw label text
                    cv2.putText(
                        img,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

            else:
                # No detections found
                label = "not detected"

                # Draw label background (green)
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(img, (0, 0), (label_size[0], 25), (0, 255, 0), -1)

                # Draw label text (white)
                cv2.putText(
                    img,
                    label,
                    (0, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )
                print(f"No cancerous tissue detected.")

            # Save the annotated image to the output folder
            filename, ext = os.path.splitext(os.path.basename(img_path))
            output_filename = f"{filename}-annotated{ext}"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, img)

        except Exception as e:
            print(f"Error processing {result.path}: {e}")

    print("\n" + "-" * 40)
    print(f"Annotated images saved in '{output_folder}' folder")
    print(f"-" * 40)


if __name__ == "__main__":
    process_and_annotate_images()

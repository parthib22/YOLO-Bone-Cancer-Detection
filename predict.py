from ultralytics import YOLO
from utils import available_hardware
import numpy as np

# Load the YOLOv8 model
try:
    model = YOLO("runs/classify/train/weights/best.pt")  # If you have freshly trained, the model will be in this directory...
except Exception as e:
    model = YOLO("models/best.pt") # ...otherwise you can already use the one I trained.

device = available_hardware(log=False)

def test_on_dataset(dataset_folder):

    print(f"Processing {dataset_folder}...")

    results = model.predict(source=dataset_folder, conf=0.25, device=device, stream=True, verbose=False)

    # Initialize counters for each class
    normal_count = 0
    cancer_count = 0

    # Process the results
    for result in results:
        try:
            # Assuming the prediction results have a 'names' attribute
            predicted_class = result.names[np.argmax(result.probs.data.cpu().numpy())]

            if predicted_class == "normal":
                normal_count += 1
            elif predicted_class == "cancer":
                cancer_count += 1

        except Exception as e:
            print(f"Error processing result: {e}")


    total_images = normal_count + cancer_count

    # Calculate the prediction percentages
    if total_images > 0:
        normal_percentage = (normal_count / total_images) * 100
        cancer_percentage = (cancer_count / total_images) * 100
    else:
        normal_percentage = 0
        cancer_percentage = 0

    print(f"Normal accuracy: {normal_percentage:.2f}%")
    print(f"Cancer accuracy: {cancer_percentage:.2f}%")

test_on_dataset("bone-cancer-detection--1/train/cancer")
test_on_dataset("bone-cancer-detection--1/train/normal")

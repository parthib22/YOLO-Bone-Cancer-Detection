# pip install ultralytics os opencv-python

import os
from ultralytics import YOLO
import cv2
import torch

# Load the YOLOv8 model
try:
    model = YOLO("runs/classify/train/weights/best.pt")  # Update with the correct path
except Exception as e:
    model = YOLO("best.pt")

dataset_folder = (
    "bone-cancer-detection--1/test/cancer"  # Update with your dataset folder path
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# prompt: predict the images using yolo v8 from dataset_folder, refer from the above codes. also find the final prediction percentage of both the classes.

import numpy as np

results = model.predict(source=dataset_folder, conf=0.25, device=device, verbose=False)

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

print(f"Normal prediction: {normal_percentage:.2f}%")
print(f"Cancer prediction: {cancer_percentage:.2f}%")

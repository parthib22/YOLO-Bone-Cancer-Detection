from ultralytics import YOLO
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load a pretrained YOLOv8 classification model
model = YOLO("yolov8n-cls.pt")  # Use a classification variant of YOLOv8

# Train the model with your custom dataset
model.train(
    data="bone-cancer-detection--1", epochs=10, imgsz=224, batch=16, device=device
)

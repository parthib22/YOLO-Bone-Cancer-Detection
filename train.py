from ultralytics import YOLO
from utils import available_hardware

device = available_hardware()

# Load a pretrained YOLOv8 classification model
model = YOLO("models/yolov8n-cls.pt") # This will automatically download the model if not found in the directory

train_dataset = "bone-cancer-detection--1"

# Train the model with your custom dataset
model.train(
    data=train_dataset, val=True, epochs=10, imgsz=224, batch=16, device=device
)

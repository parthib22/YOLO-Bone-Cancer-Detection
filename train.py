from ultralytics import YOLO
from utils import available_hardware, download_dataset

# Ensure the dataset is downloaded before training
# Skip if you already have another dataset
download_dataset()

device = available_hardware()

# Load a pretrained YOLOv11 segmentation model
model = YOLO(
    "yolov11n-seg.pt"
)  # This will automatically download the segmentation model if not found

dataset_path = "BONE-CANCER-SEGMENTATION-1"  # change the dataset path as needed

# Train the model with your custom dataset for segmentation
model.train(data=dataset_path, val=True, epochs=50, imgsz=640, batch=16, device=device)

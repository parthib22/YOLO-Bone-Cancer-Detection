# YOLO Bone Cancer Detection Project

This project uses YOLOv8 for bone cancer detection in medical images. It demonstrates how to set up, train, and use a YOLOv8 model for classifying bone images as either normal or cancerous.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Prediction](#prediction)
- [Results](#results)

## Prerequisites

- Python 3.x
- CUDA-capable GPU (recommended)

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/parthib22/yolov8_bone_cancer_detection.git
   cd yolov8-bone-cancer-detection
   ```

2. Install the required packages:
   ```
   pip install roboflow ultralytics torch numpy python-dotenv
   ```

## Usage

### Dataset

This project uses a dataset from Roboflow. To download the dataset:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("normal-bones").project("bone-cancer-detection-xa7ru")
version = project.version(1)
dataset = version.download("folder")
```

### Training

To train the YOLOv8 model:

```python
from ultralytics import YOLO

# Load a pretrained YOLOv8 classification model
model = YOLO('models/yolov8n-cls.pt')

# Train the model with your custom dataset
model.train(data='bone-cancer-detection--1', epochs=50, imgsz=224, batch=16)
```

### Prediction

To make predictions using the trained model:

```python
import os
from ultralytics import YOLO
import cv2
import numpy as np

# Load the trained model
model = YOLO('models/best.pt')

# Specify the path to your test images
dataset_folder = 'bone-cancer-detection--1/test/cancer'

# Make predictions
results = model.predict(source=dataset_folder, conf=0.25)

# Process and display results
# (See the full code for details on processing results)
```

## Results

After running the prediction on the test set, the model provides classification percentages for normal and cancerous bone images. The exact percentages may vary depending on your trained model and test set.

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

## License

This project is open-source and available under the [MIT License](LICENSE).

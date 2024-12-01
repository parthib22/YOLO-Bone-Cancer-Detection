# Bone Cancer Detection using YOLO

This project uses YOLOv8 for bone cancer detection in medical images. It demonstrates how to set up, train, and use a YOLOv8 model for classifying bone images as either normal or cancerous.

## Table of Contents

- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Get Started](#-get-started)
- [Training](#training)
- [Prediction](#prediction)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## ✅ Prerequisites

- [Python 3.8 and above](https://www.python.org/downloads/)

### Windows
- Windows 10 or 11
- CUDA-capable GPU (recommended)
- Microsoft Visual C++ Redistributable

### Linux
- Ubuntu 20.04/22.04 or compatible Linux distribution
- GCC and build essentials
- CUDA-capable GPU (recommended)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (for GPU support)

### macOS
- macOS 11 Big Sur or later
- Apple Silicon M-series or Intel-based Mac
- [Xcode Command Line Tools](https://www.freecodecamp.org/news/install-xcode-command-line-tools/)

## ⤵️ Installation

### 1. Create a project directory and open a terminal inside it.

### 2. The next step differs on your OS -

   - ### _Windows_

     **Optional**: Create a virtual environment:
      ```powershell
      python -m venv venv
      .\venv\Scripts\activate
      ```

   - ### _Linux_

      Install system dependencies:
      ```bash
      sudo apt-get update
      sudo apt-get install python3-venv python3-dev git
      ```

      **Optional**: Create and activate a virtual environment:
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```

      **Optional**: Install CUDA Toolkit for GPU support:
      ```bash
      # For Ubuntu
      wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
      sudo dpkg -i cuda-keyring_1.0-1_all.deb
      sudo apt-get update
      sudo apt-get install cuda
      ```

   - ### _macOS_

      Install Homebrew (if not already installed):
      ```bash
      /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
      ```

      Install Python via Homebrew:
      ```bash
      brew install python
      ```

      **Optional**: Create and activate a virtual environment:
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```

### 3. Install the required packages:
   ```bash
   pip install numpy torch roboflow ultralytics opencv-python
   ```

## 🚀 Get Started

### Dataset

This project uses a dataset from Roboflow. To download the dataset:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("normal-bones").project("bone-cancer-detection-xa7ru")
version = project.version(1)
dataset = version.download("folder")
```

Feel free to use any other dataset (from Roboflow, etc.), or create a dataset of your own. Just maintain the following folder structure:

```markdown
yolov8_bone_cancer_detection
├── custom_user_dataset
│   ├── test
│   │   ├── cancer
│   │   └── normal
│   ├── train
│   │   ├── cancer
│   │   └── normal
│   └── valid
│       ├── cancer
│       └── normal
```

### Training

To train the YOLOv8 model:

```python
from ultralytics import YOLO
import torch

# Use a GPU if available else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load a pretrained YOLOv8 classification model
model = YOLO('yolov8n-cls.pt')
```

Find more YOLOv8 classification models [here](https://docs.ultralytics.com/models/yolov8/#__tabbed_1_4).

```python
# Train the model with your dataset
model.train(data='bone-cancer-detection--1', epochs=50, imgsz=224, batch=16, device=device)
```

After training, a folder is created in the root directory with all the information about the training.

```markdown
runs/
└── classify/
    └── train/
        ├── weights/
        │   ├── best.pt
        │   └── last.pt
        ├── args.yaml
        ├── confusion_matrix.png
        ├── confusion_matrix_normalized.png
        ├── results.csv
        ├── results.png
        ├── train_batch0.jpg
        ├── train_batch1.jpg
        ├── train_batch2.jpg
        ├── val_batch0_labels.jpg
        ├── val_batch0_pred.jpg
        ├── val_batch1_labels.jpg
        ├── val_batch1_pred.jpg
        ├── val_batch2_labels.jpg
        └── val_batch2_pred.jpg
```

In the directory ```runs/classify/train/weights```, the models are created. We will be using the ```best.pt``` model.

### Prediction

To make predictions using the trained model:

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('path/to/best.pt')

# Specify the path to your test images
dataset_folder = 'bone-cancer-detection--1/test/cancer'

# Make predictions
results = model.predict(source=dataset_folder, conf=0.25)
```

### Results

Additionally, you can visualize and store the results as labelled images.

```python
import os
import cv2

for result in results:

    # Load the image
    img_path = result.path
    img = cv2.imread(img_path)

    # Class with the highest probability and it's confidence
    class_id, confidence = result.probs.top1, result.probs.top1conf
    
    # Get the classification label and confidence on the image
    label = f"{model.names[class_id]}: {confidence:.2f}"

    # Annotate the image with the classification result
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save the annotated image to the output folder
    output_path = os.path.join("path/to/output/folder", os.path.basename(img_path))
    cv2.imwrite(output_path, img)

    # Display the annotated image (optional)
    cv2.imshow('Annotated Image', img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```

## ⚙️ Troubleshooting

### GPU Support
- **Windows/Linux**: Ensure CUDA is correctly installed and compatible with your NVIDIA GPU (if available)
- **macOS**: For M-series Macs, ensure you're using a PyTorch version with Metal Performance Shaders (MPS) support

### Common Installation Issues
- Use a virtual environment to avoid package conflicts
- Verify Python and pip versions before installation
- Check CUDA and GPU driver compatibility

## 🚩 Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

## 🔱 License

This project is open-source and available under the [MIT License](LICENSE).

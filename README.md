# 🦴 YOLO Bone Cancer Segmentation

A deep learning project for detecting and segmenting cancerous tissue in bone X-ray images using YOLOv11 segmentation model.

## 🎯 Overview

This project implements an automated bone cancer detection and segmentation system using the latest YOLOv11 architecture. The model can analyze X-ray images of bones and identify cancerous tissue with bounding boxes and confidence scores.

## 🔬 Features

- **Automated Cancer Detection**: Detect cancerous tissue in bone X-ray images
- **Real-time Processing**: Fast inference using optimized YOLO architecture
- **Visual Annotations**: Generate annotated images with bounding boxes and confidence scores
- **GPU Acceleration**: Automatic GPU detection and utilization when available
- **Easy-to-Use**: Simple Python scripts for training and prediction

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended)

## 🚀 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/YOLO-Bone-Cancer-Segmentation.git
   cd YOLO-Bone-Cancer-Segmentation
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Roboflow API** (optional: to download dataset for training)
   - Sign up at [Roboflow](https://roboflow.com/)
   - Get your API key
   - Replace `ROBOFLOW_API_KEY` in `utils.py` with your actual API key

## 🗂 Project Structure

```
YOLO-Bone-Cancer-Segmentation/
├── train.py                    # Training script
├── predict.py                  # Prediction and
├── utils.py                    # Utility
├── requirements.txt            # Python
├── content/                    # Input images
│   ├── bone1.jpg
│   ├── bone2.jpg
│   └── ...
├── result/                     # Output
├── models/                     # Trained model
    └── best.pt
```

## 🏃‍♂️ Quick Start

### 1. Training a New Model (Optional)

```bash
python train.py
```

This will:

- Download the dataset automatically (if you dont have one)
- Train a YOLOv11 segmentation model for 50 epochs
- Save the best model weights

**Model Selection**: You can choose from 5 different YOLOv11 segmentation models in `train.py`:

- `yolov11n-seg.pt` - Nano (fastest, smallest)
- `yolov11s-seg.pt` - Small (balanced speed/accuracy)
- `yolov11m-seg.pt` - Medium (good accuracy)
- `yolov11l-seg.pt` - Large (high accuracy)
- `yolov11x-seg.pt` - Extra Large (highest accuracy, slowest)

**Epochs Configuration**: Adjust training duration based on your device:

- **CPU only**: 10-25 epochs (will be slow)
- **Basic GPU**: 25-50 epochs
- **High-end GPU**: 50-100+ epochs
- **Limited time**: 10-20 epochs for quick testing

### 2. Making Predictions

Place your X-ray images in the `content/` folder and run:

```bash
python predict.py
```

This will:

- Process all images in the `content/` folder
- Generate annotated images with detection results
- Save annotated images in the `result/` folder

## 📊 Model Details

- **Architecture**: YOLOv11 Medium Segmentation (yolov11m-seg)
- **Model Type**: SegmentationModel
- **Parameters**: 27,617,459 (27.6M)
- **Classes**: 1 (cancer)
- **Input Size**: 640x640 pixels
- **Model Stride**: [8, 16, 32]

## 🎨 Output Examples

The model generates annotated images with:

- **Red bounding boxes** around detected cancerous tissue
- **Confidence scores** showing detection certainty
- **Green labels** when no cancer is detected

## ⚙ Configuration

### Training Parameters

You can modify training parameters in `train.py`:

- `epochs`: Number of training epochs (default: 50)
- `imgsz`: Input image size (default: 640)
- `batch`: Batch size (default: 16)

### Prediction Parameters

You can modify prediction parameters in `predict.py`:

- `conf`: Confidence threshold (default: 0.25)
- Content folder path (default: "content")
- Output folder path (default: "result")

## 🔧 Hardware Requirements

### Minimum Requirements

- RAM: 8GB
- Storage: 2GB free space
- CPU: Multi-core processor

### Recommended Requirements

- RAM: 16GB+
- GPU: NVIDIA GPU with 4GB+ VRAM
- Storage: 5GB+ free space

## 📝 Usage Examples

### Basic Prediction

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO("models/best.pt")

# Make prediction
results = model.predict("path/to/xray.jpg", conf=0.25)
```

### Custom Dataset Training

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolov11n-seg.pt")

# Train with custom dataset
model.train(
    data="path/to/your/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚠ Disclaimer

This project is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified medical professionals for accurate diagnosis and treatment.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for the YOLOv11 implementation
- [Roboflow](https://roboflow.com/) for dataset management and hosting
- The medical imaging community for advancing AI in healthcare

## 📞 Contact

For questions, suggestions, or collaborations, please open an issue on GitHub.

---

⭐ If you find this project helpful, please consider giving it a star!

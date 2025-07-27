# 🦴 YOLO Bone Cancer Detection

An AI-powered bone cancer detection system using YOLOv8 segmentation models. This project can detect and localize cancerous tissue in bone X-ray images with high accuracy.

## 🎯 What This Project Does

- **Detects Cancer**: Identifies cancerous tissue in bone X-ray images
- **Localizes Tumors**: Draws bounding boxes around detected cancer regions
- **Confidence Scoring**: Provides confidence levels for each detection
- **Batch Processing**: Processes multiple images at once
- **Visual Results**: Saves annotated images with detection results

## 🚀 Quick Start (3 Simple Steps)

### Step 1: Download the Project

```bash
git clone https://github.com/parthib22/YOLO-Bone-Cancer-Detection.git
cd YOLO-Bone-Cancer-Detection
```

### Step 2: Set Up Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Run Detection

```bash
# Put your X-ray images in the 'content' folder
# Then run:
python predict.py
```

**That's it!** Your results will be in the `result` folder with detected cancer regions highlighted.

## 📁 Project Structure

```
YOLO-Bone-Cancer-Detection/
├── content/           # 📂 Put your X-ray images here
├── result/            # 📂 Processed images appear here
├── models/            # 🤖 Pre-trained AI models
│   ├── best.pt        # Segmentation model (detects cancer regions)
│   └── better.pt      # Classification model (cancer/normal)
├── predict.py         # 🔍 Main detection script
├── train.py           # 🏋️ Training script
├── main.py            # 🖼️ Visual demo script
├── utils.py           # 🛠️ Utility functions
└── requirements.txt   # 📋 Dependencies list
```

## 🖥️ How to Use

### Option 1: Quick Detection (Recommended)

1. **Add Images**: Put your bone X-ray images in the `content/` folder
2. **Run Detection**: Execute `python predict.py`
3. **View Results**: Check the `result/` folder for annotated images

### Option 2: Interactive Demo

```bash
python main.py
```

This opens a window showing each image with detections.

### Option 3: Custom Folders

```python
from predict import process_and_annotate_images

# Use custom input/output folders
process_and_annotate_images("my_images", "my_results")
```

## 📊 Sample Output

When you run the detection, you'll see:

```
----------------------------------------
Processing images from 'content' folder
----------------------------------------
----------------------------------------
Processing | bone1.jpg | ...
Cancerous tissue detected.
Confidence: 95.2%
----------------------------------------
Processing | bone2.jpg | ...
No cancerous tissue detected.
----------------------------------------
Annotated images saved in 'result' folder
----------------------------------------
```

**Visual Results:**

- 🔴 **Red boxes**: Detected cancer regions with confidence scores
- 🟢 **Green label**: "not detected" when no cancer is found
- 📁 **Saved files**: `filename-annotated.jpg` in the result folder

## ⚙️ System Requirements

### Minimum Requirements

- **OS**: Windows 10+, macOS 11+, or Ubuntu 20.04+
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum
- **Storage**: 2GB free space

### Recommended for Better Performance

- **GPU**: NVIDIA GPU with CUDA support
- **RAM**: 8GB or more
- **CPU**: Multi-core processor

### Installation by Operating System

<details>
<summary>🪟 <strong>Windows</strong></summary>

```powershell
# Install Python from python.org if not installed
# Open Command Prompt or PowerShell

# Clone project
git clone https://github.com/parthib22/YOLO-Bone-Cancer-Detection.git
cd YOLO-Bone-Cancer-Detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run detection
python predict.py
```

</details>

<details>
<summary>🐧 <strong>Linux (Ubuntu/Debian)</strong></summary>

```bash
# Install dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv git

# Clone project
git clone https://github.com/parthib22/YOLO-Bone-Cancer-Detection.git
cd YOLO-Bone-Cancer-Detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run detection
python predict.py
```

</details>

<details>
<summary>🍎 <strong>macOS</strong></summary>

```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python

# Clone project
git clone https://github.com/parthib22/YOLO-Bone-Cancer-Detection.git
cd YOLO-Bone-Cancer-Detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run detection
python predict.py
```

</details>

## 🧠 Understanding the AI Models

This project includes two types of AI models:

### 1. Segmentation Model (`models/best.pt`) - **Main Model**

- **Purpose**: Detects and localizes cancer regions
- **Output**: Bounding boxes around cancerous tissue
- **Use Case**: Detailed analysis showing exactly where cancer is located

### 2. Classification Model (`models/better.pt`) - **Alternative**

- **Purpose**: Classifies entire image as cancer/normal
- **Output**: Single prediction for the whole image
- **Use Case**: Quick screening of images

## 🔧 Troubleshooting

### Common Issues & Solutions

<details>
<summary><strong>❌ "ModuleNotFoundError" or Import Errors</strong></summary>

**Solution:**

```bash
# Make sure virtual environment is activated
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

</details>

<details>
<summary><strong>❌ "No images found in content folder"</strong></summary>

**Solution:**

1. Create a `content` folder in the project directory
2. Add your X-ray images (.jpg, .png, .jpeg, .bmp, .tiff files)
3. Make sure images are valid and not corrupted
</details>

<details>
<summary><strong>❌ GPU/CUDA Issues</strong></summary>

**Solution:**
The project automatically uses CPU if GPU is not available. For GPU support:

**Windows/Linux with NVIDIA GPU:**

```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

</details>

<details>
<summary><strong>❌ "Permission denied" errors</strong></summary>

**Solution:**

```bash
# On Linux/Mac, you might need permissions
chmod +x predict.py

# Or run with python explicitly
python predict.py
```

</details>

### Getting Help

If you encounter issues:

1. **Check the error message** - it usually tells you what's wrong
2. **Verify your Python version**: `python --version` (should be 3.8+)
3. **Make sure virtual environment is activated**
4. **Try reinstalling dependencies**: `pip install -r requirements.txt --force-reinstall`

## 🏋️ Advanced Usage

### Training Your Own Model

If you want to train with your own data:

```python
# 1. Prepare your dataset in YOLO format
# 2. Update the dataset path in train.py
# 3. Run training
python train.py
```

### Analyzing Model Performance

```python
# Analyze model metrics and performance
python utils.py
```

### Batch Processing Large Datasets

```python
from predict import process_and_annotate_images

# Process hundreds of images
process_and_annotate_images("large_dataset_folder", "results_folder")
```

## 📈 Model Performance

- **Accuracy**: High precision in detecting cancerous regions
- **Speed**: Processes images in real-time
- **Reliability**: Pre-trained on medical imaging datasets

## 🤝 Contributing

Want to improve the project?

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request

Ideas for contributions:

- Support for different image formats
- Web interface for easier use
- Mobile app version
- Additional AI models
- Performance optimizations

## 📄 License

This project is open-source under the [MIT License](LICENSE). Feel free to use, modify, and distribute.

## 🆘 Support

- **Issues**: Report bugs on [GitHub Issues](https://github.com/parthib22/YOLO-Bone-Cancer-Detection/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/parthib22/YOLO-Bone-Cancer-Detection/discussions)

---

**⚠️ Medical Disclaimer**: This tool is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified medical professionals for actual medical decisions.

**Made with ❤️ for the medical AI community**

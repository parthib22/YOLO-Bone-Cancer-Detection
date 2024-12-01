import torch

# Fetching environment variable (API key)

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

def download_dataset():
    # Using roboflow to download the dataset

    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("normal-bones").project("bone-cancer-detection-xa7ru")
    version = project.version(1)
    dataset = version.download("folder")

def available_hardware(log=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if log:
        print("GPU available: ", torch.cuda.is_available())
        print("Number of GPUs: ", torch.cuda.device_count())
        print(
            "GPU Name: ",
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU",
        )

        print("Device available to use: ", device)

    return device


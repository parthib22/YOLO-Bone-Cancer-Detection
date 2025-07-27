import torch

api_key = "ROBOFLOW_API_KEY"


def download_dataset():
    # Using roboflow to download the dataset
    import os
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(
        "modern-academy-for-engineering-and-technology-u9m8v"
    ).project("bone-cancer-segmentation")
    version = project.version(1)
    dataset = version.download("yolov8")

    print(f"Dataset downloaded successfully.")


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

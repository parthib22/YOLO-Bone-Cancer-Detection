# pip install roboflow

from roboflow import Roboflow

# pip install python-dotenv

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=api_key)
project = rf.workspace("normal-bones").project("bone-cancer-detection-xa7ru")
version = project.version(1)
dataset = version.download("folder")

# p4j33hpzHmY3bWLP2F6O

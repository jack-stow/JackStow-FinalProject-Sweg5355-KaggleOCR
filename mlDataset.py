import kagglehub
import os
import shutil

# Download latest version
path = kagglehub.dataset_download("landlord/handwriting-recognition")

print("Path to dataset files:", path)

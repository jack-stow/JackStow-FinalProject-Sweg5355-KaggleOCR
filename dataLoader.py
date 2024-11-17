import os  # Make sure os is imported here
from PIL import Image
import numpy as np

# Custom function for loading images from a specified directory
def load_images_from_directory(images_dir, image_files):
    images = []
    
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)  # This line uses os.path.join
        image = Image.open(img_path)
        image = image.convert('RGB')  # Convert to RGB if needed
        image = np.array(image)  # Convert to a numpy array
        images.append(image)
    
    return np.array(images)

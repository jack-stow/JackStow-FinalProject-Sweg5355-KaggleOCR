import os
from PIL import Image
import numpy as np
import tensorflow as tf

def load_images_from_directory_in_batches(images_dir, image_files, batch_size=32, target_size=(224, 224)):
    """Load images from a directory in batches, resizing on the fly."""
    total_images = len(image_files)
    
    for i in range(0, total_images, batch_size):
        batch_files = image_files[i:i+batch_size]  # Select a batch of files
        images = []
        
        for img_file in batch_files:
            img_path = os.path.join(images_dir, img_file)
            image = tf.io.read_file(img_path)
            image = tf.image.decode_jpeg(image, channels=3)  # Decode JPEG images
            image = tf.image.resize(image, target_size)  # Resize image
            images.append(image)
        
        # Stack images into a batch and yield
        yield np.array(images)  # Yield the batch of images
        
def cache_resized_images(images_dir, image_files, target_size=(224, 224), cache_dir='resized_images'):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        cache_path = os.path.join(cache_dir, img_file)
        
        # Check if the image has already been resized and cached
        if not os.path.exists(cache_path):
            image = Image.open(img_path)
            image = image.convert('RGB')
            image = image.resize(target_size)  # Resize image to target size
            image.save(cache_path)  # Save resized image to cache

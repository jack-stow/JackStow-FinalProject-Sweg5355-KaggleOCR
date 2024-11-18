import os
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

def load_images_from_directory_in_batches(images_dir, image_files, batch_size=256, target_size=(224, 224)):
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
        
def crop_and_resize_image(image_path, target_size=(224, 224), padding_color=(255, 255, 255)):
    """Crop significant white space and resize the image while preserving aspect ratio."""
    image = Image.open(image_path).convert('RGB')
    
    # Convert to grayscale to detect the actual content
    gray_image = image.convert('L')
    bbox = gray_image.getbbox()  # Get bounding box of non-white areas
    
    if bbox:
        image = image.crop(bbox)  # Crop to the bounding box

    # Resize with aspect ratio preserved and padding
    image = ImageOps.pad(image, target_size, color=padding_color, centering=(0.5, 0.5))
    return image

def cache_resized_images(images_dir, image_files, target_size=(224, 224), cache_dir='resized_images'):
    """Cache preprocessed images into the specified cache directory."""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    for img_file in image_files:
        
        img_path = os.path.join(images_dir, img_file)
        cache_path = os.path.join(cache_dir, img_file)
        
        # Skip if already cached
        if not os.path.exists(cache_path):
            try:
                preprocessed_image = crop_and_resize_image(img_path, target_size)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                preprocessed_image.save(cache_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
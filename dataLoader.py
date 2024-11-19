import os
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import tensorflow as tf
import cv2

        
def crop_and_resize_image(image_path, target_size=(388, 72), padding_color=(255, 255, 255)):
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


def preprocess_image(image_path, target_size=(388, 72)):
    try:
        # Read image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Could not load image")
            
        # 1. Gentle denoising while preserving edges
        denoised = cv2.fastNlMeansDenoising(image, None, h=10, searchWindowSize=21)
        
        # 2. Enhance contrast using CLAHE instead of regular histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 3. Use Otsu's thresholding instead of adaptive thresholding
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 4. Resize as the final step
        resized = cv2.resize(binary, target_size, interpolation=cv2.INTER_AREA)
        
        # 5. Optional: Remove small noise after resize
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(resized, cv2.MORPH_OPEN, kernel)
        
        return cleaned

    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}")
        return None

def load_images_from_directory_in_batches(images_dir, image_files, batch_size=256, target_size=(388, 72)):
    """Load and preprocess images in batches."""
    total_images = len(image_files)
    
    for i in range(0, total_images, batch_size):
        batch_files = image_files[i:i+batch_size]
        images = []
        
        for img_file in batch_files:
            img_path = os.path.join(images_dir, img_file)
            processed_image = preprocess_image(img_path, target_size)
            images.append(np.array(processed_image))
        
        yield np.array(images)  # Yield the batch of processed images


def cache_resized_images(images_dir, image_files, target_size=(388, 72), cache_dir='resized_images'):
    """Cache preprocessed images into the specified cache directory."""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        cache_path = os.path.join(cache_dir, img_file)
        
        # Skip if already cached
        if not os.path.exists(cache_path):
            try:
                preprocessed_image = preprocess_image(img_path, target_size)
                
                if preprocessed_image is not None:
                    # Save using OpenCV (since the image is a numpy array)
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    cv2.imwrite(cache_path, preprocessed_image)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")



def preprocess_image(image_path, target_size=(388, 72)):
    try:
        # Read image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Could not load image")
        
        # 1. Gentle Gaussian blur to reduce noise while preserving strokes
        blurred = cv2.GaussianBlur(image, (3,3), 0)
        
        # 2. Enhanced contrast using CLAHE with smaller tiles
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
        enhanced = clahe.apply(blurred)
        
        # 3. Use Otsu's thresholding to ensure pure black and white (0 and 255 values only)
        thresh_value, _ = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, binary = cv2.threshold(enhanced, thresh_value - 10, 255, cv2.THRESH_BINARY)
        
        # 4. Resize to exactly 388x72
        resized = cv2.resize(binary, target_size, interpolation=cv2.INTER_LINEAR)
        
        # 5. Connect nearby components to fix dotted lines
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
        connected = cv2.dilate(resized, kernel, iterations=1)
        connected = cv2.erode(connected, kernel, iterations=1)
        
        # 6. Final threshold to ensure pure black and white after all operations
        _, final = cv2.threshold(connected, 127, 255, cv2.THRESH_BINARY)
        
        # Verify dimensions
        height, width = final.shape
        assert height == 72 and width == 388, f"Incorrect dimensions: {width}x{height}"
        
        # Verify binary (only 0 and 255 values)
        assert set(np.unique(final)).issubset({0, 255}), "Image is not pure black and white"
        
        return final

    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}")
        return None
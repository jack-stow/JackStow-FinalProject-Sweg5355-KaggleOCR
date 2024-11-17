import os
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical

def load_data(images_dir, path):
    
    csv_path = os.path.join(path, 'written_name_train_v2.csv')
    images_dir = os.path.join(path, images_dir)
    # Load CSV
    df = pd.read_csv(csv_path)
    
    images = []
    labels = []
    
    for index, row in df.iterrows():
        # Load image
        img_path = os.path.join(images_dir, row['FILENAME'])
        img = Image.open(img_path)
        
        # Resize image to a fixed size (e.g., 32x32 or 64x64)
        img = img.resize((64, 64))  # You can adjust the size
        img = np.array(img) / 255.0  # Normalize pixel values
        
        images.append(img)
        
        # Prepare label (convert to integer encoding or one-hot encoding)
        label = row['IDENTITY']
        labels.append(label)
    
    return np.array(images), labels

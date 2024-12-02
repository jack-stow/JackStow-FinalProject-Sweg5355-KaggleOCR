import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model as keras_load_model, Model as KerasModel
from keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from modelBuilder import create_model
from typing import Union  # For Python < 3.10
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def directory_to_df(path: str, numbers_only=True):
    """
    Retrieve all images from targeted folder, filtering for numbers if specified.
    
    Arguments:
    path: String -> the main folder directory containing image folders
    numbers_only: Boolean -> whether to include only number classes
    
    Returns:
    DataFrame: contains the images path and label corresponding to every image
    """
    df = []
    
    # If numbers_only, only process numeric folder names
    valid_classes = [str(i) for i in range(10)] if numbers_only else None
    
    for cls in os.listdir(path):
        # Skip if numbers_only is True and cls is not a number
        if valid_classes and cls not in valid_classes:
            continue
        
        cls_path = os.path.join(path, cls)
        
        # Ensure it's a directory
        if not os.path.isdir(cls_path):
            continue
        
        for img_path in os.listdir(cls_path):
            full_img_path = os.path.join(cls_path, img_path)
            df.append([full_img_path, cls])
    
    df = pd.DataFrame(df, columns=['image', 'label'])
    print("Number of samples found:", len(df))
    return df

def preprocess_data(main_path='dataset', img_shape=(32, 32), test_size=0.3, val_size=0.25, random_state=41):
    """
    Preprocess the dataset by splitting into train, validation, and test sets.
    
    Arguments:
    main_path: String -> path to the dataset
    img_shape: Tuple -> target image shape
    test_size: Float -> proportion of test set
    val_size: Float -> proportion of validation set from training data
    random_state: Int -> random seed for reproducibility
    
    Returns:
    Tuple of generators and class information
    """
    # Read the dataset
    df = directory_to_df(main_path, False)
    
    # Verify the data
    print("Dataset distribution:")
    print(df['label'].value_counts())
    
    # Split into train and test
    X, y = df['image'], df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train)
    
    # Create dataframes
    training_df = pd.concat([X_train, y_train], axis=1)
    validation_df = pd.concat([X_val, y_val], axis=1)
    testing_df = pd.concat([X_test, y_test], axis=1)
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Normalize pixel values
        rotation_range=10,  # Random rotation
        width_shift_range=0.1,  # Horizontal shift
        height_shift_range=0.1,  # Vertical shift
        shear_range=0.1,  # Shear intensity
        zoom_range=0.1,  # Random zoom
        horizontal_flip=False,  # No horizontal flipping for numbers
        fill_mode='nearest'
    )
    
    # Validation and test data generator (only rescaling)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_gen = train_datagen.flow_from_dataframe(
        training_df, 
        x_col='image', 
        y_col='label', 
        target_size=img_shape, 
        batch_size=128, 
        class_mode='categorical'  # Changed to categorical
    )
    
    val_gen = val_test_datagen.flow_from_dataframe(
        validation_df, 
        x_col='image', 
        y_col='label', 
        target_size=img_shape, 
        batch_size=128, 
        class_mode='categorical',
        shuffle=False
    )
    
    test_gen = val_test_datagen.flow_from_dataframe(
        testing_df, 
        x_col='image', 
        y_col='label', 
        target_size=img_shape, 
        batch_size=128, 
        class_mode='categorical',
        shuffle=False
    )
    
    # Get class information
    mapping = train_gen.class_indices
    mapping_inverse = {v: k for k, v in mapping.items()}
    num_classes = len(mapping_inverse)
    
    print("\nClass Mapping:")
    print(mapping_inverse)
    print("\nNumber of classes:", num_classes)
    
    return train_gen, val_gen, test_gen, mapping_inverse, num_classes

# Usage
train_gen, val_gen, test_gen, mapping, num_classes = preprocess_data('dataset')


def train_model(file_name: Union[str, None] = None):
    model: KerasModel  # Explicit type annotation for Pylance
    
    if file_name is not None and os.path.exists(file_name):
        #import the model
        model = keras_load_model(file_name) # type: ignore
    else:
        # Create the model
        model = create_model((32, 32, 3), num_classes, learning_rate=0.001)


    # Modify fit parameters
    history = model.fit(
        train_gen, 
        epochs=100,  # Increased epochs
        validation_data=val_gen,
        callbacks=[
            # Each epoch, if the current model iteration is the best so far, save it. otherwise, don't bother.
            ModelCheckpoint(file_name, save_best_only=True),
            # if 10 epochs pass without improvement, give up.
            EarlyStopping(patience=10, restore_best_weights=True),
            # Reduce learning rate when a plateau is reached.
            ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=2, min_lr=1e-7)
        ]
    )
    
    
if __name__ == '__main__':
    train_model(file_name='models/neo_character_model_v5.h5')

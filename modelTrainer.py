import os
import pandas as pd
import tensorflow as tf
from dataLoader import load_images_from_directory_in_batches, cache_resized_images
from modelBuilder import create_model, save_model
from labelEncoder import encode_labels
import pickle
from mlDataset import path as DATASET_PATH
import numpy as np
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy

MODEL_FILENAME = "models/OCR_v1.h5"

# Function to load images from the specified directory and match them with labels from the CSV
def load_data_from_split(images_dir, labels_csv_filename, dataset_path):
    # Prepend the dataset path to the image and CSV files
    labels_csv_path = os.path.join(dataset_path, labels_csv_filename)
    labels_df = pd.read_csv(labels_csv_path)
    
    # Print out column names to inspect
    print("Columns in the CSV:", labels_df.columns)
    
    # Construct the full path for images
    image_files = labels_df['FILENAME'].values
    images_dir_path = os.path.join(dataset_path, images_dir)

    return images_dir_path, image_files, labels_df

# Function to create a data generator for batches
def data_generator(images_dir, image_files, labels, batch_size=32):
    while True:  # Loop forever so that the generator can be used by the model's fit method
        # Load a batch of images
        images = next(load_images_from_directory_in_batches(images_dir, image_files, batch_size))
        
        # Ensure the images are NumPy arrays
        images = np.array(images)
        
        # Extract the corresponding labels for the batch of images
        batch_labels = labels[labels['FILENAME'].isin(image_files[:len(images)])]['IDENTITY'].values
        
        # Encode labels
        encoded_labels, _ = encode_labels(batch_labels)
        
        # Ensure labels are NumPy arrays
        encoded_labels = np.array(encoded_labels)
        
        # Return images and labels as NumPy arrays
        yield images, encoded_labels




# Enable memory growth for GPUs (if available)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth for GPUs is enabled.")
    except RuntimeError as e:
        print("Error setting memory growth:", e)

# Set TensorFlow log level to display all logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0 = display all logs

# Check if a pickled model exists
def train_and_save_model():
    print("Dataset path:", DATASET_PATH)
    
    # Load and preprocess the data
    images_dir_path, image_files, labels_df = load_data_from_split('train_v2/train/', 'written_name_train_v2.csv', DATASET_PATH)
    validation_images_dir, validation_files, validation_labels_df = load_data_from_split('validation_v2/validation/', 'written_name_validation_v2.csv', DATASET_PATH)
    test_images_dir, test_files, test_labels_df = load_data_from_split('test_v2/test/', 'written_name_test_v2.csv', DATASET_PATH)
    
    # Cache resized images for efficiency
    cache_dir = 'resized_images/train'
    cache_resized_images(images_dir_path, image_files, target_size=(224, 224), cache_dir=cache_dir)
    
    test_cache_dir = 'resized_images/test'
    cache_resized_images(test_images_dir, test_files, target_size=(224, 224), cache_dir=test_cache_dir)
    
    validation_cache_dir = 'resized_images/validation'
    cache_resized_images(validation_images_dir, validation_files, target_size=(224, 224), cache_dir=validation_cache_dir)

    # Now, use the cache directory for the data generator
    train_images_dir = cache_dir
    validation_images_dir = validation_cache_dir

    # Encode labels for training (use the 'IDENTITY' column)
    encoded_labels, label_encoder = encode_labels(labels_df['IDENTITY'])  # Correct column name

    # Define the model
    model = create_model((224, 224, 3), len(label_encoder.classes_))  # Example image shape (224, 224, 3)

    # **Compile the model before training**
    model.compile(
        optimizer=Adam(),  # Adam optimizer
        loss=SparseCategoricalCrossentropy(from_logits=False),  # For multi-class classification
        metrics=[SparseCategoricalAccuracy()]  # Accuracy as a metric
    )

    # Train the model using the resized images from the cache
    with tf.device('/GPU:0'):  # This block runs on the GPU if available
        model.fit(
            data_generator(train_images_dir, image_files, labels_df),  # Use the data generator with cached images
            steps_per_epoch=len(image_files) // 32,  # Number of batches per epoch
            validation_data=data_generator(validation_images_dir, validation_files, validation_labels_df),
            validation_steps=len(validation_files) // 32,
            epochs=2,
            verbose=2
        )

    # Save the trained model to a pickle file
    save_model(model, MODEL_FILENAME)
    print(f"Model trained and saved to {MODEL_FILENAME}")


    
# If the model file doesn't exist, train a new one
if not os.path.exists(MODEL_FILENAME):
    train_and_save_model()
else:
    print(f"Model loaded from {MODEL_FILENAME}")

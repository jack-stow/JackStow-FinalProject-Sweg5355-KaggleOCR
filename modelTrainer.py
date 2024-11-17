import os
import pandas as pd
from tensorflow.keras.utils import to_categorical
from dataLoader import load_images_from_directory  # Custom function for loading images
from modelBuilder import create_model, save_model
from labelEncoder import encode_labels
import pickle
from mlDataset import path as DATASET_PATH

MODEL_FILENAME = "trained_model.pkl"

# Function to load images from the specified directory and match them with labels from the CSV
def load_data_from_split(images_dir, labels_csv, dataset_path):
    # Prepend the dataset path to the image and CSV files
    labels_csv_path = os.path.join(dataset_path, labels_csv)
    labels_df = pd.read_csv(labels_csv_path)
    
    # Construct the full path for images
    image_files = labels_df['FILENAME'].values
    images_dir_path = os.path.join(dataset_path, images_dir)

    # Load the images from the directory
    images = load_images_from_directory(images_dir_path, image_files)
    
    return images, labels

# Check if a pickled model exists
def train_and_save_model():
    
    print("Dataset path:", DATASET_PATH)  # You can use this to verify the path is correct
    
    # Load and preprocess the data
    train_images, train_labels = load_data_from_split('train_v2/train/', 'written_name_train_v2.csv', DATASET_PATH)
    validation_images, validation_labels = load_data_from_split('validation_v2/validation/', 'written_name_validation_v2.csv', dataset_path)

    # Encode labels (for categorical output)
    encoded_labels, label_encoder = encode_labels(train_labels)

    # One-hot encode the labels
    train_labels_one_hot = to_categorical(encoded_labels, num_classes=len(label_encoder.classes_))

    # Encode validation labels (same encoder)
    encoded_val_labels = label_encoder.transform(validation_labels)
    validation_labels_one_hot = to_categorical(encoded_val_labels, num_classes=len(label_encoder.classes_))

    # Define the model
    model = create_model(train_images.shape[1:], len(label_encoder.classes_))

    # Train the model
    model.fit(train_images, train_labels_one_hot, validation_data=(validation_images, validation_labels_one_hot), epochs=10, batch_size=32)

    # Save the trained model to a pickle file
    save_model(model, MODEL_FILENAME)
    print(f"Model trained and saved to {MODEL_FILENAME}")
    
# If the model file doesn't exist, train a new one
if not os.path.exists(MODEL_FILENAME):
    train_and_save_model()
else:
    print(f"Model loaded from {MODEL_FILENAME}")

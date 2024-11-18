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
from keras import mixed_precision
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.schedules import ExponentialDecay




# Enable mixed-precision training for efficiency
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

MODEL_FILENAME = "models/OCR_v9.h5"

def get_exponential_decay_lr_schedule(initial_learning_rate, decay_steps, end_learning_rate):
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=(end_learning_rate / initial_learning_rate),  # Small decay rate to approximate linearity
        staircase=True
    )


# Function to load images and corresponding labels
def load_data_from_split(images_dir, labels_csv_filename, dataset_path):
    labels_csv_path = os.path.join(dataset_path, labels_csv_filename)
    labels_df = pd.read_csv(labels_csv_path)
    print("Columns in the CSV:", labels_df.columns)  # Debugging aid
    images_dir_path = os.path.join(dataset_path, images_dir)
    image_files = labels_df['FILENAME'].values
    return images_dir_path, image_files, labels_df

# Data generator for training batches
def data_generator(images_dir, image_files, labels, batch_size=32):
    while True:
        images = next(load_images_from_directory_in_batches(images_dir, image_files, batch_size))
        images = np.array(images)
        batch_labels = labels[labels['FILENAME'].isin(image_files[:len(images)])]['IDENTITY'].values
        encoded_labels, _ = encode_labels(batch_labels)
        yield images, np.array(encoded_labels)

def custom_data_generator_as_dataset(images_dir, image_files, labels, batch_size=32, target_size=(224, 224)):
    """
    Generator that yields batches of processed images from the correct directory.
    """
    while True:
        for start in range(0, len(image_files), batch_size):
            end = start + batch_size
            batch_files = image_files[start:end]
            images = []
            for filename in batch_files:
                # Use images from the resized_images directory
                filepath = os.path.join(images_dir, filename)
                try:
                    # Ensure images are loaded with correct target size
                    image = tf.keras.utils.load_img(filepath, target_size=target_size)
                    image = tf.keras.utils.img_to_array(image)
                    images.append(image)
                except Exception as e:
                    print(f"Error loading image {filename}: {e}")
            images = np.array(images)  # Convert list of images to a NumPy array
            batch_labels = labels.loc[labels['FILENAME'].isin(batch_files), 'IDENTITY']
            encoded_labels, _ = encode_labels(batch_labels)
            yield images, np.array(encoded_labels)

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth for GPUs is enabled.")
    except RuntimeError as e:
        print("Error setting memory growth:", e)

# Function to train and save the model
def train_and_save_model(batch_size=32):
    print("Dataset path:", DATASET_PATH)

    # Load datasets
    train_images_dir, train_files, train_labels_df = load_data_from_split(
        'train_v2/train/', 'written_name_train_v2.csv', DATASET_PATH)
    validation_images_dir, validation_files, validation_labels_df = load_data_from_split(
        'validation_v2/validation/', 'written_name_validation_v2.csv', DATASET_PATH)
    test_images_dir, test_files, test_labels_df = load_data_from_split(
        'test_v2/test/', 'written_name_test_v2.csv', DATASET_PATH)

    # Cache resized images
    cache_resized_images(train_images_dir, train_files, target_size=(224, 224), cache_dir='resized_images/train')
    cache_resized_images(validation_images_dir, validation_files, target_size=(224, 224), cache_dir='resized_images/validation')
    cache_resized_images(test_images_dir, test_files, target_size=(224, 224), cache_dir='resized_images/test')

    # Encode labels
    _, label_encoder = encode_labels(train_labels_df['IDENTITY'])

    # Create datasets with prefetch optimization
    train_dataset = custom_data_generator_as_dataset(
        train_images_dir, train_files, train_labels_df, batch_size=batch_size, target_size=(224, 224)
    )
    validation_dataset = custom_data_generator_as_dataset(
        validation_images_dir, validation_files, validation_labels_df, batch_size=batch_size, target_size=(224, 224)
    )

    # Calculate steps per epoch
    steps_per_epoch = len(train_files) // batch_size
    validation_steps = len(validation_files) // batch_size

    # Build the model
    model = create_model((224, 224, 3), len(label_encoder.classes_))

    # Set up the learning rate schedule
    lr_schedule = get_exponential_decay_lr_schedule(initial_learning_rate=0.01, decay_steps=100000, end_learning_rate=0.0001)

    # Compile the model with the linear decay learning rate
    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss=SparseCategoricalCrossentropy(from_logits=False),
        metrics=[SparseCategoricalAccuracy()]
    )

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(MODEL_FILENAME, save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    tensorboard_callback = TensorBoard(log_dir="logs", histogram_freq=1)

    print("TRAINING START")
    # Train the model
    with tf.device('/GPU:0'):  # Use GPU if available
        model.fit(
            train_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_dataset,
            validation_steps=validation_steps,
            epochs=20,
            verbose=2,
            callbacks=[checkpoint_callback, early_stopping, tensorboard_callback]
        )

    save_model(model, MODEL_FILENAME)
    print(f"Model trained and saved to {MODEL_FILENAME}")


# Train the model if it doesn't already exist
if not os.path.exists(MODEL_FILENAME):
    train_and_save_model(8)
else:
    print(f"Model loaded from {MODEL_FILENAME}")


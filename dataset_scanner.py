"""_summary_

    This script scans my dataset, compares each item against the model, and then evaluates how good the model is at predicting that value.
    This allows me to cull bad data. (some of the images are misclassified, or are horrendously drawn)
    
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import traceback


# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU Configuration Error: {e}")

class DatasetProcessor:
    def __init__(self, model_path, mapping_path, dataset_path, batch_size=64):
        self.model_path = model_path
        self.mapping_path = mapping_path
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        
        # Load model and mapping
        self.model = self._load_model()
        self.mapping = self._load_mapping()
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}

    def _load_model(self):
        try:
            model = load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            raise

    def _load_mapping(self):
        try:
            with open(self.mapping_path, 'r') as f:
                mapping = json.load(f)
                return {int(key): value for key, value in mapping.items()}
        except Exception as e:
            print(f"Error loading mapping: {e}")
            traceback.print_exc()
            raise

    def preprocess_images(self, image_paths, img_shape=(32, 32)):
        processed_images = []
        valid_paths = []
        
        for image_path in image_paths:
            try:
                # Open and convert image
                img = Image.open(image_path).convert('RGB')
                
                # Resize with LANCZOS interpolation
                img = ImageOps.fit(img, img_shape, Image.LANCZOS)
                
                # Convert to numpy and normalize
                img_array = np.array(img) / 255.0
                
                processed_images.append(img_array)
                valid_paths.append(image_path)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        return np.array(processed_images), valid_paths

    def process_class_folder(self, class_folder):
        class_path = os.path.join(self.dataset_path, class_folder)
        
        # Validate class folder
        if not os.path.isdir(class_path):
            print(f"Skipping non-directory: {class_path}")
            return []

        # Get class index
        class_index = self.reverse_mapping.get(class_folder)
        if class_index is None:
            print(f"No mapping found for class: {class_folder}")
            return []

        # Load existing checkpoint
        checkpoint_file = f"{class_folder}_checkpoint.json"
        
        # Initialize an empty list for scores
        confidence_scores = []
        completed_images = set()

        # Try to load existing checkpoint if it exists
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    existing_scores = json.load(f)
                    confidence_scores = existing_scores
                    completed_images = {entry[0] for entry in existing_scores}
                print(f"Loaded existing checkpoint for {class_folder} with {len(existing_scores)} entries")
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error reading checkpoint for {class_folder}: {e}")
                # If there's an error, we'll process the entire folder

        # Get unprocessed images
        image_list = [
            os.path.join(class_path, img) 
            for img in os.listdir(class_path) 
            if os.path.join(class_path, img) not in completed_images
        ]

        # Process in batches
        for i in range(0, len(image_list), self.batch_size):
            batch = image_list[i:i+self.batch_size]
            
            try:
                # Preprocess batch
                processed_images, valid_paths = self.preprocess_images(batch)
                
                # Skip empty batches
                if len(processed_images) == 0:
                    continue

                # Predict batch
                predictions = self.model.predict(processed_images, verbose=0)
                
                # Collect confidence scores
                batch_scores = [
                    (valid_paths[j], float(predictions[j][class_index]))
                    for j in range(len(valid_paths))
                ]
                
                confidence_scores.extend(batch_scores)

                # Periodic checkpointing
                if len(confidence_scores) % (self.batch_size * 10) == 0:
                    self._save_checkpoint(checkpoint_file, confidence_scores)
                    print(f"Checkpoint: Processed {len(confidence_scores)} images for {class_folder}")

            except Exception as e:
                print(f"Batch processing error for {class_folder}: {e}")
                traceback.print_exc()

        # Final checkpoint
        self._save_checkpoint(checkpoint_file, confidence_scores)
        print(f"Completed processing {class_folder}: {len(confidence_scores)} images")
        
        return confidence_scores

    def _save_checkpoint(self, checkpoint_file, confidence_scores):
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(confidence_scores, f, indent=4)
        except Exception as e:
            print(f"Error saving checkpoint {checkpoint_file}: {e}")

    def cull_bottom_percentile(self, bottom_percentage=5):
        for class_folder in os.listdir(self.dataset_path):
            checkpoint_file = f"{class_folder}_checkpoint.json"
            
            if not os.path.exists(checkpoint_file):
                continue

            try:
                with open(checkpoint_file, 'r') as f:
                    confidence_scores = json.load(f)

                if confidence_scores:
                    confidence_scores.sort(key=lambda x: x[1])
                    cutoff_index = max(1, int(len(confidence_scores) * (bottom_percentage / 100)))
                    culled_scores = confidence_scores[cutoff_index:]

                    with open(checkpoint_file, 'w') as f:
                        json.dump(culled_scores, f, indent=4)

                print(f"Processed bottom {bottom_percentage}% for {class_folder}")
            
            except Exception as e:
                print(f"Error culling {class_folder}: {e}")

    def process_dataset(self):
        # Process each class sequentially to avoid multiprocessing complexities
        for class_folder in sorted(os.listdir(self.dataset_path)):
            if os.path.isdir(os.path.join(self.dataset_path, class_folder)):
                try:
                    self.process_class_folder(class_folder)
                except Exception as e:
                    print(f"Error processing {class_folder}: {e}")
                    traceback.print_exc()

        # Optional culling step
        #self.cull_bottom_percentile()

def main():
    processor = DatasetProcessor(
        model_path='models/neo_character_model_v4.h5',
        mapping_path='neo_character_model_mapping.json',
        dataset_path='dataset/',
        batch_size=64  # Adjust based on your GPU memory
    )
    
    processor.process_dataset()

if __name__ == "__main__":
    main()
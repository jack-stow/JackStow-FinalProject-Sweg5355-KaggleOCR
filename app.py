import flask
from flask import Flask, render_template, request, jsonify
import json
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import os
import pickle
from keras.preprocessing import image
from keras.models import load_model as keras_load_model
#from modelTrainer import mapping
#from prototyping2 import mapping as mapping_inverse
#from modelBuilder import load_model
#from modelTrainer import train_and_save_model, MODEL_FILENAME
#from test2 import load_model#, MODEL_FILENAME
#from EMNIST_test import MODEL_FILENAME
import keras_ocr
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def save_model(model, filename):
    model.save(filename)

def load_model(filename):
    model = keras_load_model(filename)
    return model


import base64
import io
from PIL import Image
import numpy as np

def preprocess_image_for_prediction(image_data, target_size=(32, 32)):
    """
    Preprocess a Base64-encoded image string to match the model's input requirements.
    
    Parameters:
    - image_data: Base64-encoded image string (e.g., "data:image/png;base64,...")
    - target_size: Tuple of target image dimensions (default matches model training)
    
    Returns:
    - Preprocessed numpy array ready for model prediction
    """
    
    # Check and remove the prefix "data:image/<type>;base64," if it exists
    if image_data.startswith("data:image"):
        header, image_data = image_data.split(",", 1)
    
    # Decode the Base64 string to bytes
    image_bytes = base64.b64decode(image_data)
    
    # Read the image bytes using PIL
    img = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if it's not already
    img = img.convert('RGB')
    
    # Resize the image
    img = img.resize(target_size, Image.LANCZOS) # type: ignore
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Normalize the image (match training preprocessing)
    img_array = img_array / 255.0
    
    # Expand dimensions to match model input shape (add batch dimension)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array



# Preprocessing function for consistency
def preprocess_image(image_path_or_bytes):
    """
    Preprocess an image for the EMNIST model.
    :param image_path_or_bytes: Path to the image file or image bytes (for Flask).
    :return: Preprocessed image as a NumPy array.
    """
    # Load image
    if isinstance(image_path_or_bytes, str):  # File path
        img = Image.open(image_path_or_bytes)
    else:  # Image bytes (e.g., from Flask request)
        img = Image.open(BytesIO(image_path_or_bytes))

    # Check the mode before any conversion
    print(f"Original Image Mode: {img.mode}")

    # If the image has an alpha channel (transparency), handle it
    if img.mode == 'RGBA':
        # Convert transparent pixels to white (255)
        img = img.convert("RGBA")  # Ensure it's RGBA
        new_data = []
        for item in img.getdata():
            # Change all transparent pixels to white (255, 255, 255)
            if item[3] == 0:
                new_data.append((255, 255, 255, 255))  # White background for transparency
            else:
                new_data.append(item)
        img.putdata(new_data)

        # Now convert to RGB (removes alpha channel)
        img = img.convert('RGB')
    
    # Check pixel values before grayscale conversion
    img_array_before_grayscale = np.array(img)
    print(f"Pixel values before convert('L'): {np.unique(img_array_before_grayscale)}")

    # Convert to grayscale ('L')
    img = img.convert('L')  # Convert to grayscale ('L' mode)
    img_array_after_grayscale = np.array(img)
    print(f"Pixel values after convert('L'): {np.unique(img_array_after_grayscale)}")

    # Resize, normalize, and reshape
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    
    # Double-check the shape and the content of img_array
    print(f"Shape after resizing and normalizing: {img_array.shape}")
    print(f"Pixel values after reshaping: {np.unique(img_array)}")
    
    # Ensure we are reshaping the array correctly
    img_array = img_array.reshape(1, 28, 28, 1)  # Add batch dimension and channel
    
    # Double-check the reshaped array shape
    print(f"Shape after reshaping: {img_array.shape}")

    return img_array




app = Flask(__name__)

MODEL_FILENAME = "neo_character_model_v3.h5"

# Function to load the model when the server starts
def load_trained_model():
    if os.path.exists(MODEL_FILENAME):
        # If the model file exists, load it
        model = load_model(MODEL_FILENAME)
        print("Model loaded successfully")
    else:
        print("Model not found. Ensure the model is trained and saved at:", MODEL_FILENAME)
        raise FileNotFoundError(f"Model file not found at {MODEL_FILENAME}")
    return model

# Initialize the model
model = load_trained_model()
# with open('ocr_mapping.json', 'r') as f:
#     mapping = json.load(f)

with open('neo_character_model_mapping.json', 'r') as f:
    mapping_json = json.load(f)
    mapping = {int(key): value for key, value in mapping_json.items()}

@app.route('/')
def home():
    return render_template(template_name_or_list='home.html', user_name='Jack')  # Render the home page

@app.route('/submit', methods=['POST'])
def submit_drawing():
    image_data = request.form['image']  # Get the image data from the POST request

    try:
        # Preprocess the image
        processed_image = preprocess_image_for_prediction(image_data)
        
        # Make prediction
        prediction = model.predict(processed_image)  # type: ignore

        # Check if prediction is a 2D array (batch of 1)
        if isinstance(prediction, np.ndarray) and prediction.ndim == 2:
            prediction = prediction[0]  # Extract the first element from the batch

        # Sort the predictions and get indices of top probabilities
        sorted_indices = np.argsort(prediction)[::-1]  # Indices of sorted predictions in descending order

        # Initialize the cumulative sum and selected predictions
        cumulative_confidence = 0
        selected_predictions = []
        max_results = 10  # Set a limit on the number of results
        min_confidence = 0.0075  # 0.75% threshold. (do not return any results below this confidence)

        # Iterate through the sorted predictions and accumulate until reaching 99% confidence or 10 results
        for idx in sorted_indices:
            confidence = prediction[idx]
            
            # Only include predictions that meet the 0.75% confidence threshold
            if confidence < min_confidence:
                continue
            
            cumulative_confidence += confidence
            selected_predictions.append({
                'label': mapping[int(idx)], 
                'confidence': float(confidence)
            })
            
            # Stop once cumulative confidence reaches at least 99% or we've added 10 results
            if cumulative_confidence >= 0.99 or len(selected_predictions) >= max_results:
                break

        # Get the most probable class (first item in the selected predictions)
        predicted_class = int(sorted_indices[0])  # The most probable class
        predicted_label = mapping[predicted_class]

        # Prepare the response data
        response = {
            'prediction': predicted_label,
            'confidence': float(np.max(prediction)),  # Max confidence
            'top_predictions': selected_predictions  # Selected predictions with their confidence
        }
        
        # Debugging print to log the response before returning
        print("Prediction response:", response)
        
        # Return the response as JSON
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True, port=9090)

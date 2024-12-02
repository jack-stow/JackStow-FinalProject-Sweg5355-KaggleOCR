import flask
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import json
import base64
import io
from PIL import Image, ImageOps
import numpy as np
import cv2
import os
import pickle
from keras.preprocessing import image
from keras.models import load_model as keras_load_model
import keras_ocr
import tensorflow as tf
from PIL import Image, ImageOps

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def save_model(model, filename):
    model.save(filename)

def load_model(filename):
    model = keras_load_model(filename)
    return model



def preprocess_image_for_prediction(image_file, target_size=(32, 32)):
    try:
        if isinstance(image_file, str):
            if image_file.startswith("data:image"):
                header, image_data = image_file.split(",", 1)
            else:
                image_data = image_file
            
            image_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(image_bytes))
        
        elif hasattr(image_file, 'read'):
            img = Image.open(image_file)
        
        else:
            raise ValueError("Unsupported image input type")
        
        # Convert to RGB instead of grayscale
        img = img.convert('RGB')
        
        # Resize with anti-aliasing
        img = img.resize(target_size, Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Normalize the image
        img_array = img_array / 255.0
        
        # Expand dimensions to match model input shape (batch, height, width, channels)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise


def preprocess_image_for_prediction2(image_file, target_size=(32, 32)):
    try:
        if isinstance(image_file, str):
            if image_file.startswith("data:image"):
                header, image_data = image_file.split(",", 1)
            else:
                image_data = image_file
            
            image_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(image_bytes))
        
        elif hasattr(image_file, 'read'):
            img = Image.open(image_file)
        
        else:
            raise ValueError("Unsupported image input type")
        
        # Convert to grayscale for processing
        img_gray = img.convert('L')
        
        # Convert to numpy array for OpenCV processing
        img_np = np.array(img_gray)
        
        # Threshold to create binary image
        _, thresh = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the bounding rectangle of the largest contour
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Add a bit of padding
            pad = max(w, h) // 8
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(img_np.shape[1] - x, w + 2*pad)
            h = min(img_np.shape[0] - y, h + 2*pad)
            
            # Crop the image
            img_cropped = img.crop((x, y, x+w, y+h))
        else:
            # If no contour found, use original image
            img_cropped = img
        
        # Resize while maintaining aspect ratio
        img_cropped = img_cropped.convert('RGB')
        img_cropped.thumbnail(target_size, Image.LANCZOS)
        
        # Create a white background image
        background = Image.new('RGB', target_size, (255, 255, 255))
        
        # Calculate position to paste the image (centered)
        offset = ((target_size[0] - img_cropped.width) // 2, 
                  (target_size[1] - img_cropped.height) // 2)
        
        # Paste the cropped image onto the white background
        background.paste(img_cropped, offset)
        
        # Convert to numpy array
        img_array = np.array(background)
        
        # Normalize the image
        img_array = img_array / 255.0
        
        # Expand dimensions to match model input shape (batch, height, width, channels)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise
    

def validate_image(file):
    """
    Validate the uploaded image file.
    
    Parameters:
    - file: Uploaded file object
    
    Returns:
    - Boolean indicating if the file is a valid image
    """
    try:
        # Check if file can be opened as an image
        img = Image.open(file)
        
        # Additional checks
        img.verify()  # Verify the image is not corrupted
        
        # Optional: Check file size (adjust as needed)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # Reset file pointer
        
        # Max file size (e.g., 5MB)
        if file_size > 5 * 1024 * 1024:
            return False
        
        return True
    except Exception:
        return False

app = Flask(__name__)

MODEL_FILENAME = "models/neo_character_model_v5.h5"

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

# get model mapping
with open('neo_character_model_mapping.json', 'r') as f:
    mapping_json = json.load(f)
    mapping = {int(key): value for key, value in mapping_json.items()}


def create_response(data=None, error=None, status=200):
    response = {
        'status': 'success' if error is None else 'error',
        'data': data,
        'error': error
    }
    return jsonify(response), status


@app.route('/')
def home():
    return render_template(template_name_or_list='home.html', user_name='Jack')  # Render the home page

@app.route('/predictions', methods=['POST'])
def create_prediction():
    
    if 'image' not in request.files:
        return create_response({}, 'No file uploaded', 400)
    
    image_data = request.files['image']
    
    if image_data.filename == '':
        return create_response({}, 'No selected file', 400)
    
    if not validate_image(file=image_data):
        return create_response({}, 'Invalid file type', 400)

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

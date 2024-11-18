import flask
from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import os
import pickle
#from tensorflow.keras.preprocessing import image
from keras.preprocessing import image
from modelBuilder import load_model
from modelTrainer import train_and_save_model

app = Flask(__name__)

MODEL_FILENAME = "models/OCR_v1.h5"

# Function to load the model when the server starts
def load_trained_model():
    if os.path.exists(MODEL_FILENAME):
        # If the model file exists, load it
        model = load_model(MODEL_FILENAME)
        print("Model loaded successfully")
    else:
        # If the model doesn't exist, train and save a new model
        print("Model not found, training a new one...")
        train_and_save_model()  # Train and save the model
        model = load_model(MODEL_FILENAME)  # Load the newly trained model
        print("New model trained and loaded successfully")
    
    return model

# Initialize the model
model = load_trained_model()

@app.route('/')
def home():
    return render_template('home.html', user_name='Jack')  # Render the home page

@app.route('/submit', methods=['POST'])
def submit_drawing():
    image_data = request.form['image']  # Get the image data from the POST request
    
    # The data URL contains a prefix (data:image/png;base64,...) that we need to remove
    image_data = image_data.split(',')[1]
    
    # Decode the base64 string
    image_bytes = base64.b64decode(image_data)
    
    # Convert to an image using PIL
    img = Image.open(BytesIO(image_bytes))
    img = img.resize((64, 64))  # Ensure the image size matches the model's input
    
    # Convert image to numpy array and normalize it
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make a prediction using the model
    prediction = model.predict(img_array)
    
    # Get the predicted class
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    print(f"Prediction for submitted drawing: {predicted_class}")
    
    return jsonify({'message': f'Prediction: {predicted_class}'})


if __name__ == '__main__':
    app.run(debug=True, port=9090)

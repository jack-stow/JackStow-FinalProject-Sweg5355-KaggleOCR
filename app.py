import flask
from flask import Flask, render_template, request, jsonify
import mlDataset
import base64
from io import BytesIO
from PIL import Image

app=Flask(__name__)

def init():
    print("initializing... ") 
  
# @app.route('/')
# def index():
#     return flask.render_template('index.html')

# @app.route('/predict', methods = ['POST'])
# def predict():
#     tagValuePairs = request.form.to_dict()
#     print(tagValuePairs)

#     return render_template('result.html', rc="hello from predict..")


# @app.route('/')
# def index():
#     return render_template('drawing.html')

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
    image = Image.open(BytesIO(image_bytes))
    print(type(image))
    # Save the image (you can customize the file path here)
    image.save('submitted_drawing.png')
    
    return jsonify({'message': 'Drawing submitted successfully!'})


if __name__ == '__main__':
    init()
    app.run(debug=True, port=9090)

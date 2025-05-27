from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import cv2
import os

app = Flask(__name__)

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Model input details
input_shape = input_details[0]['shape']  # e.g. [1, 128, 128, 3]
input_height, input_width = input_shape[1], input_shape[2]

# Load class names (update this list based on your dataset classes)
class_names = ['anadenanthera', 'arecaceae', 'arrabidaea', 'cecropia', 'chromolaena',
    'combretum', 'croton', 'dipteryx', 'eucalipto', 'faramea', 'hyptis', 'mabea',
    'matayba', 'mimosa', 'myrcia', 'protium', 'qualea', 'schinus', 'senegalia',
    'serjania', 'syagrus', 'tridax', 'urochloa']

def preprocess_image(image):
    """
    Preprocess uploaded image for model inference:
    - Resize to model input size
    - Normalize pixel values to [0,1]
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_width, input_height))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(image):
    """
    Run inference on the preprocessed image and return predicted class and confidence.
    """
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # output shape e.g. (23,)
    
    predicted_index = np.argmax(output_data)
    confidence = float(output_data[predicted_index])
    predicted_class = class_names[predicted_index]

    return predicted_class, confidence

@app.route('/')
def home():
    return render_template('index.html')  # create a simple form in index.html for image upload

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read image as OpenCV format
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    # Preprocess and predict
    img_processed = preprocess_image(img)
    pred_class, confidence = predict(img_processed)

    return jsonify({
        'predicted_class': pred_class,
        'confidence': confidence
    })


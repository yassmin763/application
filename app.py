from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import io
from PIL import Image

app = Flask(__name__)

# Load your trained model
MODEL_PATH = 'model.h5'  # change if your model filename is different
model = load_model(MODEL_PATH)

# Your class names (ensure this matches your label encoder classes)
class_names = [
    'Acalypha', 'Adenanthera', 'Alchornea', 'Alnus', 'Amaranthus', 'Anadenanthera', 
    'Anona', 'Artocarpus', 'Bauhinia', 'Bignoniaceae', 'Borassus', 'Calliandra', 
    'Canavalia', 'Casuarina', 'Ceiba', 'Cocos', 'Combretaceae', 'Convolvulaceae', 
    'Cordia', 'Costus', 'Eucalyptus', 'Euphorbia', 'Fabaceae'
]

IMG_SIZE = 128  # your model input size


def preprocess_image(image_bytes):
    # Open image
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # Resize
    img = img.resize((IMG_SIZE, IMG_SIZE))
    # Convert to numpy array
    img_array = np.array(img)
    # Normalize
    img_array = img_array / 255.0
    # Expand dims to match model input shape (1, IMG_SIZE, IMG_SIZE, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        img_bytes = file.read()
        img = preprocess_image(img_bytes)

        # Predict
        preds = model.predict(img)
        pred_class_idx = np.argmax(preds, axis=1)[0]
        pred_class_name = class_names[pred_class_idx]
        confidence = float(np.max(preds))

        return jsonify({
            'predicted_class': pred_class_name,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def index():
    return """
    <h1>Pollen Grain Classifier</h1>
    <p>Use POST /predict with an image file to get prediction.</p>
    """



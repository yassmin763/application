from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load TFLite model and allocate tensors.
MODEL_PATH = 'model.tflite'
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Your class names (ensure this matches your label encoder classes)
class_names = [
    'Acalypha', 'Adenanthera', 'Alchornea', 'Alnus', 'Amaranthus', 'Anadenanthera', 
    'Anona', 'Artocarpus', 'Bauhinia', 'Bignoniaceae', 'Borassus', 'Calliandra', 
    'Canavalia', 'Casuarina', 'Ceiba', 'Cocos', 'Combretaceae', 'Convolvulaceae', 
    'Cordia', 'Costus', 'Eucalyptus', 'Euphorbia', 'Fabaceae'
]

IMG_SIZE = 128  # Model input size (adjust if needed)


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype(np.float32)
    img_array = img_array / 255.0  # normalize to 0-1
    img_array = np.expand_dims(img_array, axis=0)  # add batch dim
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
        input_data = preprocess_image(img_bytes)

        # Set the tensor to point to the input data to be inferred
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        output_data = interpreter.get_tensor(output_details[0]['index'])

        pred_class_idx = np.argmax(output_data, axis=1)[0]
        pred_class_name = class_names[pred_class_idx]
        confidence = float(np.max(output_data))

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
    <p>Send a POST request to /predict with an image file.</p>
    """



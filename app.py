from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
import io

# Create FastAPI app
app = FastAPI()

# Enable CORS for all origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple health check route
@app.get("/health")
def health_check():
    return {"status": "API is running"}

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ✅ Replace with your actual class names
class_names = [
    'anadenanthera', 'arecaceae', 'arrabidaea', 'cecropia', 'chromolaena',
    'combretum', 'croton', 'dipteryx', 'eucalipto', 'faramea', 'hyptis', 'mabea',
    'matayba', 'mimosa', 'myrcia', 'protium', 'qualea', 'schinus', 'senegalia',
    'serjania', 'syagrus', 'tridax', 'urochloa'
]

# Preprocess image for prediction
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((128, 128))  # Match the model's input size
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0).astype(np.float32)
    return image_array

# Predict using TFLite model
def predict(image_bytes):
    input_data = preprocess_image(image_bytes)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = int(np.argmax(output_data))
    confidence = float(np.max(output_data))

    print(f"Predicted index: {predicted_class}, Confidence: {confidence}")
    print(f"Raw model output: {output_data}")

    return predicted_class, confidence

# Prediction endpoint
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    predicted_class, confidence = predict(image_bytes)

    if 0 <= predicted_class < len(class_names):
        class_name = class_names[predicted_class]
    else:
        class_name = "Unknown"

    return JSONResponse({
        "predicted_class": predicted_class,
        "class_name": class_name,
        "confidence": round(confidence, 4)
    })

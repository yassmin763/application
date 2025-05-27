from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import io

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def home():
    return {"message": "Pollen Grain Image Classification API is running"}

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class labels
labels = [
    'anadenanthera', 'arecaceae', 'arrabidaea', 'cecropia', 'chromolaena',
    'combretum', 'croton', 'dipteryx', 'eucalipto', 'faramea', 'hyptis', 'mabea',
    'matayba', 'mimosa', 'myrcia', 'protium', 'qualea', 'schinus', 'senegalia',
    'serjania', 'syagrus', 'tridax', 'urochloa'
]

# Preprocessing function
def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((128, 128))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # Preprocess image
    input_data = preprocess(contents)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])

    class_index = int(np.argmax(output_data))
    confidence = float(np.max(output_data))

    return {
        "predicted_class": labels[class_index],
        "confidence": confidence
    }

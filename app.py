from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os
import zipfile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/fati")
def root():
    return {"message": "API is running"}

# === Unzip model.zip if not already extracted ===
MODEL_DIR = "model"
ZIP_FILE = "model.zip"
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")

if not os.path.exists(TFLITE_MODEL_PATH):
    print("📦 Unzipping model...")
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)
    print("✅ Model unzipped!")

# === Load TFLite model ===
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Class labels ===
labels = [
    'cecropia', 'combretum', 'mabea', 'serjania', 'protium', 'arecaceae',
    'arrabidaea', 'senegalia', 'matayba', 'chromolaena', 'urochloa',
    'mimosa', 'tridax', 'qualea', 'dipteryx', 'anadenanthera',
    'eucalipto', 'croton', 'syagrus', 'schinus', 'faramea', 'hyptis', 'myrcia'
]

# === Preprocessing Function ===
def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((128, 128))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# === Prediction Route ===
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    input_data = preprocess(contents)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    class_index = int(np.argmax(output_data))
    confidence = float(np.max(output_data))
    result = {
        "class": labels[class_index],
        "confidence": round(confidence, 4)
    }
    return result

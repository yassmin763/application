import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import cv2

# FastAPI app
app = FastAPI()

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = 128  # Use the same size you trained with

# Define your class names (in order)
class_names = ['anadenanthera', 'arecaceae', 'arrabidaea', 'cecropia', 'chromolaena',
    'combretum', 'croton', 'dipteryx', 'eucalipto', 'faramea', 'hyptis', 'mabea',
    'matayba', 'mimosa', 'myrcia', 'protium', 'qualea', 'schinus', 'senegalia',
    'serjania', 'syagrus', 'tridax', 'urochloa']
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Load and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE))
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Run prediction
        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        class_idx = int(np.argmax(output_data))
        confidence = float(np.max(output_data))

        return JSONResponse({
            "class": class_names[class_idx],
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Run with: uvicorn main:app --reload

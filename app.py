import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # ✅ هذا هو السطر الناقص

import numpy as np
from PIL import Image
import tensorflow as tf
import io
import cv2

# FastAPI app
app = FastAPI()

# تفعيل CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # يمكنك تخصيصها لاحقاً مثل ["https://yourflutterapp.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/fati")
def root():
    return {"message": "API is running"}
# تحميل النموذج
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# الحصول على معلومات الإدخال والإخراج
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# أسماء الأصناف (استبدل بالأسماء الحقيقية)
class_names = ['anadenanthera', 'arecaceae', 'arrabidaea', 'cecropia', 'chromolaena',
    'combretum', 'croton', 'dipteryx', 'eucalipto', 'faramea', 'hyptis', 'mabea',
    'matayba', 'mimosa', 'myrcia', 'protium', 'qualea', 'schinus', 'senegalia',
    'serjania', 'syagrus', 'tridax', 'urochloa']

# إعداد الصورة
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((128, 128))  # الحجم المطلوب من النموذج
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0).astype(np.float32)
    return image_array

# التنبؤ
def predict(image_bytes):
    input_data = preprocess_image(image_bytes)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = int(np.argmax(output_data))
    confidence = float(np.max(output_data))
    return predicted_class, confidence

# نقطة النهاية الرئيسية
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    predicted_class, confidence = predict(image_bytes)
    class_name = class_names[predicted_class]
    return JSONResponse({
        "predicted_class": predicted_class,
        "class_name": class_name,
        "confidence": round(confidence, 4)
    })

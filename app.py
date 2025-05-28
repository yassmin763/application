from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import io
import os
import zipfile

app = FastAPI()

# إعداد CORS للسماح بطلبات خارجية
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# مسارات الملفات
MODEL_DIR = "model"
ZIP_FILE = "model.zip"
H5_MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")

# تصنيفات الصور
labels = [
    'cecropia', 'combretum', 'mabea', 'serjania', 'protium', 'arecaceae',
    'arrabidaea', 'senegalia', 'matayba', 'chromolaena', 'urochloa',
    'mimosa', 'tridax', 'qualea', 'dipteryx', 'anadenanthera',
    'eucalipto', 'croton', 'syagrus', 'schinus', 'faramea', 'hyptis', 'myrcia'
]

# تحميل Lazy للنموذج
model = None  # سيتم تحميله عند أول طلب

# دالة لفك ضغط النموذج عند الحاجة
def extract_model():
    if not os.path.exists(H5_MODEL_PATH):
        print("📦 فك ضغط النموذج...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)
        print("✅ تم فك ضغط النموذج!")

# دالة لتحميل النموذج عند الطلب
def load_model_lazy():
    global model
    if model is None:
        extract_model()
        print("⚙️ تحميل النموذج...")
        model = load_model(H5_MODEL_PATH)
        print("✅ النموذج جاهز!")

# دالة المعالجة المسبقة للصورة
def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((128, 128))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# نقطة اختبار
@app.get("/fati")
def root():
    return {"message": "🚀 API تعمل بنجاح!"}

# نقطة التنبؤ
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    input_data = preprocess(contents)

    load_model_lazy()  # تحميل النموذج عند أول استخدام

    predictions = model.predict(input_data)
    class_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    return {
        "class": labels[class_index],
        "confidence": round(confidence, 4)
    }

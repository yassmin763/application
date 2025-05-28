from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io

# إنشاء التطبيق
app = FastAPI()

# إعداد CORS للسماح لجميع الطلبات (خلال التطوير فقط)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# نقطة اختبار (صحية)
@app.get("/health")
def root():
    return {"message": "API is running"}

# تحميل نموذج TFLite
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# تفاصيل الإدخال والإخراج
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ✅ أسماء الأصناف (نفس الترتيب الذي استخدمته أثناء التدريب)
labels = [
    'anadenanthera', 'arecaceae', 'arrabidaea', 'cecropia', 'chromolaena',
    'combretum', 'croton', 'dipteryx', 'eucalipto', 'faramea', 'hyptis',
    'mabea', 'matayba', 'mimosa', 'myrcia', 'protium', 'qualea', 'schinus',
    'senegalia', 'serjania', 'syagrus', 'tridax', 'urochloa'
]

# دالة تهيئة الصورة للإدخال
def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((128, 128))  # يجب أن يطابق حجم إدخال النموذج
    img_array = np.array(image).astype(np.float32) / 255.0  # تحويل لقيم بين 0 و 1
    img_array = np.expand_dims(img_array, axis=0)  # إضافة بعد للدفعة (batch)
    return img_array

# نقطة التنبؤ
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    input_data = preprocess(contents)

    # إدخال البيانات للنموذج
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # استخراج التنبؤ
    class_index = int(np.argmax(output_data))
    confidence = float(np.max(output_data))
    class_name = labels[class_index] if class_index < len(labels) else "Unknown"

    return JSONResponse({
        "class_index": class_index,
        "class_name": class_name,
        "confidence": round(confidence, 4)
    })

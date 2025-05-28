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
# تحميل نموذج TFLite
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# معلومات الإدخال والإخراج
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_HEIGHT = input_details[0]['shape'][1]
IMG_WIDTH = input_details[0]['shape'][2]

# أسماء الفئات
CLASS_NAMES = ['anadenanthera', 'arecaceae', 'arrabidaea', 'cecropia', 'chromolaena',
    'combretum', 'croton', 'dipteryx', 'eucalipto', 'faramea', 'hyptis', 'mabea',
    'matayba', 'mimosa', 'myrcia', 'protium', 'qualea', 'schinus', 'senegalia',
    'serjania', 'syagrus', 'tridax', 'urochloa']
# تجهيز الصورة
def preprocess_image(image: Image.Image):
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image).astype(np.float32)
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# التنبؤ
def predict(image_array):
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = int(np.argmax(output))
    confidence = float(np.max(output))
    return CLASS_NAMES[predicted_index], confidence

# نقطة التنبؤ
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    input_array = preprocess_image(image)
    label, confidence = predict(input_array)

    return JSONResponse(content={
        "prediction": label,
        "confidence": round(confidence * 100, 2)
    })
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

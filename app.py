from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# تحميل النموذج tflite
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# الحصول على تفاصيل المدخلات والمخرجات
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/')
def home():
    return "TFLite Model API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = np.array(data['input'], dtype=np.float32).reshape(input_details[0]['shape'])

        # إدخال البيانات إلى المفسر
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # الحصول على النتيجة
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = output_data.tolist()

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

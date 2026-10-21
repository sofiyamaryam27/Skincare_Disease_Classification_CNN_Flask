from collections import deque
from io import BytesIO
import base64
import os

import numpy as np
from PIL import Image
from flask import Flask, request, render_template, session
import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")
app.config['MAX_PREV_PREDICTIONS'] = 5

model = keras.models.load_model(os.path.join(BASE_DIR, "skincare_disease_model.h5"))
prev_predictions = deque(maxlen=app.config['MAX_PREV_PREDICTIONS'])

CLASS_NAMES = [
    "Actinic Keratosis - Benign.",
    "Basal Cell Carcinoma - Malignant.",
    "Dermatofibroma - Benign.",
    "Melanoma - Malignant.",
    "Nevus - Benign.",
    "Pigmented Benign Keratosis - Benign.",
    "Seborrheic Keratosis - Benign.",
    "Squamous Cell Carcinoma - Malignant.",
    "Vascular Lesion - Benign.",
]

def preprocess_image(image):
    image = image.resize((100, 75))
    image_array = np.array(image, dtype=np.float32) - 162.57063208888889
    image_array /= 41.96703347421026
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def to_data_uri(image):
    thumb = image.copy()
    thumb.thumbnail((400, 400))
    buffer = BytesIO()
    thumb.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/types', methods=['GET'])
def types():
    return render_template('types.html')

@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/predict-type', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file or file.filename == '':
        return render_template('predict.html', data="Please upload an image to predict.")

    image = Image.open(file).convert("RGB")
    prediction = model.predict(preprocess_image(image), verbose=0)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100

    if predicted_class < len(CLASS_NAMES):
        className = CLASS_NAMES[predicted_class]
    else:
        className = "Unknown cancer type."

    image_path = to_data_uri(image)

    session['prediction'] = className
    session['confidence'] = confidence
    prev_predictions.append({'class_name': className, 'image_path': image_path})

    return render_template('predict.html', data=className, image_path=image_path,
                           confidence=round(confidence, 2))

@app.route('/history', methods=['GET'])
def history():
    return render_template('history.html', prev_predictions=prev_predictions)

if __name__ == "__main__":
    app.run()
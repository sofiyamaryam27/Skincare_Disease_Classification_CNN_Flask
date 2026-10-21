from collections import deque
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, render_template, session
import keras
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
app.secret_key = 'your_secret_key' 
app.config['MAX_PREV_PREDICTIONS'] = 5

model = keras.models.load_model("skincare_disease_model.h5")
prev_predictions = deque(maxlen=app.config['MAX_PREV_PREDICTIONS'])

def preprocess_image(image):
    image = image.resize((100, 75))
    image_array = np.array(image, dtype=np.float32) - 162.57063208888889
    image_array /= 41.96703347421026
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

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
    className = ''
    confidence = 0.0  # Initialize confidence to 0
    file = request.files['image']
    if not file:
        className = "Please upload an image to predict."
        return render_template('predict.html', data=className)

    # Generate a unique filename
    unique_filename = str(uuid.uuid4()) + '.jpg'
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    image = Image.open(file)
    image_array = preprocess_image(image)
    prediction = model.predict(image_array)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100  # Convert to percentage

    if predicted_class == 0:
        className = "Actinic Keratosis - Benign."
    elif predicted_class == 1:
        className = "Basal Cell Carcinoma - Malignant."
    elif predicted_class == 2:
        className = "Dermatofibroma - Malignant."
    elif predicted_class == 3:
        className = "Melanoma - Malignant."  
    elif predicted_class == 4:
        className = "Nevus - Benign."  
    elif predicted_class == 5:
        className = "Pigmented Benign Keratosis - Benign."
    elif predicted_class == 6:
        className = "Seborrheic Keratosis - Benign."
    elif predicted_class == 7:
        className = "Squamous Cell Carcinoma - Malignant."
    elif predicted_class == 8:
        className = "Vascular Lesion - Benign."    
    else:
        className = "Unknown cancer type."

    session['prediction'] = className
    session['image_path'] = image_path
    session['confidence'] = confidence

    # Add the current prediction to the deque
    prev_predictions.append({'class_name': className, 'image_path': image_path})

    image.save(image_path)
    print(image_path)
    return render_template('predict.html', data=className, image_path=image_path, confidence=round(confidence, 2))

@app.route('/history', methods=['GET'])
def history():
    return render_template('history.html', prev_predictions=prev_predictions)

if __name__ == "__main__":
    app.run()

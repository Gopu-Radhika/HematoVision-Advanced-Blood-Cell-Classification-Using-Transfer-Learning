import os
print("Current working dir:", os.getcwd())
print("Files in dir:", os.listdir())

import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'Blood Cell.h5'
model = load_model(MODEL_PATH)

# Define class labels as per your model
class_labels = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file:
        filepath = os.path.join('static', file.filename)
        file.save(filepath)

        # Load and preprocess the image
        img = image.load_img(filepath, target_size=(64, 64))  # Adjust size if needed
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = round(100 * np.max(prediction), 2)

        return render_template('result.html',
                               label=predicted_class,
                               confidence=confidence,
                               image_file=filepath)

if __name__ == '__main__':
    app.run(debug=True)

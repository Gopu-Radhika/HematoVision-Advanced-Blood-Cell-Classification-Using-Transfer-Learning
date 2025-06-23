from flask import Flask, render_template, request # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np # type: ignore
import os
app = Flask(__name__)
model = load_model("Blood Cell.h5")

# Define class labels (update based on your model's output classes)
class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        img_path = os.path.join('static', file.filename)
        file.save(img_path)

        # Image preprocessing
        img = image.load_img(img_path, target_size=(64, 64))  # Adjust size as per model
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        return render_template('result.html', prediction=predicted_class, image_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)

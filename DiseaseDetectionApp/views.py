import tensorflow as tf
from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import numpy as np
import os
from django.conf import settings
import json

# Load the model
# model = load_model('C:/proj/PlantDiseaseDetection/DiseaseDetectionApp/tomato_disease_model.keras')
model_path = os.path.join(os.path.dirname(__file__),'models','my_model.h5')
model = load_model(model_path)
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

# Class labels
class_labels = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Prediction function
def Prediction(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)

    return predicted_class, confidence

def detect_disease(request):
    result = 'No prediction made.'
    confidence = "0"

    prevention_file_path = os.path.join(settings.BASE_DIR, 'DiseaseDetectionApp', 'prevention_methods.json')
    with open(prevention_file_path, 'r') as file:
        prevention_methods = json.load(file)

    context = {'result': result, 'confidence': confidence}

    if request.method == "POST" and 'image_input' in request.FILES:
        uploaded_file = request.FILES['image_input']
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploaded_images')
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        try:
            img = load_img(file_path, target_size=(256, 256))
            predicted_class, confidence = Prediction(model, img)
            prevention = prevention_methods.get(predicted_class, ["No prevention methods available."])

            context = {
                'result': f"Predicted label: {predicted_class}",
                'confidence': f"Confidence score: {confidence}%",
                'prevention': prevention
            }
        except Exception as e:
            context = {'result': f"Error processing the image: {e}"}

        os.remove(file_path)  # Clean up uploaded file after prediction

    return render(request, 'index.html', context)


# Create your views here.
def index(request):
    return render(request,'index.html')

def register(request):
    return

def login(request):
    return
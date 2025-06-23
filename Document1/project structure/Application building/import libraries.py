import os
import numpy as np
import cv2
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import io
import base64 

# Initialize Flask app
app = Flask(__name__) # Load the trained model
model = load_model("Blood Cell.h5")
# Define the class labels for predictions
class_labels = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']

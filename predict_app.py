import base64
import numpy as np
import io
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import model_from_json
from tensorflow.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
from base64 import b64encode
from matplotlib import image
from matplotlib import pyplot
from numpy import asarray
import os
import tensorflow
import keras.backend.tensorflow_backend as tb
#tb._SYMBOLIC_SCOPE.value = True


app = Flask(__name__)

def get_model():
    global model
    model.load("xray_trial_model.h5")
    print("Loaded model from disk")
    print(" * Model loaded!")
    
def preprocess_image(img, img_dims):
    img = asarray(img)
    img = cv2.resize(img, (img_dims, img_dims))
    img = np.dstack([img, img, img])
    img = img.astype('float32') / 255
    img = np.reshape(img,[1,150,150,3])
    print("Image processed")
    return img

print(" * Loading Keras model ...:")
get_model()
@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    #encoded = message['image']
    decoded = base64.b64decode( message['image'])
    imge = Image.open(io.BytesIO(decoded))
    print("Image decoded")
    processed_image = preprocess_image(imge,150)
    prediction = model.predict(processed_image)
    print('Image predicted')
    response = {
        'prediction': 'Chances '+ str(prediction[0][0]*100) +'%'
        }
    return jsonify(response)

app.run(host="0.0.0.0",port=5000,threaded=False)

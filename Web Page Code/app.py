from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.preprocessing import image
from keras.models import load_model
import h5py

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'E:/MAYURESH/SDP/dev/covid_classifier_final_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)


print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path,model):
    img = image.load_img(img_path,target_size=(256,256))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    result = model.predict(img)
    for i in result:
        index = np.argmax(i)
        if index == 0:
            text = "COVID POSITIVE"
        elif index == 1:
            text = "HEALTHY"
        else:
            text = "VIRAL PNEUMONIA"
        return text


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        return preds
    return None

@app.route('/form',methods=['GET','POST'])
def form():
    if request.method == 'GET':
        return render_template('test.html')
    elif request.method == 'POST':
         result = request.form
         return render_template("result.html",result = result)



@app.route('/info',methods=['GET'])
def info():
    #formpage
    return render_template('tips.html')

if __name__ == '__main__':
    app.run(debug=True)
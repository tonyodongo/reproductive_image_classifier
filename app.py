from flask import Flask, request, make_response, render_template

import numpy as np
import pandas as pd 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import random
import os
import time
import os, sys
from PIL import Image

app = Flask(__name__)

#set paths to upload folder
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['IMAGE_UPLOADS'] = os.path.join(APP_ROOT, 'static')

@app.route("/image-classifier",methods=["GET","POST"])
def classify_image():
    if request.method == "POST":
        
        #read and upload resized files to folder
        image = request.files['input_file']
        filename = image.filename
        file_path = os.path.join(app.config["IMAGE_UPLOADS"], filename)
        image_pil = Image.open(image)
        
        #crop and resize
        width, height = image_pil.size
        left = width/10
        top = height/8
        right = 9 * width/10
        bottom = 7 * height/8
        image_pil.crop((left, top, right, bottom))
        image_pil.resize((550,850), Image.ANTIALIAS)
        image_pil.thumbnail((550,850), Image.ANTIALIAS)
        image_pil.save(file_path)
        
        #classify image
        image = load_img(image, target_size=(850, 550))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        
        #predict
        prediction = model.predict(image)
        prediction = decode_predictions(prediction)[0][0][1]
        prediction = prediction.replace('_',' ')
        
        #display prediction and image
        return render_template("upload.html", image_path = filename, prediction = 'Prediction: '+prediction)
    return render_template("upload.html", image_path = 'landing_page_pic.jpg')

if __name__ == '__main__':
    model = load_model('./reproductive_xception_850_cont4_lr_93.207%.hdf5')
    app.run(host='0.0.0.0', debug=False, threaded=False, port=8000)
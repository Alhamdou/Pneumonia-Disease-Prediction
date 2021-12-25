
from six import class_types
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras import models
from tensorflow.python.ops.gen_math_ops import imag, mod
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
# loading my hdf5 model
def load_model():
	model = tf.keras.models.load_model("save_models/x_ray_model.hdf5")
	# model = tf.keras.models.load_model("model1_pkl")
	
	return model
st.title("Welcome to the Pneumonia image classification site by Alhamdou")

# starting to used streamlit
with st.spinner("Loading model"):
	model = load_model()

st.write("X-ray image classifier")

# collecting image from the user system
file = st.file_uploader("Upload your image here", type =["jpg","png","jpeg"])

# hence am dealing with images i need to use cv2 and pillow

import cv2
from PIL import Image, ImageOps
st.set_option("deprecation.showfileUploaderEncoding", False)

# import and predicting image

def import_image_predict(image_data, model):
	size  = (256, 256)
	image = ImageOps.fit(image_data,size,Image.ANTIALIAS)
	image = np.asarray(image)
	img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# image_reshape = img[np.newaxis]
	prediction = model.predict(img) 
	# image_reshape
	return prediction

# add more functionalities and help in determining the precentage of the predicted image
if file is None:
	st.text("Please upload an image")
else:
	image = Image.open(file)
	st.image(image, use_column_width=True)
	predictions = import_image_predict(image, model)
	score = tf.nn.softmax(predictions[0])
	st.write(predictions)
	st.write(score)
	if score[1] >= 0.8:
		st.write("pneumonia")
	else:
		st.write("non pneumonia ") 
	# print("This image is {} and it has a  {:.2f} percentage of confidence".format(class_names[np.argmax(score)],100*np.max(score)))
	
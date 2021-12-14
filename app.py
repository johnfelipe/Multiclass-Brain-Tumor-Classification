# Import Libraries
import streamlit as st
import tensorflow as tf
from path import Path
import os
from PIL import Image, ImageOps
import numpy as np
import pathlib
d = Path(__file__).parent

# Class names
data_dir = pathlib.Path("project/volume/data/raw/archive/Training")
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))


@st.cache(allow_output_mutation=True)
def load_model():
    model_1 = tf.keras.models.load_model(os.path.join(d, 'project/volume/models') + '/CNN_V2.hdf5')
    return model_1


with st.spinner('Model is being loaded..'):
    model_1 = load_model()

st.write("""
         # Classification of Brain Tumors from MRI Images Using a Convolutional Neural Network

         ## The convolutional network was developed in-house
         """
         )

file = st.file_uploader("Please upload an MRI brain scan file", type=["jpg", "png", "jpeg"])

st.set_option('deprecation.showfileUploaderEncoding', False)


def import_and_predict(image_data, model):
    """
    This function takes in the uploaded image and makes predictions on it
    :param image_data:
    :param model_1:
    :return:
    """
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    img_reshape = img_reshape.reshape(-1, 256, 256, 1)
    prediction = model.predict(img_reshape)
    return prediction


if file is None:
    st.text('Upload An Image file')
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image_data=image, model=model_1)
    class_names = ['Glioma', 'Meningioma', 'NoTumor', 'Pituitary']
    string = "This is image is most likely: " + class_names[np.argmax(predictions[0])]
    st.success(string)



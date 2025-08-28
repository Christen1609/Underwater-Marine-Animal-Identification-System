import os
import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import streamlit as  st
import tensorflow as tf 
from keras import models
from keras.models import load_model
import keras  
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing import image
import base64
CWD = os.getcwd()
model_path = r'C:/Users/loyol/OneDrive/Documents/projects/Final_Year_Project_April_2nd/final_year_project/trained_model.h5'
loaded_model = load_model(model_path)

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define a function to make predictions
def predict_image_class(image_path, model):
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)
    return predicted_class

def get_row_data(species_code):
    df = pd.read_csv(f'{CWD}/Image_recog_Project_Dataset_30Classes.csv')
    row_data = df[df['Species'].str.lower() == species_code.lower()]
    return row_data

def main():
    st.title("Underwater Animal Classifier")
    image_path = st.text_input("Upload fish image path")

    if image_path:
        try:
            predicted_class = predict_image_class(image_path, loaded_model)

            dir_content = os.listdir(f'{CWD}/Validation_Set')
            dir_content.sort()
            mapping ={}
            i=0
            for label in dir_content:
                mapping[i] = label
                i +=1
            if predicted_class:
                
                Information = get_row_data(mapping[predicted_class])
                st.subheader("The name of the species is : ")
                st.write(mapping[predicted_class])
                st.subheader("Information :")
                st.write(Information)
            else:
                st.subheader("No match found")
            
        except Exception as e:
            st.write(f"Error: {e}")    
        
    
           
if __name__ == "__main__":
    main()
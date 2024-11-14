import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource 
def load_model():
    # Ensure the model is saved in a compatible version
    model = tf.keras.models.load_model('lung_cancer_classification_model.h5')  # Assuming you re-saved the model
    return model

# Load and use the model
model = load_model()

# Streamlit title and description
st.title("Lung Cancer Classification")
st.write("""
This application classifies lung cancer into three categories: 
1. Adenocarcinoma 
2. Neuroendocrine tumors 
3. Squamous cell carcinoma
Upload a lung CT scan or X-ray image to get a prediction.
""")

# Sidebar for patient information
st.sidebar.title("Patient Information")
name = st.sidebar.text_input("Patient Name",value='Unkown')
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

st.sidebar.write("Please enter correct patient details before proceeding.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image (CT scan or X-ray)...", type=["jpg", "jpeg", "png"])

def preprocess_image(image_data):
    """
    Preprocess the uploaded image to fit the model input
    """
    image = Image.open(image_data)
    # Use LANCZOS for high-quality downsampling
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)  # Resize to model input size
    img = np.asarray(image)
    img = img.astype(np.float32) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Class labels for lung cancer types
class_labels = ['Adenocarcinoma', 'Neuroendocrine tumors', 'Squamous cell carcinoma']

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Lung Image", use_column_width=True)
    st.write("Classifying the image...")

    # Preprocess the image and make predictions
    img = preprocess_image(uploaded_file)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)  # Get the index of the highest probability
    
    # Get the confidence score for the predicted class
    confidence = prediction[0][predicted_class[0]] * 100
    
    # Display only the predicted class and its confidence
    st.write(f"### Predicted Class: *{class_labels[predicted_class[0]]}*")
    st.write(f"#### Confidence: {confidence:.2f}%")
    
    # Display patient details
    st.write(f"Patient: {name}, Age: {age}, Gender: {gender}")
    st.write("*Note*: Please consult a medical professional for an official diagnosis.")

# Footer
st.write("*Disclaimer:* This tool is for educational purposes.")

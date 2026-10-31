
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
from huggingface_hub import hf_hub_download

st.title("Apple Leaf Disease Detection")
st.write("Upload an apple leaf image to detect its disease.")

# Hugging Face model details
REPO_ID = "YOUR_USERNAME/YOUR_MODEL_REPO"
MODEL_FILE = "YOUR_MODEL_FILE.keras"

# Replace these with your actual training classes
classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy"
]

@st.cache_resource
def load_model():
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILE
    )
    return tf.keras.models.load_model(path)

model = load_model()

# Take image from user
uploaded_file = st.file_uploader(
    "Upload Apple Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Apple Leaf")

    if st.button("Predict Disease"):

        # Image preprocessing
        image = image.resize((128, 128))

        image_array = np.array(image)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Model prediction
        prediction = model.predict(image_array)

        result = np.argmax(prediction[0])
        confidence = prediction[0][result] * 100

        st.success(f"Disease: {classes[result]}")
        st.write(f"Confidence: {confidence:.2f}%")

import streamlit as st
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

st.title("Apple Leaf Disease Detection")
st.write("Upload an apple leaf image to detect its disease using your Hugging Face DINOv2 model.")

# Your actual Hugging Face Repository ID
REPO_ID = "shahab1/apple-disease-dinov2"

# Your complete list of classes from your dataset
classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___healthy"
]
# Note: Adjust the list above to match the exact order and number of classes 
# from your training dataset if you have more than 4 classes.

@st.cache_resource
def load_model_and_processor():
    processor = AutoImageProcessor.from_pretrained(REPO_ID)
    model = AutoModelForImageClassification.from_pretrained(REPO_ID)
    model.eval()
    return processor, model

processor, model = load_model_and_processor()

# Take image from user
uploaded_file = st.file_uploader(
    "Upload Apple Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Apple Leaf", use_column_width=True)

    if st.button("Predict Disease"):
        with st.spinner("Analyzing leaf..."):
            # Preprocess image using the Hugging Face processor
            inputs = processor(images=image, return_tensors="pt")

            # Model prediction
            with torch.no_grad():
                outputs = model(**inputs).logits
                probabilities = torch.nn.functional.softmax(outputs, dim=-1)
                confidence, predicted_idx = torch.max(probabilities, dim=-1)

            result_idx = predicted_idx.item()
            conf_score = confidence.item() * 100
            
            # Safe check in case class index is within bounds
            if result_idx < len(classes):
                predicted_class = classes[result_idx]
            else:
                predicted_class = f"Class Index {result_idx}"

            st.success(f"**Disease:** {predicted_class}")
            st.write(f"**Confidence:** {conf_score:.2f}%")

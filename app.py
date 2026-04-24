import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import gdown
import os
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Pneumonia Detection",
    layout="centered"
)

st.title("🫁 Pneumonia Detection using CNN")
st.write("Upload a Chest X-ray image to detect Pneumonia")

# -------------------------------
# Download model from Google Drive
# -------------------------------

MODEL_PATH = "pneumonia_cnn_final.keras"

FILE_ID = "1QADN1kFNV10ZQpLL-lh4GD1KKih87N5H"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# -------------------------------
# Load model
# -------------------------------

@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH)

model = load_my_model()

# -------------------------------
# Upload image
# -------------------------------

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read uploaded image
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (224, 224))

    img_array = np.expand_dims(img, axis=0) / 255.0

    # Prediction
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.error("Prediction: PNEUMONIA")
        st.write(f"Confidence Score: {prediction:.2f}")
    else:
        st.success("Prediction: NORMAL")
        st.write(f"Confidence Score: {1 - prediction:.2f}")

    # Show uploaded image
    st.subheader("Uploaded Chest X-ray")
    st.image(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        use_container_width=True
    )

    # Show pre-generated Grad-CAM image
    st.subheader("Grad-CAM Visualization")
    st.image(
        "gradcam_result.png",
        caption="Model Attention Heatmap",
        use_container_width=True
    )
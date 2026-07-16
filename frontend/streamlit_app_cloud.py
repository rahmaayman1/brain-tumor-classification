import streamlit as st
import numpy as np
from PIL import Image
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.preprocessing import preprocess_image
from src.config import CLASS_NAMES, CLASS_DESCRIPTIONS

@st.cache_resource
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model("models/transfer_best_phase2.keras")

st.set_page_config(
    page_title="Brain Tumor Classification",
    layout="centered"
)

st.title("Brain Tumor Classification")
st.markdown("Upload a brain MRI scan to classify the tumor type.")

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    if st.button("Classify", type="primary"):
        with st.spinner("Loading model..."):
            model = load_model()

        with st.spinner("Analyzing MRI scan..."):
            img_array = preprocess_image(image)
            predictions = model.predict(img_array, verbose=0)[0]

        predicted_idx = int(predictions.argmax())
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(predictions[predicted_idx]) * 100

        st.markdown("---")

        if predicted_class == "notumor":
            st.success(f"No Tumor Detected - {confidence:.1f}% confidence")
        else:
            st.error(f"{predicted_class.capitalize()} Detected - {confidence:.1f}% confidence")

        st.markdown(f"*{CLASS_DESCRIPTIONS[predicted_class]}*")

        st.subheader("All Probabilities")
        probs = {
            CLASS_NAMES[i]: float(predictions[i]) * 100
            for i in range(len(CLASS_NAMES))
        }
        for class_name, prob in sorted(probs.items(),
                                        key=lambda x: x[1],
                                        reverse=True):
            st.progress(prob / 100, text=f"{class_name}: {prob:.1f}%")

st.markdown("---")
st.caption("Model: EfficientNetB0 | Test Accuracy: 91.8%")
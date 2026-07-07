import streamlit as st
import requests
from PIL import Image
import io
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Classification")
st.markdown("Upload a brain MRI scan to classify the tumor type.")

uploaded_file = st.file_uploader(
    "Choose an MRI image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    if st.button("Classify", type="primary"):
        with st.spinner("Analyzing MRI scan..."):
            files = {"file": (uploaded_file.name,
                              uploaded_file.getvalue(),
                              uploaded_file.type)}
            response = requests.post(f"{API_URL}/predict", files=files)

        if response.status_code == 200:
            result = response.json()["prediction"]

            predicted = result["predicted_class"]
            confidence = result["confidence"]
            description = result["description"]
            probs = result["all_probabilities"]

            if predicted == "notumor":
                st.success(f"✅ No Tumor Detected ({confidence}% confidence)")
            else:
                st.error(f"⚠️ {predicted.capitalize()} Detected ({confidence}% confidence)")

            st.markdown(f"*{description}*")

            st.subheader("All Probabilities")
            for class_name, prob in sorted(probs.items(),
                                           key=lambda x: x[1],
                                           reverse=True):
                st.progress(prob / 100, text=f"{class_name}: {prob}%")
        else:
            st.error("Error connecting to API. Make sure the API is running.")

st.markdown("---")
st.caption("Model: EfficientNetB0 | Test Accuracy: 91.8%")
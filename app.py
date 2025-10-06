import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model("coral_model.h5")

st.title("🌊 Coral Health Detection Demo")
st.markdown("Upload a coral reef image and get an AI-based bleaching prediction.")

uploaded_file = st.file_uploader("Upload Coral Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((128,128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    if pred < 0.5:
        st.success("Healthy Coral ")
        st.metric("Bleaching Probability", f"{round(pred*100, 2)}%")
    else:
        st.error("Bleached Coral Detected")
        st.metric("Bleaching Probability", f"{round(pred*100, 2)}%")

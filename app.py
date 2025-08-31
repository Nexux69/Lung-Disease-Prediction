import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import base64

# ------------------------------
# Function to set full background + custom CSS
# ------------------------------
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}

    .main-block {{
        background: rgba(255, 255, 255, 0.15);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }}

    h1 {{
        text-align: center;
        color: #ffffff;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.6);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# ------------------------------
# Class Labels (Update as per training)
# ------------------------------
class_labels = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']


# ------------------------------
# Function to check if image looks like an X-ray
# ------------------------------
def is_xray(img):
    arr = np.array(img)
    if len(arr.shape) == 2:  # Grayscale
        return True
    elif len(arr.shape) == 3 and arr.shape[2] == 3:  # RGB
        if np.allclose(arr[:,:,0], arr[:,:,1], atol=20) and np.allclose(arr[:,:,1], arr[:,:,2], atol=20):
            return True
    return False


# ------------------------------
# Custom Prediction Box
# ------------------------------
def show_prediction(disease, confidence, is_disease=True):
    color = "#00FF7F" if is_disease else "#FF4C4C"
    icon = "✅" if is_disease else "❌"
    st.markdown(f"""
        <div style="
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            background-color: rgba(0,0,0,0.7);
            color: {color};
        ">
            {icon} <span style="color:#1E90FF;">{disease.upper()}</span> 
            with {confidence:.2f}% confidence
        </div>
    """, unsafe_allow_html=True)


# ------------------------------
# Set Background
# ------------------------------
set_background("assets/background.png")


# ------------------------------
# Streamlit UI
# ------------------------------
st.markdown("<h1>🩻 X-Ray Disease Detection</h1>", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="main-block">', unsafe_allow_html=True)

    st.write("📂 Upload an X-ray image and let the AI predict possible diseases.")
    uploaded_file = st.file_uploader("Upload X-Ray Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # ✅ Check if it's really an X-ray
        if not is_xray(img):
            st.error("❌ This does not look like an X-ray image. Please upload a valid chest X-ray.")
        else:
            # ✅ Preprocess image
            img = img.resize((200, 200))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            preds = model.predict(img_array)
            predicted_class = class_labels[np.argmax(preds)]
            confidence = np.max(preds) * 100

            # ✅ Show result
            if confidence > 50:
                show_prediction(predicted_class, confidence, is_disease=True)
            else:
                show_prediction("NON-DISEASE (Normal)", confidence, is_disease=False)

    st.markdown('</div>', unsafe_allow_html=True)


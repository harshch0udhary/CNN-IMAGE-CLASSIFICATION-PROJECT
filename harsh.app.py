import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Set Page Config (Themes and Layout)
st.set_page_config(
    page_title="Image Classifier Dashboard",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. Custom CSS for Beautiful UI Styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #4CAF50;
    }
    </style>
""", unsafe_allow_html=True) # <-- Fixed here

# 2. Cache Model for Faster Loading
@st.cache_resource
def load_my_model():
    # Make sure 'bawli.keras' is in the same folder as this script
    model = tf.keras.models.load_model("HARSH_MODEL.keras")
    return model

try:
    model = load_my_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {e}. Please ensure 'bawli.keras' is in the working directory.")

# 6 Standard Classes for Intel Image Classification (Update if your classes differ)
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# --- SIDEBAR FEATURE ---

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #000000;
}

[data-testid="stSidebar"] * {
    color: white;
}
</style>
""", unsafe_allow_html=True)
st.sidebar.markdown("""
<h2 style="
    color:#FFFFFF;
    text-align:center;
    font-size:28px;
    font-family:Arial;
    font-weight:900;
    text-shadow:
        1px 1px 0px #000,
        2px 2px 0px #111,
        3px 3px 0px #222,
        4px 4px 0px #333,
        5px 5px 10px rgba(0,0,0,0.6);
">
🚀 MODEL DASHBOARD SETTINGS
</h2>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Feature 1: Adjust Confidence Threshold
confidence_threshold = st.sidebar.slider(
    "SET MINIMUM CONFIDENCE THRESHOLD (%)", 
    min_value=0, max_value=100, value=50, step=5
)

st.sidebar.info(
    "🌿 **About This Model**\n\n"
    "This custom CNN has been trained to classify images into the following "
    "environmental categories:\n\n"
    "🏢 Buildings\n\n"
    "🌲 Forest\n\n"
    "🧊 Glacier\n\n"
    "⛰️ Mountain\n\n"
    "🌊 Sea\n\n"
    "🛣️ Street"
)

# --- MAIN APP PAGE ---

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #fff8e1, #fce4ec, #e3f2fd, #e8f5e9, #ffffff);
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div style="
    background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.4);
    margin-bottom: 20px;
">
    <h1 style="
        color: #00E5FF;
        font-size: 42px;
        font-family: 'Trebuchet MS', sans-serif;
        font-weight: bold;
        text-shadow:
            2px 2px 0px #004D66,
            4px 4px 6px rgba(0,0,0,0.6);
        margin: 0;
    ">
        🖼️ IMAGE CLASSIFICATION & ANALYSIS DASHBOARD
    </h1>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<b style='color:black;'>🚀 Upload an image below to see live predictions, class probabilities, and visual analysis.</b>",
    unsafe_allow_html=True
)

# Layout Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
<h3 style="
    color:#FFD700;
    text-shadow:
        1px 1px 0 #B8860B,
        2px 2px 0 #8B7500,
        3px 3px 6px rgba(0,0,0,0.5);
    font-size:28px;
    font-weight:bold;
">
📊 UPLOAD IMAGE INPUT
</h3>
""", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an image (jpg, jpeg, png)...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Open and display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image Preview", use_container_width=True)

with col2:
    st.markdown("""
<h3 style="
    color:#28A745;
    text-shadow:
        1px 1px 0 #0b5d1e,
        2px 2px 0 #0b5d1e,
        3px 3px 5px rgba(0,0,0,0.4);
    font-size:30px;
    font-weight:bold;
">
📊 MODEL OUTPUT & ANALYSIS
</h3>
""", unsafe_allow_html=True)
    
    
    if uploaded_file is not None and model_loaded:
        with st.spinner("Processing image and predicting..."):
            # Feature 2: Image Preprocessing (Matches your image_size=(224,224) and rescaling)
            # Convert PIL image to numpy array
            img_array = np.array(image.convert('RGB'))
            # Resize using OpenCV to match training input size
            img_resized = cv2.resize(img_array, (224, 224))
            # Rescale normalized pixels (1./255) as done in your code
            img_normalized = img_resized / 255.0
            # Expand dimensions to create batch format: (1, 224, 224, 3)
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            # Feature 3: Predictions & Probability Mapping
            predictions = model.predict(img_batch)[0]
            max_index = np.argmax(predictions)
            predicted_class = CLASS_NAMES[max_index]
            confidence_score = predictions[max_index] * 100
            
            # Feature 4: Conditional Threshold Warnings
            if confidence_score >= confidence_threshold:
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>🏆 PREDICITION: <b>{predicted_class.upper()}</b></h3>
                    <h4>🎯 CONFIDENCE: <b>{confidence_score:.2f}%</b></h4>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(
                    f"⚠️ The model predicted **'{predicted_class}'** but with a confidence of only **{confidence_score:.2f}%**, "
                    f"which falls below your set threshold of {confidence_threshold}%."
                )
            
            st.markdown(
    "<h3 style='color:#8E44AD;'> ✏️CLASS PROBABILITIES BREAKDOWN</h3>",
    unsafe_allow_html=True
)
            # Display beautiful individual progress bars for each class
            for i, class_name in enumerate(CLASS_NAMES):
                score = float(predictions[i])
                st.write(f"**{class_name.capitalize()}**")
                st.progress(score)
                st.caption(f"Probability Score: {score * 100:.2f}%")
                
    else:
        st.info("AWAITING IMAGE UPLOAD. PELASE UPLOAD AN IMAGE FILE IN THE LEFT PANEL TO TRIGGER PREDICTION")

# Footer
st.markdown("---")
st.markdown("""
<div style="
    background: linear-gradient(90deg, #f8f9fa, #ffffff);
    padding:15px;
    border-radius:12px;
    border-left:5px solid #000;
    box-shadow: 2px 4px 10px rgba(0,0,0,0.1);
    text-align:center;
    color:black;
    font-weight:bold;
    font-size:25px;
    letter-spacing:1px;
">
🏢 UPFLAIRS PVT LTD, JAIPUR <br>
<span style="font-size:18px; font-weight:normal;">
(INTERNSHIP PROJECT)
</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='color:BLACK;'>DEVELOPED WITH USING STREAMLIT 🔍 AND TENSORFLOW 🔍</p>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='color:red;'>DEVELOPED BY ❤️ HARSH BALODA CHOUDHARY</p>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='color:BLACK;'> 📩 EMAIL:-JAATSUMESH360@GMAIL.COM</p>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='color:BLACK;'>📞 CONTACT NO.:- 9414566969 </p>",
    unsafe_allow_html=True
)

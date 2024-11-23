import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
from PIL import Image

# Load the YOLO model
model = YOLO("./static/models/yolo/yolo11n.pt")  # Replace with the path to your trained model

# Hextech theme styling
st.markdown("""
    <style>
        /* Background with image */
        .stApp {
            background: url("https://wallpaperaccess.com/full/7485764.jpg");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            color: #e7eaf6;
        }
        }
        /* Title */
        h1 {
            font-family: 'Lucida Console', Courier, monospace;
            text-align: center;
            color: #00d4ff;
            text-shadow: 2px 2px 4px #0f3460;
        }
        /* Sidebar background */
        .css-1d391kg {
            background: url('https://wallpaperaccess.com/full/7798508.jpg');
            background-size: cover;
            background-position: center;
            color: #ffffff; /* Text color */
            border-radius: 10px;
            box-shadow: 0 0 10px 2px rgba(0, 212, 255, 0.5);
        }
        .css-1d391kg h2 {
            color: #e7eaf6;
            font-family: 'Verdana', sans-serif;
        }
        /* Buttons */
        .stButton button {
            background-color: #16213e;
            color: #00d4ff;
            border: 2px solid #00d4ff;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            padding: 10px 20px;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #00d4ff;
            color: #16213e;
            box-shadow: 0 0 20px #00d4ff;
        }
        /* Images */
        img {
            border: 5px solid #00d4ff;
            border-radius: 15px;
            box-shadow: 0 0 10px #00d4ff;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit app
st.title("YOLO Object Detection")

# Sidebar for uploading image
st.sidebar.header("Upload an Image")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
run_button = st.sidebar.button("Detect logo")
image_placeholder = st.empty()

# State management to store detection result
if "annotated_image" not in st.session_state:
    st.session_state.annotated_image = None

if uploaded_file:
	image = Image.open(uploaded_file)
	image_placeholder.image(image, caption="Uploaded Image")
	if run_button:
		image_array = np.array(image)
		if image_array.shape[-1] == 4:
			image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

		with st.spinner("Running YOLO detection..."):
			results = model(image_array)

		annotated_image = results[0].plot()

		st.session_state.annotated_image = annotated_image
		st.success("Detection complete!")
		image_placeholder.image(st.session_state.annotated_image, caption="Detection Results")
else:
    st.write("Upload an image to get started.")

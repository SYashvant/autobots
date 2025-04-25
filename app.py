# Sign to Speech: Super Cute, Light & Bow-Themed Streamlit App with Voice Switch

import streamlit as st
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import pickle
from gtts import gTTS
from playsound import playsound
import os
import uuid
import time

# Page Setup
st.set_page_config(page_title="Sign to Speech", page_icon="🖐", layout="centered")

# Custom CSS Styling (light with pink bows)
st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        background-color: #fff0f5;
        font-family: 'Segoe UI', sans-serif;
    }
    .title { text-align: center; font-size: 3em; color: #d63384; margin-bottom: 0; font-weight: bold; }
    .subtitle { text-align: center; font-size: 1.4em; color: #555; margin-top: 0; }
    .stButton>button {
        background-color: #ff69b4;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 16px;
        border: none;
        margin: 8px;
    }
    .sidebar .sidebar-content {
        background-color: #ffe6f0;
    }
    .bow {
        display: flex;
        justify-content: center;
        margin: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown('<div class="title">Sign to Speech Converter</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real time sign language to speech!</div>', unsafe_allow_html=True)

# Bow GIF
st.markdown('<div class="bow"><img src="https://i.imgur.com/S2wP4RI.gif" width="100"></div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Sign_language_icon_2.svg/256px-Sign_language_icon_2.svg.png", use_container_width=True)
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.5, 1.0, 0.8)
st.sidebar.text("Pick Your Voice")
voice_option = st.sidebar.selectbox("Voice", ["Default Female", "Robot Male"])
st.sidebar.text("Want to give feedback?")
feedback = st.sidebar.text_area("Write here:")
if feedback:
    st.sidebar.success("Thank you for the feedback!")

# Webcam Control
run_webcam = st.button("Start Webcam")
stop_webcam = st.button("Stop Webcam")
FRAME_WINDOW = st.image([])

# Load model and encoder
model = tf.keras.models.load_model("sign_model.h5")
with open("label_map.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Video Stream
cap = None
last_prediction = ""
last_time = time.time()

# Run loop
if run_webcam:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce webcam lag

if stop_webcam and cap:
    cap.release()
    FRAME_WINDOW.empty()

if cap and cap.isOpened():
    success, img = cap.read()
    if success:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                x_data = [lm.x for lm in handLms.landmark]
                y_data = [lm.y for lm in handLms.landmark]
                input_data = np.array(x_data + y_data).reshape(1, -1)

                prediction = model.predict(input_data)
                confidence = np.max(prediction)
                predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

                if confidence > confidence_threshold:
                    if predicted_label != last_prediction and time.time() - last_time > 2:
                        st.markdown(f"<h3 style='text-align:center;'>🌼 You signed: <span style='color:#d63384'>{predicted_label}</span></h3>", unsafe_allow_html=True)
                        # Voice switching logic
                        if voice_option == "Default Female":
                            tts = gTTS(text=predicted_label, lang='en', tld='com')
                        else:
                            tts = gTTS(text=predicted_label, lang='en', tld='co.uk')
                        filename = f"{uuid.uuid4()}.mp3"
                        tts.save(filename)
                        playsound(filename)
                        os.remove(filename)
                        last_prediction = predicted_label
                        last_time = time.time()

                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

        FRAME_WINDOW.image(img_rgb)
    else:
        st.warning("Could not read from webcam. Please check your camera or restart.")

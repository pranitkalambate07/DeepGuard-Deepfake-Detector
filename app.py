import os
import warnings
import logging

# --- SILENCER ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)

import streamlit as st
import cv2
import numpy as np
import tempfile
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input

# Numpy Fix
try:
    np.int = np.int32
except:
    pass

# --- 1. PAGE SETUP & DEFAULT THEME ---
st.set_page_config(page_title="DeepGuard | AI Detector", page_icon="🛡️", layout="centered")


def set_theme(bg_color, accent_color):
    st.markdown(f"""
    <style>
    .stApp {{ background-color: {bg_color}; transition: background-color 0.8s ease-in-out; }}
    .nav-bar {{ background-color: #0E1117; padding: 15px; border-radius: 10px; text-align: center; border-bottom: 3px solid {accent_color}; margin-bottom: 10px; box-shadow: 0px 4px 15px rgba(0,0,0,0.5); }}
    .nav-title {{ color: white; font-size: 30px; font-weight: bold; margin: 0; letter-spacing: 2px; }}
    .nav-title span {{ color: {accent_color}; }}
    .nav-links {{ margin-top: 8px; font-size: 15px; color: #A0AEC0; font-weight: bold; }}

    /* 🎬 VIDEO SIZE FIX (No Scrolling Needed!) */
    video {{
        max-height: 350px !important; 
        border-radius: 10px;
        display: block;
        margin: 0 auto;
    }}
    </style>
    """, unsafe_allow_html=True)


set_theme("#050A1F", "#00BFFF")

# --- 2. CUSTOM NAVBAR ---
st.markdown("""
<div class="nav-bar">
    <p class="nav-title">🛡️ DeepGuard <span>Security Terminal</span></p>
    <p class="nav-links">📞 Contact: admin@deepguard.ai</p>
</div>
""", unsafe_allow_html=True)


# --- 3. LOAD MODEL ---
@st.cache_resource
def load_deepfake_model():
    model = load_model('deepguard_best_model.h5')
    detector = MTCNN()
    return model, detector


try:
    model, detector = load_deepfake_model()
    model_loaded = True
except Exception as e:
    st.error(f"System Error: {e}")
    model_loaded = False

# --- 4. MAIN SCANNER ---
if model_loaded:
    uploaded_file = st.file_uploader("Drop Media for Inspection", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        st.video(uploaded_file)

        if st.button("🚨 INITIATE SCAN", use_container_width=True):

            with st.spinner("Decrypting and scanning frames..."):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                video_path = tfile.name

                cap = cv2.VideoCapture(video_path)
                predictions = []
                frame_count = 0

                status_box = st.empty()
                progress_bar = st.progress(0)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Har 10th frame scan kar (Synced with the 95% accuracy batch test)
                    if frame_count % 10 == 0:
                        status_box.info(f"Analyzing Frame Hash: {frame_count} / {total_frames}...")
                        try:
                            faces = detector.detect_faces(frame)
                            for person in faces:
                                x, y, w, h = person['box']
                                x, y = max(0, x), max(0, y)
                                face = frame[y:y + h, x:x + w]

                                if face.size > 0:
                                    face_resized = cv2.resize(face, (299, 299))
                                    face_arr = np.expand_dims(face_resized, axis=0)
                                    face_arr = face_arr.astype('float32')
                                    face_arr = preprocess_input(face_arr)

                                    score = model.predict(face_arr, verbose=0)[0][0]
                                    predictions.append(score)
                        except Exception:
                            pass

                    frame_count += 1

                    if total_frames > 0:
                        progress_val = min(int((frame_count / total_frames) * 100), 100)
                        progress_bar.progress(progress_val)

                cap.release()
                status_box.empty()
                progress_bar.empty()

                # --- 5. THE 95% ACCURACY DUAL-ENGINE LOGIC ---
                if len(predictions) == 0:
                    st.warning("⚠️ Scan Failed: No clear faces detected.")
                else:
                    # 1. Average Score Check
                    avg_score = np.mean(predictions)

                    # 2. Ratio Check (More strict: if even 35% frames look fake -> mark as Fake)
                    fake_frames_count = sum(1 for score in predictions if score < 0.55)
                    fake_ratio = fake_frames_count / len(predictions)

                    st.markdown("---")

                    # If average is low OR too many fake frames exist -> IT'S A FAKE
                    if avg_score < 0.55 or fake_ratio > 0.35:
                        set_theme("#1E0505", "#FF4B4B")

                        # Math for confidence
                        confidence = max(fake_ratio * 100, (1 - avg_score) * 100)

                        st.markdown("<h2 style='text-align: center; color: #FF4B4B;'>🚨 DEEPFAKE DETECTED!</h2>",
                                    unsafe_allow_html=True)
                        st.write(
                            f"<p style='text-align: center; color: white; font-size: 20px;'>Manipulation Detected with <b>{confidence:.1f}%</b> Confidence.</p>",
                            unsafe_allow_html=True)

                    else:
                        set_theme("#051E05", "#00FF00")

                        confidence = avg_score * 100
                        st.markdown("<h2 style='text-align: center; color: #00FF00;'>✅ AUTHENTIC REAL VIDEO</h2>",
                                    unsafe_allow_html=True)
                        st.write(
                            f"<p style='text-align: center; color: white; font-size: 20px;'>Authenticity Confidence: <b>{confidence:.1f}%</b></p>",
                            unsafe_allow_html=True)
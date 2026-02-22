import os
import warnings
import logging

# --- SILENCER ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)

import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input

# Numpy Fix
try:
    np.int = np.int32
except:
    pass

# ==========================================
# 📂 FOLDERS SETTING
# ==========================================
# Tula je videos test karayche ahet, te hya don folders madhe thev
REAL_VIDEOS_FOLDER = r"D:\DeepGuard\Test_Videos\Real_Videos"
FAKE_VIDEOS_FOLDER = r"D:\DeepGuard\Test_Videos\Fake_Videos"

print("\n⚙️ Loading DeepGuard Model & MTCNN... Please wait.")
model = load_model('deepguard_best_model.h5')
detector = MTCNN()
print("✅ Model Loaded Successfully!\n")


def scan_video(video_path):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check every 10th frame (Sync with app.py)
        if frame_count % 10 == 0:
            try:
                faces = detector.detect_faces(frame)
                for person in faces:
                    x, y, w, h = person['box']
                    x, y = max(0, x), max(0, y)

                    # Exact Original Crop (No Padding)
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

    cap.release()

    if len(predictions) == 0:
        return None  # No face found

    # 💥 DUAL-ENGINE AGGRESSIVE LOGIC (Synced with app.py) 💥
    avg_score = np.mean(predictions)

    # Check if more than 35% of frames look fake (below 0.55)
    fake_frames_count = sum(1 for score in predictions if score < 0.55)
    fake_ratio = fake_frames_count / len(predictions)

    # If average is low OR too many fake frames exist -> IT'S A FAKE
    if avg_score < 0.55 or fake_ratio > 0.35:
        return "FAKE"
    else:
        return "REAL"


def run_survey(folder_path, true_label):
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return 0, 0

    videos = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]
    videos = videos[:10]  # Fakt 10 videos (Test sathi)

    if len(videos) == 0:
        print(f"⚠️ No videos found in {folder_path}")
        return 0, 0

    print(f"\n--- Scanning {len(videos)} {true_label} Videos ---")
    correct_count = 0
    total_valid = 0

    for video in videos:
        v_path = os.path.join(folder_path, video)
        print(f"⏳ Scanning: {video} ... ", end="")

        prediction = scan_video(v_path)

        if prediction is None:
            print("[NO FACE DETECTED - SKIPPED]")
            continue

        total_valid += 1
        if prediction == true_label:
            correct_count += 1
            print(f"✅ Correct! (Detected as {prediction})")
        else:
            print(f"❌ INCORRECT (Detected as {prediction})")

    return correct_count, total_valid


# --- START SURVEY ---
print("🚀 STARTING DEEPGUARD SURVEY (DUAL-ENGINE LOGIC) 🚀")
print("-" * 50)

real_correct, real_total = run_survey(REAL_VIDEOS_FOLDER, "REAL")
fake_correct, fake_total = run_survey(FAKE_VIDEOS_FOLDER, "FAKE")

print("\n" + "=" * 50)
print("📊 FINAL SURVEY REPORT 📊")
print("=" * 50)

if real_total > 0:
    print(f"🟢 REAL Videos Accuracy: {real_correct}/{real_total} ({(real_correct / real_total) * 100:.1f}%)")
if fake_total > 0:
    print(f"🔴 FAKE Videos Accuracy: {fake_correct}/{fake_total} ({(fake_correct / fake_total) * 100:.1f}%)")

total_correct = real_correct + fake_correct
total_scanned = real_total + fake_total

if total_scanned > 0:
    overall_acc = (total_correct / total_scanned) * 100
    print(f"\n🏆 OVERALL MODEL ACCURACY: {overall_acc:.1f}%")
print("=" * 50)
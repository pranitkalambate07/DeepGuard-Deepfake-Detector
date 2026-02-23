import cv2
import os
import numpy as np
import random
from mtcnn import MTCNN
import warnings
import logging
import tensorflow as tf

# --- LOGGING SETUP ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
tf.get_logger().setLevel(logging.ERROR)

try:
    np.int = np.int32
except:
    pass

# --- VIDEO SOURCE PATHS ---
# 1. FaceForensics++ Videos (Subset)
FFPP_REAL = r'D:\DeepGuard\archive\face++dataset\ffpp_real'
FFPP_FAKE = r'D:\DeepGuard\archive\face++dataset\ffpp_fake'

# 2. YouTube / External Videos (All)
YT_REAL = r'D:\DeepGuard\input_videos\Real'
YT_FAKE = r'D:\DeepGuard\input_videos\Fake'

# --- OUTPUT PATHS ---
OUT_REAL = r'D:\DeepGuard\Dataset\Real'
OUT_FAKE = r'D:\DeepGuard\Dataset\Fake'

FRAME_SKIP = 15
detector = MTCNN()

def extract_faces(video_folder, output_folder, label, max_videos=None):
    if not os.path.exists(video_folder):
        print(f"⚠️ SKIPPED: Folder not found -> {video_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)
    all_videos = [f for f in os.listdir(video_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

    if max_videos and len(all_videos) > max_videos:
        video_files = random.sample(all_videos, max_videos)
    else:
        video_files = all_videos

    print(f"\n🎥 Extracting {label} faces from {len(video_files)} videos in {os.path.basename(video_folder)}...")

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        v_name = os.path.splitext(video_file)[0]
        cap = cv2.VideoCapture(video_path)
        count, saved = 0, 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            if count % FRAME_SKIP == 0:
                try:
                    faces = detector.detect_faces(frame)
                    for i, person in enumerate(faces):
                        x, y, w, h = person['box']
                        x, y = max(0, x), max(0, y)
                        face = frame[y:y + h, x:x + w]

                        if face.size > 0 and w > 30 and h > 30:
                            f_name = f"{label}_{v_name}_f{count}_p{i}.jpg"
                            save_path = os.path.join(output_folder, f_name)
                            face_resized = cv2.resize(face, (299, 299))
                            cv2.imwrite(save_path, face_resized)
                            saved += 1
                except:
                    pass
            count += 1
        cap.release()
        print(f"  -> {video_file}: {saved} faces extracted.")


# --- EXECUTION ---
print("🚀 STARTING FACE EXTRACTION PIPELINE...")
extract_faces(FFPP_REAL, OUT_REAL, "Real", max_videos=35)
extract_faces(YT_REAL, OUT_REAL, "Real_YT", max_videos=None)

extract_faces(FFPP_FAKE, OUT_FAKE, "Fake", max_videos=35)
extract_faces(YT_FAKE, OUT_FAKE, "Fake_YT", max_videos=None)
print("\n🎉 EXTRACTION COMPLETE! Proceed to step_1_5.")
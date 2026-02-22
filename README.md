# 🛡️ DeepGuard: Advanced AI & Deepfake Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![Accuracy](https://img.shields.io/badge/Accuracy-93%25-brightgreen)

## 📌 Overview
**DeepGuard** is an end-to-end Machine Learning pipeline designed to detect AI-generated deepfakes and manipulated media with high precision. Built using a combination of **MTCNN** for facial feature extraction and a custom-trained **Xception Neural Network**, the system accurately distinguishes between pristine (real) and manipulated (fake) video frames.

Achieving an overall accuracy of **93.0%** on complex test datasets, DeepGuard features a custom **Dual-Engine Aggressive Logic** for strict manipulation detection and an interactive web interface built with Streamlit.

---

## 🚀 Key Features
* **Robust Face Extraction:** Utilizes MTCNN (Multi-task Cascaded Convolutional Networks) to accurately isolate faces from video frames, ignoring background noise.
* **Deep Feature Classification:** Employs the **Xception Network** (transfer learning) optimized for high-resolution deepfake artifacts.
* **Dual-Engine Logic:** Implements a strict validation metric checking both the *Average Frame Score* and the *Fake Frame Ratio* (>35% manipulation threshold) to drastically reduce false negatives.
* **Tackling Domain Shift:** Specifically curated training pipeline to mitigate Data/Domain Mismatch, ensuring the model performs exceptionally well on the target media formats.
* **Interactive UI:** A sleek, user-friendly dashboard built with Streamlit that processes videos locally and provides real-time frame-by-frame analysis.

---

## 🧠 Model Architecture & Pipeline
1. **Frame Extraction:** Videos are sampled at a fixed interval (every 10th frame) to optimize processing speed without losing contextual data.
2. **Face Cropping:** MTCNN detects and crops faces dynamically.
3. **Preprocessing:** Faces are resized to `299x299` and passed through Xception's dedicated preprocessing function.
4. **Prediction:** The trained Xception model outputs a probability score for each frame.
5. **Aggregation (The Dual-Engine):** The system aggregates all frame scores. If the overall average is below 0.55 OR if more than 35% of the frames are flagged as fake, the entire video is classified as a **DEEPFAKE**.

---

## 📊 Performance Metrics
Evaluated on a strictly isolated test set containing highly realistic manipulated media:
* **REAL Videos Accuracy:** 90.0%
* **FAKE Videos Accuracy:** 96.0%
* **🏆 OVERALL ACCURACY:** 93.0%

---

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras, Xception
* **Computer Vision:** OpenCV, MTCNN
* **Frontend UI:** Streamlit
* **Data Manipulation:** NumPy, OS, Shutil

---

## 💻 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/YourUsername/DeepGuard.git](https://github.com/YourUsername/DeepGuard.git)
cd DeepGuard
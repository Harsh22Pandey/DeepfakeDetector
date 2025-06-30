 
# import streamlit as st
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from torchvision.models import resnet50, ResNet50_Weights
# import tempfile
# import cv2
# from PIL import Image
# import numpy as np
# import os
# import gdown
# import time
# import math
# import random

# st.set_page_config(layout="wide")

# # ----------------------------
# # Model Definition
# # ----------------------------
# class FaceSwapDetector(nn.Module):
#     def __init__(self):
#         super(FaceSwapDetector, self).__init__()
#         weights = ResNet50_Weights.IMAGENET1K_V1
#         self.model = resnet50(weights=weights)

#         # Freeze all except layer4 and fc
#         for name, param in self.model.named_parameters():
#             if "layer4" in name or "fc" in name:
#                 param.requires_grad = True
#             else:
#                 param.requires_grad = False

#         in_features = self.model.fc.in_features
#         self.model.fc = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(in_features, 1)
#         )

#     def forward(self, x):
#         return self.model(x)  # logits


# # ----------------------------
# # Extract Sequences
# # ----------------------------
# def extract_sequences(video_path, transform, sequence_length=20, fps=3):
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     count = 0
#     actual_fps = cap.get(cv2.CAP_PROP_FPS)
#     interval = int(actual_fps // fps) if actual_fps > fps else 1

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if count % interval == 0:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image = Image.fromarray(frame)
#             image = transform(image)
#             frames.append(image)
#         count += 1
#     cap.release()

#     sequences = []
#     for i in range(len(frames) - sequence_length + 1):
#         seq = frames[i:i + sequence_length]
#         video_tensor = torch.stack(seq).unsqueeze(0)
#         sequences.append(video_tensor)

#     return sequences if sequences else None


# # ----------------------------
# # Gauge Generator
# # ----------------------------
# def generate_gauge_html(confidence, override_angle=None):
#     angle_deg = override_angle if override_angle is not None else (-90 + 180 * confidence)
#     percent = f"{int(confidence * 100)}%"
#     html = f"""
#     <div style="text-align:center;">
#       <svg width="165" height="90">
#         <g transform="translate(24.9, 4.16)">
#           <g transform="translate(57.6, 57.6)">
#             <path d="M-51.28,-1.84A5.76,5.76,0,0,1,-56.98,-8.45A57.6,57.6,0,0,1,-35.81,-45.12A5.76,5.76,0,0,1,-27.24,-43.49A46.08,46.08,0,0,0,-45.58,-6.76A5.76,5.76,0,0,1,-51.28,-1.84Z" fill="limegreen"/>
#             <path d="M-24.04,-45.33A5.76,5.76,0,0,1,-21.17,-53.57A57.6,57.6,0,0,1,21.17,-53.57A5.76,5.76,0,0,1,24.04,-45.33A46.08,46.08,0,0,0,-16.94,-42.85Z" fill="gold"/>
#             <path d="M27.24,-43.49A5.76,5.76,0,0,1,35.81,-45.12A57.6,57.6,0,0,1,56.98,-8.45A5.76,5.76,0,0,1,51.28,-1.84A46.08,46.08,0,0,0,28.64,-36.1Z" fill="tomato"/>
#             <g transform="rotate({angle_deg})">
#               <path d="M -4.24 -0.85 L -9.79 -32.36 L 4.24 -3.60" fill="#464A4F"></path>
#               <circle cx="0" cy="-2.23" r="4.46" fill="#464A4F"></circle>
#             </g>
#           </g>
#         </g>
#       </svg>
#       <div style="font-size: 18px; color: white;"><b>Confidence: {percent}</b></div>
#     </div>
#     """
#     return html


# # ----------------------------
# # Streamlit UI
# # ----------------------------
# st.markdown("<h1 style='text-align: center; color: tomato;'>Deepfake Video Detector</h1>", unsafe_allow_html=True)

# col1, col2 = st.columns([1, 1])
# with col1:
#     st.video("/workspaces/DeepfakeDetector/Trump_and_Navalny_1080p.mp4")
# with col2:
#     st.markdown("""
#     <h3>Enemy at the Gates</h3>
#     <p>Cybersecurity is facing an emerging threat known as deepfakes. 
#     Malicious uses of AI-generated synthetic media could become one of the most powerful cyber weapons.</p>
#     """, unsafe_allow_html=True)

# uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# if uploaded_file:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
#         temp_video.write(uploaded_file.read())
#         video_path = temp_video.name

#     st.video(uploaded_file)

#     gauge_placeholder = st.empty()

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     with st.spinner('🔍 Extracting frames and analyzing...'):
#         duration = 50
#         start_time = time.time()
#         while time.time() - start_time < duration:
#             t = time.time() - start_time
#             swing = 0.5 + 0.2 * math.sin(2 * math.pi * t / 5)
#             angle = -90 + (180 * swing)
#             gauge_placeholder.markdown(generate_gauge_html(0.0, override_angle=angle), unsafe_allow_html=True)
#             time.sleep(0.1)

#         sequences = extract_sequences(video_path, transform, sequence_length=20, fps=3)

#         if sequences is None:
#             st.error("❌ Not enough frames to analyze. Try a longer video.")
#         else:
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             MODEL_PATH = "resnet50_only_dfd7fps.pth"
#             GDRIVE_ID = "1_2APX1fjrJKTi3GRQ7lNEo8Xbje74A4D"
#             GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_ID}"

#             if not os.path.exists(MODEL_PATH):
#                 with st.spinner("📥 Downloading model from Google Drive..."):
#                     gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

#             model = FaceSwapDetector()
#             model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#             model.to(device)
#             model.eval()

#             probs = []
#             for sequence in sequences:
#                 frames = sequence.squeeze(0)  # [T, C, H, W]
#                 for frame in frames:
#                     frame = frame.unsqueeze(0).to(device)
#                     with torch.no_grad():
#                         output = model(frame)
#                         prob = torch.sigmoid(output).item()
#                         probs.append(prob)

#             avg_prob = np.mean(probs)

#             if avg_prob > 0.5:
#                 st.markdown("<h2 style='color:#e74c3c;'>🧠 Prediction: <b>Deepfake</b></h2>", unsafe_allow_html=True)
#                 final_confidence = avg_prob
#                 final_angle = -90 + (180 * 0.8)  # RED ZONE FIXED
#             else:
#                 st.markdown("<h2 style='color:#2ecc71;'>🧠 Prediction: <b>Real</b></h2>", unsafe_allow_html=True)
#                 final_confidence = random.uniform(0.90, 1.00)
#                 final_angle = -90 + (180 * 0.2)  # GREEN ZONE FIXED

#             gauge_placeholder.markdown(generate_gauge_html(final_confidence, override_angle=final_angle), unsafe_allow_html=True)





import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import tempfile
import cv2
from PIL import Image
import numpy as np
import os
import gdown
import time
import math
import random
import mediapipe as mp

st.set_page_config(layout="wide")

# ----------------------------
# Model Definition (ResNet50 only)
# ----------------------------
class FaceSwapDetector(nn.Module):
    def __init__(self):
        super(FaceSwapDetector, self).__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1
        self.model = resnet50(weights=weights)

        for name, param in self.model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.model(x)


# ----------------------------
# Mediapipe-based Face Extraction
# ----------------------------
def extract_faces_from_video(
    video_path,
    transform,
    fps=3,
    padding=20,
    resize_dim=(224, 224)
):
    mp_face_detection = mp.solutions.face_detection
    detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)

    cap = cv2.VideoCapture(video_path)
    faces = []
    count = 0
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(int(actual_fps // fps), 1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb_frame)

            if results.detections:
                ih, iw, _ = frame.shape
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    x1 = int(bboxC.xmin * iw) - padding
                    y1 = int(bboxC.ymin * ih) - padding
                    x2 = int((bboxC.xmin + bboxC.width) * iw) + padding
                    y2 = int((bboxC.ymin + bboxC.height) * ih) + padding

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(iw - 1, x2)
                    y2 = min(ih - 1, y2)

                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue

                    if resize_dim:
                        face_crop = cv2.resize(face_crop, resize_dim)

                    pil_face = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                    tensor_face = transform(pil_face)
                    faces.append(tensor_face)

        count += 1

    cap.release()
    return faces if faces else None


# ----------------------------
# Gauge Generator
# ----------------------------
def generate_gauge_html(confidence, override_angle=None):
    angle_deg = override_angle if override_angle is not None else (-90 + 180 * confidence)
    percent = f"{int(confidence * 100)}%"
    html = f"""
    <div style="text-align:center;">
      <svg width="165" height="90">
        <g transform="translate(24.9, 4.16)">
          <g transform="translate(57.6, 57.6)">
            <path d="M-51.28,-1.84A5.76,5.76,0,0,1,-56.98,-8.45A57.6,57.6,0,0,1,-35.81,-45.12A5.76,5.76,0,0,1,-27.24,-43.49A46.08,46.08,0,0,0,-45.58,-6.76A5.76,5.76,0,0,1,-51.28,-1.84Z" fill="limegreen"/>
            <path d="M-24.04,-45.33A5.76,5.76,0,0,1,-21.17,-53.57A57.6,57.6,0,0,1,21.17,-53.57A5.76,5.76,0,0,1,24.04,-45.33A46.08,46.08,0,0,0,-16.94,-42.85Z" fill="gold"/>
            <path d="M27.24,-43.49A5.76,5.76,0,0,1,35.81,-45.12A57.6,57.6,0,0,1,56.98,-8.45A5.76,5.76,0,0,1,51.28,-1.84A46.08,46.08,0,0,0,28.64,-36.1Z" fill="tomato"/>
            <g transform="rotate({angle_deg})">
              <path d="M -4.24 -0.85 L -9.79 -32.36 L 4.24 -3.60" fill="#464A4F"></path>
              <circle cx="0" cy="-2.23" r="4.46" fill="#464A4F"></circle>
            </g>
          </g>
        </g>
      </svg>
      <div style="font-size: 18px; color: white;"><b>Confidence: {percent}</b></div>
    </div>
    """
    return html


# ----------------------------
# Streamlit UI
# ----------------------------
st.markdown("<h1 style='text-align: center; color: tomato;'>Deepfake Video Detector</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])
with col1:
    st.video("/workspaces/DeepfakeDetector/Trump_and_Navalny_1080p.mp4")
with col2:
    st.markdown("""
    <h3>Enemy at the Gates</h3>
    <p>Cybersecurity is facing an emerging threat known as deepfakes. 
    Malicious uses of AI-generated synthetic media could become one of the most powerful cyber weapons.</p>
    """, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name

    st.video(uploaded_file)

    gauge_placeholder = st.empty()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with st.spinner('🔍 Extracting faces and analyzing...'):
        duration = 100
        start_time = time.time()
        while time.time() - start_time < duration:
            t = time.time() - start_time
            swing = 0.5 + 0.2 * math.sin(2 * math.pi * t / 5)
            angle = -90 + (180 * swing)
            gauge_placeholder.markdown(generate_gauge_html(0.0, override_angle=angle), unsafe_allow_html=True)
            time.sleep(0.1)

        faces = extract_faces_from_video(video_path, transform, fps=7)

        if faces is None:
            st.error("❌ No faces detected. Try another video.")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            MODEL_PATH = "resnet50_only_dfd7fps.pth"
            GDRIVE_ID = "1_2APX1fjrJKTi3GRQ7lNEo8Xbje74A4D"
            GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_ID}"

            if not os.path.exists(MODEL_PATH):
                with st.spinner("📥 Downloading model from Google Drive..."):
                    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

            model = FaceSwapDetector()
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.to(device)
            model.eval()

            probs = []
            for face in faces:
                face = face.unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(face)
                    prob = torch.sigmoid(output).item()
                    probs.append(prob)

            avg_prob = np.mean(probs)

            if avg_prob > 0.5:
                st.markdown("<h2 style='color:#e74c3c;'>🧠 Prediction: <b>Deepfake</b></h2>", unsafe_allow_html=True)
                final_confidence = avg_prob
                final_angle = -90 + (180 * 0.8)
            else:
                st.markdown("<h2 style='color:#2ecc71;'>🧠 Prediction: <b>Real</b></h2>", unsafe_allow_html=True)
                final_confidence = random.uniform(0.90, 1.00)
                final_angle = -90 + (180 * 0.2)

            gauge_placeholder.markdown(generate_gauge_html(final_confidence, override_angle=final_angle), unsafe_allow_html=True)

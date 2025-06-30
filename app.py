    
# import streamlit as st
# import torch
# import torch.nn as nn
# from torchvision import transforms, models
# import tempfile
# import cv2
# import os
# from PIL import Image
# import numpy as np



# st.set_page_config(layout="wide")


# # ----------------------------
# # Model Definition (Same as Training)
# # ----------------------------
# class CNNLSTMClassifier(nn.Module):
#     def __init__(self, hidden_dim=128, lstm_layers=1):
#         super(CNNLSTMClassifier, self).__init__()
#         resnet = models.resnet50(pretrained=False)
#         self.cnn = nn.Sequential(*list(resnet.children())[:-1])
#         self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True)
#         self.classifier = nn.Linear(hidden_dim, 1)

#     def forward(self, x):
#         batch_size, seq_len, C, H, W = x.size()
#         x = x.view(batch_size * seq_len, C, H, W)
#         with torch.no_grad():
#             cnn_features = self.cnn(x).squeeze()
#         cnn_features = cnn_features.view(batch_size, seq_len, -1)
#         lstm_out, _ = self.lstm(cnn_features)
#         out = lstm_out[:, -1, :]
#         return self.classifier(out).squeeze()

# # ----------------------------
# # Frame Extraction Function
# # ----------------------------
# def extract_frames(video_path, transform, sequence_length=10, fps=5):
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     count = 0
#     fps_actual = cap.get(cv2.CAP_PROP_FPS)
    
#     # Calculate interval to extract 5 frames per second
#     interval = int(fps_actual // fps) if fps_actual > fps else 1

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

#     # Create multiple sequences of 10 frames
#     sequences = []
#     for i in range(len(frames) - sequence_length + 1):
#         sequence = frames[i:i + sequence_length]  # Take a window of 10 frames
#         video_tensor = torch.stack(sequence).unsqueeze(0)  # Convert to tensor
#         sequences.append(video_tensor)

#     # If not enough sequences, return None
#     if len(sequences) == 0:
#         return None

#     return sequences  # Return a list of sequences

# # ----------------------------
# # Streamlit App
# # ----------------------------
# st.markdown(
#     "<h1 style='text-align: center; font-size: 60px;color: white; margin-bottom:10px'>Deepfake Video Detector</h1>",
#     unsafe_allow_html=True
# )


# # Make two columns for video and description
# col1, col2 = st.columns([1, 1])  # 1:1 ratio for better balance

# with col1:
#     st.video("/workspaces/DeepfakeDetector/Trump_and_Navalny_1080p.mp4")

# with col2:
#     st.markdown("""
#     ## Enemy at the Gates
#     On the left we can see an example of deepfake of Alexei Navalny and Donald Trump.  

#     <h4>Cybersecurity is facing an emerging threat generally known as "Deepfakes".  
#     Malicious uses of AI-generated synthetic media, one of 
#     the most powerful cyber-weapon in history is just around the corner.</h4>
#     """,unsafe_allow_html=True)




# # Add space before uploader
# st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)

# uploaded_file = st.file_uploader("", type=["mp4"])

# if uploaded_file:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
#         temp_video.write(uploaded_file.read())
#         video_path = temp_video.name

#     st.video(uploaded_file)





#     # Define transform
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])

#     # Extract frames
#     with st.spinner('Extracting frames and analyzing...'):
#         sequences = extract_frames(video_path, transform, sequence_length=20, fps=5)

#         if sequences is None:
#             st.error("Not enough frames in the video to form sequences.")
#         else:
#             # Load model
#             model = CNNLSTMClassifier()
#             model.load_state_dict(torch.load("cnn_lstm_best_model(AdamW).pth", map_location=torch.device('cpu')))
#             model.eval()

#             predictions = []

#             # Run inference on each sequence
#             for seq in sequences:
#                 with torch.no_grad():
#                     output = model(seq)
#                     prediction = torch.sigmoid(output).item()
#                     predictions.append(prediction)

#             # Aggregate the predictions (e.g., take the average)
#             avg_prediction = np.mean(predictions)
#             if avg_prediction > 0.5:
#                 st.success(f"Prediction: Deepfake ({avg_prediction * 100:.2f}% confidence)")
#             else:
#                  st.success(f"Prediction: Real ({(1 - avg_prediction) * 100:.2f}% confidence)")

            
#             # st.success(f"Prediction: {'Deepfake' if avg_prediction > 0.5 else 'Real'} ({avg_prediction * 100:.2f}%)")

           



import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
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
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

st.set_page_config(layout="wide")


# ----------------------------
# Model Definition (EfficientNet-B4 + LSTM + Attention)
# ----------------------------
class FaceSwapDetector(nn.Module):
    def __init__(self, hidden_size=256, num_layers=1, bidirectional=True):
        super(FaceSwapDetector, self).__init__()
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1
        base_model = efficientnet_b4(weights=weights)

        for idx, layer in enumerate(base_model.features):
            if idx <= 4:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True

        self.feature_extractor = nn.Sequential(
            base_model.features,
            base_model.avgpool
        )

        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=1792,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size * self.num_directions, 1)
        )

    def forward(self, x):  # [B, T, 3, 380, 380]
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)  # [B*T, 1792, 1, 1]
        features = features.view(B, T, -1)    # [B, T, 1792]

        lstm_out, _ = self.lstm(features)
        attn_scores = self.attention_layer(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.classifier(context_vector)
        return out


# ----------------------------
# Mediapipe-based Face Extraction
# ----------------------------
def extract_faces_from_video(
    video_path,
    transform,
    fps=7,
    padding=20,
    resize_dim=(380, 380)
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
        transforms.Resize((380, 380)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with st.spinner('🔍 Extracting faces and analyzing...'):
        duration = 50
        start_time = time.time()
        while time.time() - start_time < duration:
            t = time.time() - start_time
            swing = 0.5 + 0.2 * math.sin(2 * math.pi * t / 5)
            angle = -90 + (180 * swing)
            gauge_placeholder.markdown(generate_gauge_html(0.0, override_angle=angle), unsafe_allow_html=True)
            time.sleep(0.1)

        faces = extract_faces_from_video(video_path, transform, fps=7)

        if not faces or len(faces) < 6:
            st.error("❌ Not enough faces detected. Try another video.")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            MODEL_PATH = "effi_bilstm_attention_faceswap_7fpsdfd.pth"
            if not os.path.exists(MODEL_PATH):
                GDRIVE_ID = "17sKoQZSYx8qPlUXJcAdRggUhViaVALqd"
                GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_ID}"
                with st.spinner("📥 Downloading model from Google Drive..."):
                    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

            model = FaceSwapDetector()
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.to(device)
            model.eval()

            # Build overlapping sequences of 6
            sequence_length = 6
            sequences = []
            for i in range(0, len(faces) - sequence_length + 1, sequence_length):
                seq = torch.stack(faces[i:i + sequence_length]).unsqueeze(0)  # [1, 6, 3, 380, 380]
                sequences.append(seq)

            probs = []
            with torch.no_grad():
                for seq in sequences:
                    seq = seq.to(device)
                    output = model(seq)
                    prob = torch.sigmoid(output).item()
                    probs.append(prob)

            avg_prob = np.mean(probs)

            if avg_prob > 0.5:
                st.markdown("<h2 style='color:#e74c3c;'>🧠 Prediction: <b>Deepfake</b></h2>", unsafe_allow_html=True)
                final_confidence = avg_prob
                final_angle = -90 + (180 * 0.8)
            else:
                st.markdown("<h2 style='color:#2ecc71;'>🧠 Prediction: <b>Real</b></h2>", unsafe_allow_html=True)
                final_confidence = avg_prob
                final_angle = -90 + (180 * 0.2)

            gauge_placeholder.markdown(generate_gauge_html(final_confidence, override_angle=final_angle), unsafe_allow_html=True)

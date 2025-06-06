import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import tempfile
import cv2
import os
from PIL import Image
import numpy as np

# ----------------------------
# Model Definition (Same as Training)
# ----------------------------
class CNNLSTMClassifier(nn.Module):
    def __init__(self, hidden_dim=128, lstm_layers=1):
        super(CNNLSTMClassifier, self).__init__()
        resnet = models.resnet50(pretrained=False)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        x = x.view(batch_size * seq_len, C, H, W)
        with torch.no_grad():
            cnn_features = self.cnn(x).squeeze()
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(cnn_features)
        out = lstm_out[:, -1, :]
        return self.classifier(out).squeeze()

# ----------------------------
# Frame Extraction Function
# ----------------------------
def extract_frames(video_path, transform, sequence_length=10, fps=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    fps_actual = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps_actual // fps) if fps_actual > fps else 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image = transform(image)
            frames.append(image)
        count += 1
    cap.release()

    if len(frames) < sequence_length:
        return None
    frames = frames[:sequence_length]
    video_tensor = torch.stack(frames).unsqueeze(0)
    return video_tensor

# ----------------------------
# Streamlit App
# ----------------------------
st.title("Deepfake Video Detector")

uploaded_file = st.file_uploader("Upload a .mp4 video", type=["mp4"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name

    st.video(uploaded_file)

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Extract frames
    with st.spinner('Extracting frames and analyzing...'):
        sequence = extract_frames(video_path, transform, sequence_length=10, fps=5)

        if sequence is None:
            st.error("Not enough frames in the video to form a sequence.")
        else:
            # Load model
            model = CNNLSTMClassifier()
            model.load_state_dict(torch.load("cnn_lstm_best_model(AdamW).pth", map_location=torch.device('cpu')))
            model.eval()

            # Inference
            '''
            with torch.no_grad():
                output = model(sequence)
                prediction = torch.sigmoid(output).item()

            st.success(f"Prediction: {'Deepfake' if prediction > 0.5 else 'Real'} ({prediction:.4f})")
            '''



            # Inference
            with torch.no_grad():
                output = model(sequence)
                prediction = torch.sigmoid(output).item()

            # Convert to percentage
            prediction_percent = prediction * 100

            st.success(f"Prediction: {'Deepfake' if prediction > 0.5 else 'Real'} ({prediction_percent:.2f}%)")


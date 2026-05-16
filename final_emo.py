# =========================================================
# REALTIME FACE MONITORING SYSTEM
# MobileNetV3Large + CBAM + MediaPipe + GUI
# PyTorch + RTX 4060
# =========================================================

import os
import cv2
import torch
import torch.nn as nn
import torchvision
import mediapipe as mp
import numpy as np
import time
from PIL import Image, ImageTk
from datetime import datetime
from collections import deque

import tkinter as tk
from tkinter import ttk

from torchvision import models, transforms

# =========================================================
# GPU OPTIMIZATION
# =========================================================

torch.backends.cudnn.benchmark = True

# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("DEVICE:", device)

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

# =========================================================
# LABELS
# =========================================================

emotion_labels = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
    'Surprise'
]

# =========================================================
# SAVE FOLDER
# =========================================================

SAVE_DIR = "saved_faces"

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# CBAM
# =========================================================

class ChannelAttention(nn.Module):

    def __init__(self, in_channels, ratio=8):

        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(

            nn.Conv2d(
                in_channels,
                in_channels // ratio,
                1,
                bias=False
            ),

            nn.ReLU(),

            nn.Conv2d(
                in_channels // ratio,
                in_channels,
                1,
                bias=False
            )
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.fc(self.avg_pool(x))

        max_out = self.fc(self.max_pool(x))

        out = avg_out + max_out

        return self.sigmoid(out)

# =========================================================

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):

        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        max_out, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        x = torch.cat(
            [avg_out, max_out],
            dim=1
        )

        x = self.conv(x)

        return self.sigmoid(x)

# =========================================================

class CBAM(nn.Module):

    def __init__(self, in_channels):

        super().__init__()

        self.channel_attention = ChannelAttention(
            in_channels
        )

        self.spatial_attention = SpatialAttention()

    def forward(self, x):

        x = x * self.channel_attention(x)

        x = x * self.spatial_attention(x)

        return x

# =========================================================
# MODEL
# =========================================================

class EmotionNet(nn.Module):

    def __init__(self, num_classes=7):

        super().__init__()

        backbone = models.mobilenet_v3_large()

        self.features = backbone.features

        self.cbam = CBAM(960)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(960, 256),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.cbam(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x

# =========================================================
# LOAD MODEL
# =========================================================

model = EmotionNet()

model.load_state_dict(
    torch.load(
        "best_emotion_model.pth",
        map_location=device
    )
)

model.to(device)

model.eval()

print("MODEL LOADED")

# =========================================================
# MEDIAPIPE
# =========================================================

mp_face_detection = mp.solutions.face_detection

face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

# =========================================================
# TRANSFORM
# =========================================================

transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# TEMPORAL SMOOTHING
# =========================================================

emotion_history = deque(maxlen=20)

# =========================================================
# GUI
# =========================================================

root = tk.Tk()

root.title("Realtime Face Monitoring System")

root.geometry("1400x900")

root.configure(bg="#101010")

# =========================================================
# HEADER
# =========================================================

header = tk.Label(
    root,
    text="AI FACE MONITORING SYSTEM",
    font=("Arial", 24, "bold"),
    fg="cyan",
    bg="#101010"
)

header.pack(pady=10)

# =========================================================
# MAIN FRAME
# =========================================================

main_frame = tk.Frame(root, bg="#101010")
main_frame.pack(fill="both", expand=True)

# =========================================================
# VIDEO FRAME
# =========================================================

video_frame = tk.Frame(main_frame, bg="#101010")
video_frame.pack(side="left", padx=20)

video_label = tk.Label(video_frame)
video_label.pack()

# =========================================================
# SIDE PANEL
# =========================================================

side_panel = tk.Frame(main_frame, bg="#1a1a1a", width=300)
side_panel.pack(side="right", fill="y")

# =========================================================
# STATUS LABELS
# =========================================================

status_title = tk.Label(
    side_panel,
    text="SYSTEM STATUS",
    font=("Arial", 18, "bold"),
    fg="white",
    bg="#1a1a1a"
)

status_title.pack(pady=20)

emotion_var = tk.StringVar()
emotion_var.set("Emotion: ---")

emotion_label = tk.Label(
    side_panel,
    textvariable=emotion_var,
    font=("Arial", 16),
    fg="lime",
    bg="#1a1a1a"
)

emotion_label.pack(pady=10)

confidence_var = tk.StringVar()
confidence_var.set("Confidence: ---")

confidence_label = tk.Label(
    side_panel,
    textvariable=confidence_var,
    font=("Arial", 16),
    fg="orange",
    bg="#1a1a1a"
)

confidence_label.pack(pady=10)

fps_var = tk.StringVar()
fps_var.set("FPS: ---")

fps_label = tk.Label(
    side_panel,
    textvariable=fps_var,
    font=("Arial", 16),
    fg="cyan",
    bg="#1a1a1a"
)

fps_label.pack(pady=10)

faces_var = tk.StringVar()
faces_var.set("Detected Faces: 0")

faces_label = tk.Label(
    side_panel,
    textvariable=faces_var,
    font=("Arial", 16),
    fg="white",
    bg="#1a1a1a"
)

faces_label.pack(pady=10)

# =========================================================
# BUTTONS
# =========================================================

running = True


def toggle_camera():
    global running

    running = not running

# =========================================================

button_frame = tk.Frame(side_panel, bg="#1a1a1a")
button_frame.pack(pady=30)

start_btn = tk.Button(
    button_frame,
    text="START / STOP",
    font=("Arial", 14, "bold"),
    bg="green",
    fg="white",
    width=18,
    command=toggle_camera
)

start_btn.pack(pady=10)

# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# =========================================================
# FPS
# =========================================================

prev_time = cv2.getTickCount()

face_id = 0
last_save_time = 0

# =========================================================
# COLORS
# =========================================================

color_map = {

    "Happy": (0,255,0),

    "Sad": (255,0,0),

    "Angry": (0,0,255),

    "Fear": (255,255,0),

    "Neutral": (255,255,255),

    "Surprise": (0,255,255),

    "Disgust": (255,0,255),

    "Detecting...": (120,120,120)
}

# =========================================================
# UPDATE FRAME
# =========================================================


def update_frame():

    global prev_time
    global face_id
    global last_save_time

    if running:

        ret, frame = cap.read()
        alpha = 1.2
        beta = 20

        frame = cv2.convertScaleAbs(
            frame,
            alpha=alpha,
            beta=beta
        )

        if ret:

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = face_detector.process(rgb_frame)

            detected_faces = 0

            if results.detections:

                detected_faces = len(results.detections)

                for detection in results.detections:

                    bbox = detection.location_data.relative_bounding_box

                    h, w, _ = frame.shape

                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)

                    width = int(bbox.width * w)
                    height = int(bbox.height * h)

                    padding = 25

                    x = max(0, x - padding)
                    y = max(0, y - padding)

                    width += padding * 2
                    height += padding * 2

                    face = frame[
                        y:y+height,
                        x:x+width
                    ]
                    if width < 120:

                        scale = 2

                        face = cv2.resize(
                        face,
                        None,
                        fx=scale,
                        fy=scale
                        )

                    if face.size == 0:
                        continue

                    # =====================================================
                    # IMAGE PROCESSING
                    # =====================================================

                    gray_face = cv2.cvtColor(
                        face,
                        cv2.COLOR_BGR2GRAY
                    )

                    gray_face = cv2.GaussianBlur(
                        gray_face,
                        (3,3),
                        0
                    )

                    gray_face = cv2.equalizeHist(
                        gray_face
                    )
                    clahe = cv2.createCLAHE(
                        clipLimit=2.0,
                        tileGridSize=(8,8)
                    )

                    gray_face = clahe.apply(gray_face)

                    face = cv2.cvtColor(
                        gray_face,
                        cv2.COLOR_GRAY2BGR
                    )

                    # =====================================================
                    # PREPROCESS
                    # =====================================================

                    face_input = transform(face)

                    face_input = face_input.unsqueeze(0)

                    face_input = face_input.to(device)
                    
                    face = cv2.fastNlMeansDenoisingColored(
                     face,
                         None,
                         10,
                        10,
                        7,
                        21
                    )

                    # =====================================================
                    # INFERENCE
                    # =====================================================

                    with torch.no_grad():

                        with torch.cuda.amp.autocast():

                            output = model(face_input)

                            probs = torch.softmax(
                                output,
                                dim=1
                            )

                            prediction = torch.argmax(
                                probs,
                                dim=1
                            ).item()

                    # =====================================================
                    # TEMPORAL SMOOTHING
                    # =====================================================

                    emotion_history.append(prediction)

                    smooth_prediction = max(

                        set(emotion_history),
                        key=emotion_history.count
                    )

                    emotion = emotion_labels[
                        smooth_prediction
                    ]

                    confidence = probs[
                        0,
                        smooth_prediction
                    ].item()

                    if confidence < 0.45:
                        emotion = "Detecting..."

                    # =====================================================
                    # SAVE FACE
                    # =====================================================

                    current_time_save = time.time()

                    if (
                        confidence > 0.60 and
                        current_time_save - last_save_time >= 10
                    ):

                        timestamp = datetime.now().strftime(
                            "%Y%m%d_%H%M%S"
                        )

                        filename = (
                            f"face_{face_id}_{emotion}_{timestamp}.jpg"
                        )

                        filepath = os.path.join(
                            SAVE_DIR,
                            filename
                        )

                        cv2.imwrite(filepath, face)

                        last_save_time = current_time_save

                        face_id += 1

                    # =====================================================
                    # COLORS
                    # =====================================================

                    box_color = color_map.get(
                        emotion,
                        (0,255,0)
                    )

                    # =====================================================
                    # DRAW
                    # =====================================================

                    cv2.rectangle(

                        frame,

                        (x, y),

                        (x + width, y + height),

                        box_color,

                        2
                    )

                    cv2.putText(

                        frame,

                        f"{emotion} {confidence:.2f}",

                        (x, y - 10),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.8,

                        box_color,

                        2
                    )

                    emotion_var.set(
                        f"Emotion: {emotion}"
                    )

                    confidence_var.set(
                        f"Confidence: {confidence:.2f}"
                    )

            faces_var.set(
                f"Detected Faces: {detected_faces}"
            )

            # =========================================================
            # FPS
            # =========================================================

            current_time = cv2.getTickCount()

            fps = cv2.getTickFrequency() / (
                current_time - prev_time
            )

            prev_time = current_time

            fps_var.set(
                f"FPS: {int(fps)}"
            )

            cv2.putText(

                frame,

                f"FPS: {int(fps)}",

                (50, 50),

                cv2.FONT_HERSHEY_SIMPLEX,

                1,

                (0,255,255),

                2
            )

            # =========================================================
            # SHOW IN GUI
            # =========================================================

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            img = Image.fromarray(frame_rgb)

            img = img.resize((1000, 700))

            imgtk = ImageTk.PhotoImage(image=img)

            video_label.imgtk = imgtk

            video_label.configure(image=imgtk)

    root.after(1, update_frame)

# =========================================================
# START
# =========================================================

update_frame()

root.mainloop()

# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()

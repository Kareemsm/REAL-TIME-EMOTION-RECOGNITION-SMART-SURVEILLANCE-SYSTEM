# =========================================================
# REALTIME EMOTION RECOGNITION
# MobileNetV3Large + CBAM + MediaPipe
# PyTorch + OpenCV + Image Processing
# =========================================================

import cv2
import torch
import torch.nn as nn
import mediapipe as mp
import numpy as np
import os
import time

from datetime import datetime
from collections import deque
from torchvision import models, transforms

# =========================================================
# FOLDERS
# =========================================================

os.makedirs("enhanced_faces", exist_ok=True)
os.makedirs("logs", exist_ok=True)

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
# LOGGER
# =========================================================

def log_event(message):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(
        "logs/events.txt",
        "a"
    ) as f:

        f.write(
            f"[{timestamp}] {message}\n"
        )

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

        avg_out = self.fc(
            self.avg_pool(x)
        )

        max_out = self.fc(
            self.max_pool(x)
        )

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

emotion_history = deque(maxlen=15)

# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# =========================================================
# FPS
# =========================================================

prev_time = cv2.getTickCount()

# =========================================================
# SAVE TIMER
# =========================================================

last_save_time = 0

# =========================================================
# LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # =====================================================
    # BRIGHTNESS CHECK
    # =====================================================

    gray_scene = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    brightness = np.mean(gray_scene)

    night_mode = brightness < 45

    # =====================================================
    # LIGHT ENHANCEMENT
    # =====================================================

    frame = cv2.convertScaleAbs(
        frame,
        alpha=1.1,
        beta=10
    )

    # =====================================================
    # RGB
    # =====================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_detector.process(rgb_frame)

    if results.detections:

        for detection in results.detections:

            bbox = detection.location_data.relative_bounding_box

            h, w, _ = frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)

            width = int(bbox.width * w)
            height = int(bbox.height * h)

            padding = 40

            x = max(0, x - padding)
            y = max(0, y - padding)

            width = min(
                frame.shape[1] - x,
                width + padding * 2
            )

            height = min(
                frame.shape[0] - y,
                height + padding * 2
            )

            face = frame[
                y:y+height,
                x:x+width
            ]

            if face.size == 0:
                continue

            # =====================================================
            # FACE RESIZE
            # =====================================================

            face = cv2.resize(
                face,
                (224, 224)
            )

                    # =====================================================
            # IMAGE PROCESSING (BETTER VERSION)
            # =====================================================

            # resize
            face_resized = cv2.resize(face, (224, 224))

            # grayscale
            gray_face = cv2.cvtColor(
                face_resized,
                cv2.COLOR_BGR2GRAY
            )


            # =====================================================
            # DENOISE
            # =====================================================

            denoise = cv2.fastNlMeansDenoising(
                gray_face,
                None,
                5,
                7,
                21
            )

            # =====================================================
            # CLAHE CONTRAST
            # =====================================================

            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8,8)
            )

            enhanced = clahe.apply(denoise)

           

            # =====================================================
            # LIGHT SHARPEN ONLY
            # =====================================================

            blur = cv2.GaussianBlur(
                enhanced,
                (0,0),
                2
            )

            sharpened = cv2.addWeighted(
                enhanced,
                1.5,
                blur,
                -0.5,
                0
            )

            # =====================================================
            # FINAL SMOOTH
            # =====================================================

            final_face = cv2.bilateralFilter(
                sharpened,
                5,
                50,
                50
            )

   
            # =====================================================
            # MODEL INPUT
            # =====================================================

            model_face = cv2.cvtColor(
                final_face,
                cv2.COLOR_GRAY2BGR
            )

            face_input = transform(
                model_face
            )

            face_input = face_input.unsqueeze(0)

            face_input = face_input.to(device)

            # =====================================================
            # PREDICTION
            # =====================================================

            with torch.inference_mode():

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
            # SMOOTHING
            # =====================================================

            emotion_history.append(
                prediction
            )

            smooth_prediction = max(
                set(emotion_history),
                key=emotion_history.count
            )

            emotion = emotion_labels[
                smooth_prediction
            ]

            # =====================================================
            # SAVE EVERY 10 SECONDS
            # =====================================================

            current_time = time.time()

            if current_time - last_save_time > 10:

                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )

                filename = (
                    f"enhanced_faces/"
                    f"{emotion}_{timestamp}.jpg"
                )

                cv2.imwrite(
                    filename,
                    final_face
                )

                log_event(
                    f"{emotion} detected"
                )

                print(
                    f"SAVED -> {filename}"
                )

                last_save_time = current_time

            # =====================================================
            # DRAW
            # =====================================================

            cv2.rectangle(

                frame,

                (x, y),

                (x + width, y + height),

                (0,255,0),

                2
            )

            cv2.putText(

                frame,

                emotion,

                (x, y - 10),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.9,

                (0,255,0),

                2
            )

            # SHOW ENHANCED FACE

            cv2.imshow(
                "Enhanced Face",
                final_face
            )

    # =====================================================
    # FPS
    # =====================================================

    current_tick = cv2.getTickCount()

    fps = cv2.getTickFrequency() / (
        current_tick - prev_time
    )

    prev_time = current_tick

    cv2.putText(

        frame,

        f"FPS: {int(fps)}",

        (20,40),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (0,255,255),

        2
    )

    # =====================================================
    # NIGHT MODE
    # =====================================================

    if night_mode:

        cv2.putText(

            frame,

            "NIGHT MODE",

            (20,80),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0,0,255),

            3
        )

    # =====================================================
    # SHOW
    # =====================================================

    cv2.imshow(
        "Realtime Emotion Recognition",
        frame
    )

    # =====================================================
    # EXIT
    # =====================================================

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()
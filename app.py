import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
MODEL_SAVE_PATH = "best_model.pth"
IMG_SIZE        = 112
SEQUENCE_LENGTH = 10
NUM_CLASSES     = 6
DROPOUT         = 0.3

CLASS_NAMES = {
    0: "Real (Original)",
    1: "Deepfakes",
    2: "Face2Face",
    3: "FaceSwap",
    4: "NeuralTextures",
    5: "FaceShifter",
}

SUPPORTED_IMAGES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SUPPORTED_VIDEOS = (".mp4", ".avi", ".mov", ".mkv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# MODEL DEFINITION
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool    = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_pool, max_pool], dim=1)))


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx  = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg + mx).view(b, c, 1, 1)


class ImprovedCNN(nn.Module):
    def __init__(self, dropout=DROPOUT):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.MaxPool2d(2),
        )
        self.ca1 = ChannelAttention(64)
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ResidualBlock(128),
            nn.MaxPool2d(2),
        )
        self.ca2 = ChannelAttention(128)
        self.sa2 = SpatialAttention()
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            ResidualBlock(256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.stem(x)
        x = self.ca1(self.stage1(x))
        x = self.sa2(self.ca2(self.stage2(x)))
        x = self.stage3(x)
        x = self.dropout(x)
        return x.view(x.size(0), -1)


class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, lstm_hidden=256,
                 lstm_layers=2, dropout=DROPOUT):
        super().__init__()
        self.cnn = ImprovedCNN(dropout=dropout)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False,
        )
        self.layer_norm = nn.LayerNorm(lstm_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        x      = x.view(b * t, c, h, w)
        feats  = self.cnn(x).view(b, t, -1)
        out, _ = self.lstm(feats)
        out    = self.layer_norm(out)
        weights = torch.softmax(out.mean(dim=-1, keepdim=True), dim=1)
        context = (out * weights).sum(dim=1)
        return self.classifier(context)


# ============================================================
# PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def extract_face(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        h, w = frame.shape[:2]
        m    = min(h, w)
        y0, x0 = (h - m) // 2, (w - m) // 2
        face_crop = frame[y0:y0+m, x0:x0+m]
    else:
        x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        pad = int(0.2 * max(fw, fh))
        x   = max(0, x - pad);  y  = max(0, y - pad)
        fw  = min(frame.shape[1] - x, fw + 2*pad)
        fh  = min(frame.shape[0] - y, fh + 2*pad)
        face_crop = frame[y:y+fh, x:x+fw]
    return cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))


def frames_to_tensor(frames):
    imgs = []
    for f in frames:
        img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        imgs.append(transform(img))
    return torch.stack(imgs).unsqueeze(0)   # (1, T, C, H, W)


# ============================================================
# LOAD MODEL (once at startup)
# ============================================================

print(f"\nUsing device : {device}")
print(f"Loading model: {MODEL_SAVE_PATH}")

model = DeepfakeDetector().to(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH,
                                  map_location=device,
                                  weights_only=True))
model.eval()
print("Model loaded successfully!\n")


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)

from flask import render_template

@app.route("/")
def home():
    return render_template("index.html")

def run_prediction(tensor):
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs  = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    predicted = int(probs.argmax())
    return probs.tolist(), predicted


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file     = request.files["file"]
    filename = file.filename.lower()
    ext      = os.path.splitext(filename)[1]

    tmp_path = f"tmp_upload{ext}"
    file.save(tmp_path)

    try:
        # ── IMAGE ──
        if ext in SUPPORTED_IMAGES:
            frame  = cv2.imread(tmp_path)
            if frame is None:
                return jsonify({"error": "Could not read image"}), 400
            face   = extract_face(frame)
            frames = [face] * SEQUENCE_LENGTH
            tensor = frames_to_tensor(frames)

        # ── VIDEO ──
        elif ext in SUPPORTED_VIDEOS:
            cap   = cv2.VideoCapture(tmp_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total < SEQUENCE_LENGTH:
                cap.release()
                return jsonify({"error": "Video too short"}), 400

            indices = np.linspace(0, total - 1, SEQUENCE_LENGTH * 3, dtype=int)
            frames  = []
            for idx in indices:
                if len(frames) >= SEQUENCE_LENGTH:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frm = cap.read()
                if ret:
                    frames.append(extract_face(frm))
            cap.release()

            if len(frames) < SEQUENCE_LENGTH:
                return jsonify({"error": "Could not extract enough frames"}), 400
            tensor = frames_to_tensor(frames[:SEQUENCE_LENGTH])

        else:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400

        probs, predicted = run_prediction(tensor)
        is_real = (predicted == 0)

        print(f"  File: {file.filename}  →  {CLASS_NAMES[predicted]}  ({probs[predicted]*100:.1f}%)")

        return jsonify({
            "probs"     : probs,
            "predicted" : predicted,
            "isReal"    : is_real,
            "conf"      : probs[predicted],
            "class_name": CLASS_NAMES[predicted]
        })

    except Exception as e:
        print(f"  ERROR: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(device)})


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Starting Flask server on http://localhost:5000\n")
    app.run(debug=False, port=5000)
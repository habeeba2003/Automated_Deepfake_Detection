# 🎭 Automated Deepfake Detection Using Deep Learning

> A hybrid CNN–LSTM deep learning system for automated detection and classification of deepfake images and videos across multiple manipulation techniques.

**Author:** Habeeba M N — Reg. No. MAC24MCA-2029  
**Guide:** Prof. Sibu Skaria  
**Institution:** Mar Athanasius College of Engineering, Kothamangalam  
**Degree:** Master of Computer Applications — APJ Abdul Kalam Technological University  
**Year:** 2024–2026

---

## 🧠 Overview

Deepfake technology enables the creation of highly realistic manipulated images and videos using deep learning models. While it has legitimate applications in entertainment and education, its malicious misuse poses serious threats to digital security, identity integrity, and public trust.

This project proposes an **Automated Deepfake Detection System** that combines:
- **Convolutional Neural Networks (CNN)** — to extract spatial features from video frames (facial textures, blending artifacts, structural distortions)
- **Long Short-Term Memory (LSTM) networks** — to model temporal dependencies across frame sequences (motion inconsistencies, frame-to-frame anomalies)

The system performs **multi-class classification**, distinguishing between real videos and five deepfake manipulation categories, trained and evaluated on the FaceForensics++ dataset.

---

## ✨ Features

- ✅ Multi-class deepfake classification (6 categories: Real + 5 manipulation types)
- ✅ Supports both **image** and **video** inputs
- ✅ CNN–LSTM hybrid architecture trained from scratch (no ImageNet pretraining bias)
- ✅ Haar Cascade face detection for robust preprocessing
- ✅ Augmentation pipeline to improve generalization
- ✅ Mixed-precision training with `torch.amp` for efficiency
- ✅ Cosine annealing LR schedule with linear warmup
- ✅ Class-balanced sampling via `WeightedRandomSampler`
- ✅ Flask web app with drag-and-drop UI for real-time inference
- ✅ Comprehensive evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix

---

## 🏗️ Architecture

The model follows a **CNN → LSTM → Classifier** pipeline.

```
Input: B × 10 × 3 × 112 × 112  (batch × frames × channels × H × W)
         │
   ┌─────▼────────────────────────────────────────────┐
   │               CNN Feature Extractor              │
   │  Conv1 (32 filters) + BN + ReLU → MaxPool        │
   │  Conv2 (64 filters) + BN + ReLU → MaxPool        │
   │  Conv3 (128 filters) + BN + ReLU → MaxPool       │
   │  Conv4 (256 filters) + BN + ReLU                 │
   │  AvgPool (1×1) → 256-dim vector per frame        │
   └─────┬────────────────────────────────────────────┘
         │  Reshape: B × 10 × 256
   ┌─────▼────────────────────────────────────────────┐
   │            LSTM Temporal Modeler                 │
   │  Input: 256  |  Hidden: 256  |  Dropout: 0.25   │
   │  Output: Last hidden state → B × 256             │
   └─────┬────────────────────────────────────────────┘
         │
   ┌─────▼────────────────────────────────────────────┐
   │          Fully Connected + Softmax               │
   │  256 → 6 classes                                 │
   │  [Real, Deepfakes, Face2Face, FaceSwap,           │
   │   NeuralTextures, FaceShifter]                   │
   └──────────────────────────────────────────────────┘
```

### Key Hyperparameters

| Parameter | Value |
|---|---|
| Image Size | 112 × 112 |
| Sequence Length | 10 frames |
| Batch Size | 2 |
| Epochs | 90 |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Dropout | 0.25 |
| LR Schedule | Cosine Annealing + 5-epoch Linear Warmup |
| Loss Function | CrossEntropyLoss (label smoothing = 0.1) |
| Optimizer | AdamW |

---

## 📦 Dataset

**FaceForensics++ (C23 compression)** — sourced from [Kaggle](https://kaggle.com/datasets/xdxd003/ff-c23)

| Class | Videos |
|---|---|
| Original (Real) | 1,000 |
| Deepfakes | 1,000 |
| Face2Face | 1,000 |
| FaceSwap | 1,000 |
| NeuralTextures | 1,000 |
| FaceShifter | 1,000 |
| **Total** | **6,000** |

- Format: MP4 (C23 compression)
- Task: Multi-class classification
- Split: 70% Train / 15% Validation / 15% Test (stratified)

---

## 🔄 Project Pipeline

```
Video Dataset Input
        │
        ▼
Frame Extraction & Face Detection (Haar Cascade)
        │  • Sample frames at uniform intervals
        │  • Detect & crop largest face (+20% padding)
        │  • Fallback: center-square crop
        │  • Resize to 112×112 and save as JPEG
        ▼
CNN-Based Spatial Feature Extraction
        │  • Per-frame feature vectors (256-dim)
        ▼
LSTM-Based Temporal Modeling
        │  • Sequence of 10 frame features
        │  • Captures motion inconsistencies
        ▼
Model Generation (CNN-LSTM)
        │
        ├── Image Input → CNN only
        └── Video Input → CNN + LSTM
                │
                ▼
        Output + Confidence Score
        (Real / Manipulation Type + Probability)
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/automated-deepfake-detection.git
cd automated-deepfake-detection
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the FaceForensics++ C23 dataset from [Kaggle](https://kaggle.com/datasets/xdxd003/ff-c23) and place it in the project root:

```
FaceForensics++_C23/
├── original/         # 1000 real videos
├── Deepfakes/
├── Face2Face/
├── FaceSwap/
├── NeuralTextures/
└── FaceShifter/
```

---

## 🚀 Usage

### Train the model

```bash
python train.py
```

This will:
- Preprocess and extract face frames from all videos
- Build and train the CNN+LSTM model
- Save the best checkpoint to `best_model.pth`
- Generate training curves and logs

### Evaluate on test set

```bash
python evaluate.py
```

Outputs: accuracy, classification report, confusion matrix, ROC curves, per-class accuracy.

### Run inference on a single file

```bash
# For an image
python inference.py --input path/to/image.jpg

# For a video
python inference.py --input path/to/video.mp4
```

### Launch the web application

```bash
python app.py
```

Open your browser at `http://localhost:5000`

---

## 📊 Results

### Final Model (Stage 6 — 90 Epochs)

| Metric | Value |
|---|---|
| Test Accuracy | **83.89%** |
| Test Loss | 0.8968 |
| Macro F1-Score | 0.8430 |
| Weighted F1-Score | 0.8430 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Accuracy |
|---|---|---|---|---|
| Real (Original) | 0.6581 | 0.6800 | 0.6689 | 68.0% |
| Deepfakes | 0.9773 | 0.8600 | 0.9149 | 86.0% |
| Face2Face | 0.9444 | 0.9067 | 0.9252 | 90.7% |
| FaceSwap | 0.9653 | 0.9267 | 0.9456 | 92.7% |
| NeuralTextures | 0.6207 | 0.7200 | 0.6667 | 72.0% |
| FaceShifter | 0.9338 | 0.9400 | 0.9369 | **94.0%** |

### ROC-AUC Scores

| Class | AUC |
|---|---|
| FaceSwap | 0.995 |
| Face2Face | 0.990 |
| FaceShifter | 0.988 |
| Deepfakes | 0.978 |
| NeuralTextures | 0.920 |
| Real (Original) | 0.877 |

> 4 out of 6 classes achieve above 80% per-class accuracy. The primary challenge is distinguishing NeuralTextures from Real due to subtle texture-level modifications.

### Training Progression

| Stage | Epochs | Test Accuracy |
|---|---|---|
| 1 | 10 | 42.00% |
| 2 | 20 | 62.27% |
| 3 | 50 | 77.17% |
| 4 | 60 | 76.33% |
| 5 | 85 | 81.78% |
| **6 (Final)** | **90** | **83.89%** |

---

## 🌐 Deployment

The system is deployed as a **Flask web application** with a dark-themed, interactive frontend.

**Tech Stack:**
- Backend: Python + Flask + PyTorch + OpenCV
- Frontend: HTML5 + CSS3 + JavaScript (drag-and-drop interface)
- Communication: RESTful API (JSON responses)

**Workflow:**
1. User uploads image or video via the web UI
2. Flask backend performs face detection and frame extraction
3. Preprocessed input is passed through the CNN+LSTM model
4. Results returned as JSON (predicted class, confidence scores, manipulation type)
5. Frontend displays results with confidence bars and color-coded verdict

**Run locally:**
```bash
python app.py
# Visit: http://localhost:5000
```

---

## 📁 Project Structure

```
automated-deepfake-detection/
│
├── FaceForensics++_C23/          # Dataset (not included in repo)
│   ├── original/
│   ├── Deepfakes/
│   ├── Face2Face/
│   ├── FaceSwap/
│   ├── NeuralTextures/
│   └── FaceShifter/
│
├── processed_sequences/          # Extracted face frames (auto-generated)
│
├── train.py                      # Training script (preprocessing + model training)
├── evaluate.py                   # Test set evaluation and metric generation
├── inference.py                  # Single-file inference script
├── app.py                        # Flask web application
│
├── models/
│   └── deepfake_detector.py      # CNN+LSTM model definition
│
├── utils/
│   ├── preprocessing.py          # Frame extraction and face detection
│   └── dataset.py                # VideoDataset class
│
├── templates/
│   ├── index.html                # Intro page
│   ├── upload.html               # Upload page
│   └── results.html              # Results page
│
├── static/                       # CSS, JS, assets
│
├── best_model.pth                # Saved model checkpoint (auto-generated)
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.7.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
flask>=2.3.0
flask-cors>=4.0.0
tqdm>=4.65.0
pandas>=2.0.0
Pillow>=9.5.0
```

Install with:
```bash
pip install -r requirements.txt
```

## 🔭 Future Work

- [ ] **Attention Mechanisms** — Integrate visual attention (Bi-LSTM + attention) to focus on manipulated facial regions (eyes, lips, textures)
- [ ] **Transformer Architecture** — Explore hybrid CNN–LSTM–Transformer models for long-range temporal dependency modeling
- [ ] **Real-Time Detection** — Extend to live webcam/video stream inference
- [ ] **Multimodal Analysis** — Combine visual + audio features to detect lip-sync errors and voice mismatches
- [ ] **Mobile Deployment** — Model pruning and quantization for edge/mobile deployment
- [ ] **Explainability** — Grad-CAM visualizations to highlight manipulated regions
- [ ] **Diverse Datasets** — Train on more varied ethnicities, compression levels, and newer GAN-based deepfakes

---

## 🏫 Academic Information

| Field | Details |
|---|---|
| Institution | Mar Athanasius College of Engineering, Kothamangalam |
| Department | Computer Applications |
| Degree | Master of Computer Applications (MCA) |
| University | APJ Abdul Kalam Technological University, Thiruvananthapuram |

---

*This project was developed as a Main Project for the MCA programme at Mar Athanasius College of Engineering, Kothamangalam.*

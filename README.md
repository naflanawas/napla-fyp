# MURMUR - AI-Powered Breath-Based AAC System

MURMUR is an Augmentative and Alternative Communication (AAC) system designed for individuals with severe motor impairments. It translates breathing patterns into spoken phrases using deep learning.

## 🌟 Features

- **Personalized Recognition**: Few-shot learning adapts to each user's unique breath signature
- **No Retraining Required**: Uses prototypical networks for instant personalization
- **Confidence Scoring**: Clear indicators when predictions are uncertain
- **Real-Time Processing**: Continuous listening with immediate feedback
- **Cross-Platform**: Flutter app works on iOS and Android

## 🏗️ Architecture

```
Mobile App (Flutter) → WAV Audio → API Server (FastAPI) → MSTCN Model → ProtoNet → Intent → TTS
```

### Components

1. **Backend Server** (`/backend_server`)
   - FastAPI REST API
   - MSTCN deep learning model for embedding extraction
   - Prototypical Network for few-shot classification
   - User data persistence

2. **Mobile App** (`/mobile_app`)
   - Flutter/Dart application
   - Real-time audio recording
   - Breath visualization
   - Text-to-speech output

## 🚀 Quick Start

### Backend Setup

```bash
# Navigate to project
cd napla-fyp

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend_server/requirements.txt

# Run server
cd backend_server
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server will be available at `http://localhost:8000`

### Mobile App Setup

```bash
cd mobile_app

# Get dependencies
flutter pub get

# Run on connected device
flutter run
```

## 📡 API Endpoints

### Health Check
```
GET /
GET /health
```

### Calibration
```
POST /calibrate/{user_id}/{intent}
- Upload audio file to create/update intent prototype
- Optional: phrase parameter for TTS output
```

### Prediction
```
POST /predict/{user_id}
- Upload audio file to get intent prediction
- Returns: intent, confidence, phrase
```

### User Management
```
GET  /user/{user_id}/intents     - List all intents
GET  /user/{user_id}/stats       - User statistics
DELETE /user/{user_id}/intent/{intent} - Delete intent
DELETE /user/{user_id}           - Delete user
GET  /users                      - List all users
```

## 📋 Usage Guide

### Step 1: Calibration

1. Open the app and select "Add Command"
2. Choose an intent name (e.g., "water", "help")
3. Set the phrase to speak (e.g., "I need water")
4. Record 3-5 breath samples
5. System creates personalized prototype

### Step 2: Communication

1. Enable "Listening Mode"
2. Produce breath pattern
3. System detects and classifies breath
4. Matched phrase is spoken via TTS

## ⚙️ Technical Specifications

| Parameter | Value |
|-----------|-------|
| Sample Rate | 16,000 Hz |
| Window Size | 1024 frames (~64ms) |
| Audio Format | PCM WAV (Mono) |
| Embedding Dimension | 64 |
| Model | Multi-Scale TCN |

## 📁 Project Structure

```
napla-fyp/
├── backend_server/
│   ├── main.py           # FastAPI application
│   ├── model.py          # MSTCN architecture
│   ├── audio_processor.py # Audio preprocessing
│   ├── protonet.py       # Prototypical network
│   ├── user_manager.py   # User data persistence
│   ├── config.py         # Configuration
│   ├── requirements.txt  # Python dependencies
│   ├── weights/          # Model weights
│   └── user_data/        # User prototypes (gitignored)
├── mobile_app/
│   ├── lib/
│   │   ├── main.dart
│   │   ├── services/
│   │   ├── screens/
│   │   ├── widgets/
│   │   └── models/
│   └── pubspec.yaml
├── .gitignore
└── README.md
```

## 🔧 Configuration

Edit `backend_server/config.py` to customize:

- Audio parameters (sample rate, window size)
- Model paths
- Confidence thresholds
- Server settings

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please read contributing guidelines first.

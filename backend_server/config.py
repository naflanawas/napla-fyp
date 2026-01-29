"""
Configuration constants for MURMUR system
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
WEIGHTS_DIR = BASE_DIR / "weights"
USER_DATA_DIR = BASE_DIR / "user_data"
MODEL_PATH = WEIGHTS_DIR / "murmur_global.pth"

# Audio Processing
SAMPLE_RATE = 16000  # Hz
WINDOW_SIZE = 1024   # frames (approx 64ms at 16kHz)
N_MELS = 64          # Number of mel bands for spectrogram
HOP_LENGTH = 512     # Hop length for STFT
N_FFT = 1024         # FFT window size

# Model Architecture (from provided MSTCN architecture)
INPUT_CHANNELS = 3   # Input channels for Conv2d
EMBEDDING_DIM = 64   # Final embedding dimension
NUM_BRANCHES = 4     # Number of TCN branches
BRANCH_CHANNELS = 64 # Channels in each branch

# Prototypical Network
DISTANCE_METRIC = "euclidean"  # or "cosine"
CONFIDENCE_THRESHOLD = 0.7     # Minimum confidence for prediction
MIN_SAMPLES_PER_INTENT = 2     # Minimum calibration samples

# Server
HOST = "0.0.0.0"
PORT = 8000
CORS_ORIGINS = ["*"]  # Allow all origins for development

# Ensure directories exist
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

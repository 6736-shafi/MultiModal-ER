"""Shared constants and hyperparameters for all pipelines."""

import os
import torch

# Reproducibility seed
SEED = 42

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
DATASET_PATH = os.environ.get("TESS_DATASET_PATH", "../../TESS Toronto emotional speech set data")

# Emotion labels
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]
EMOTION_MAP = {e: i for i, e in enumerate(EMOTIONS)}
EMOTION_LABELS = {
    "angry": "Angry", "disgust": "Disgust", "fear": "Fear",
    "happy": "Happy", "neutral": "Neutral", "ps": "Pleasant Surprise", "sad": "Sad"
}

# Speech hyperparameters
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_AUDIO_LEN = 200  # Max number of time frames (pad/truncate)

# Text hyperparameters
MAX_SEQ_LEN = 20
EMBEDDING_DIM = 64

# Shared model hyperparameters
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3

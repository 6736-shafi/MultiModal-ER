"""Shared utilities for all emotion recognition pipelines."""

from .constants import (
    EMOTIONS, EMOTION_MAP, EMOTION_LABELS, DEVICE, SEED,
    SAMPLE_RATE, N_MFCC, MAX_AUDIO_LEN, MAX_SEQ_LEN,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, HIDDEN_SIZE,
    NUM_LAYERS, EMBEDDING_DIM, DROPOUT, DATASET_PATH
)
from .seed import set_seed
from .vocabulary import Vocabulary
from .data_loading import load_tess_data
from .visualization import (
    visualize_tsne, plot_training_history, plot_confusion_matrix,
    extract_features_for_visualization, analyze_low_confidence_correct
)
from .transcribe import transcribe_dataset, get_transcripts_for_files

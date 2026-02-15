"""TESS dataset loading utilities."""

import os
from .constants import EMOTION_MAP


def load_tess_data(dataset_path):
    """Load TESS dataset file paths, labels, and words.

    Returns:
        file_paths: list of .wav file paths
        labels: list of integer emotion labels
        words: list of spoken words extracted from filenames
    """
    file_paths = []
    labels = []
    words = []

    for root, dirs, files in os.walk(dataset_path):
        for file in sorted(files):
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                parts = file.replace(".wav", "").split("_")
                emotion = parts[-1].lower()
                if emotion in EMOTION_MAP:
                    file_paths.append(file_path)
                    labels.append(EMOTION_MAP[emotion])
                    word = "_".join(parts[1:-1]).lower()
                    words.append(word)

    return file_paths, labels, words

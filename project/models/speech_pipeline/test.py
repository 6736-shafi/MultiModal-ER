"""
Speech Emotion Recognition - Testing Pipeline
Evaluates the trained speech model on the test set.
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Add parent directory for common imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common import (
    DEVICE, EMOTIONS, EMOTION_LABELS, SEED, BATCH_SIZE, DATASET_PATH,
    set_seed, load_tess_data, visualize_tsne, plot_confusion_matrix,
    extract_features_for_visualization, analyze_low_confidence_correct
)

from train import SpeechEmotionModel, TESSAudioDataset


def evaluate_model(model, test_loader):
    """Evaluate model on test set and return predictions."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(DEVICE)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def analyze_errors(file_paths, y_true, y_pred, y_probs, n_errors=5):
    """Analyze misclassified samples."""
    errors = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            errors.append({
                "file": file_paths[i] if i < len(file_paths) else f"sample_{i}",
                "true_emotion": EMOTION_LABELS[EMOTIONS[y_true[i]]],
                "predicted_emotion": EMOTION_LABELS[EMOTIONS[y_pred[i]]],
                "confidence": float(y_probs[i][y_pred[i]])
            })

    errors.sort(key=lambda x: x["confidence"], reverse=True)
    print(f"\nTop {min(n_errors, len(errors))} Confident Misclassifications:")
    print("-" * 70)
    for err in errors[:n_errors]:
        print(f"  File: {os.path.basename(err['file'])}")
        print(f"  True: {err['true_emotion']} → Predicted: {err['predicted_emotion']} "
              f"(conf: {err['confidence']:.3f})")
        print()

    return errors[:n_errors]


def save_results(y_true, y_pred, accuracy, report, errors, low_conf, save_dir="../../Results"):
    """Save evaluation results to files."""
    os.makedirs(save_dir, exist_ok=True)

    results = {
        "model": "Speech (BiLSTM)",
        "accuracy": float(accuracy),
        "per_class_metrics": {},
        "error_analysis": errors,
        "low_confidence_correct": low_conf
    }

    emotion_names = [EMOTION_LABELS[e] for e in EMOTIONS]
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    for i, name in enumerate(emotion_names):
        tp = cm[i, i]
        total = cm[i].sum()
        results["per_class_metrics"][name] = {
            "accuracy": float(tp / total) if total > 0 else 0.0,
            "support": int(total)
        }

    with open(os.path.join(save_dir, "speech_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(save_dir, "speech_classification_report.txt"), "w") as f:
        f.write("Speech Emotion Recognition - Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write(report)

    print(f"\nResults saved to {save_dir}/")


if __name__ == "__main__":
    set_seed(SEED)

    print("=" * 60)
    print("Speech Emotion Recognition - Testing")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Load data and create test set (same split as training)
    file_paths, labels, words = load_tess_data(DATASET_PATH)
    _, X_temp, _, y_temp = train_test_split(
        file_paths, labels, test_size=0.3, random_state=SEED, stratify=labels
    )
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )

    test_dataset = TESSAudioDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Test samples: {len(X_test)}")

    # Load trained model
    model = SpeechEmotionModel().to(DEVICE)
    model.load_state_dict(torch.load("speech_model_best.pth", map_location=DEVICE, weights_only=True))
    print("Loaded speech_model_best.pth")

    # Evaluate
    y_pred, y_true, y_probs = evaluate_model(model, test_loader)

    accuracy = accuracy_score(y_true, y_pred)
    emotion_names = [EMOTION_LABELS[e] for e in EMOTIONS]
    report = classification_report(y_true, y_pred, target_names=emotion_names)

    print(f"\nOverall Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, "Speech Model - Confusion Matrix",
                          "../../Results/plots/speech_confusion_matrix.png", cmap='Blues')

    # Error analysis
    errors = analyze_errors(X_test, y_true, y_pred, y_probs)

    # Low-confidence correct predictions
    low_conf = analyze_low_confidence_correct(y_true, y_pred, y_probs, X_test)

    # t-SNE visualization of temporal features
    features, feat_labels = extract_features_for_visualization(model, test_loader, DEVICE, modality="speech")
    visualize_tsne(
        features, feat_labels,
        "Speech Temporal Modelling - Emotion Clusters (t-SNE)",
        "../../Results/plots/speech_temporal_tsne.png"
    )

    # Save results
    save_results(y_true, y_pred, accuracy, report, errors, low_conf)

    print("\nTesting complete!")

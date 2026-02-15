"""
Text Emotion Recognition - Testing Pipeline
Evaluates the trained text model on the test set.
"""

import os
import sys
import json
import pickle
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
    extract_features_for_visualization, analyze_low_confidence_correct,
    get_transcripts_for_files
)

from train import TextEmotionModel, TESSTextDataset


def evaluate_model(model, test_loader):
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for token_ids, labels in test_loader:
            token_ids = token_ids.to(DEVICE)
            outputs = model(token_ids)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def analyze_errors(texts, y_true, y_pred, y_probs, n_errors=5):
    """Analyze misclassified samples."""
    errors = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            errors.append({
                "text": texts[i] if i < len(texts) else f"sample_{i}",
                "true_emotion": EMOTION_LABELS[EMOTIONS[y_true[i]]],
                "predicted_emotion": EMOTION_LABELS[EMOTIONS[y_pred[i]]],
                "confidence": float(y_probs[i][y_pred[i]])
            })

    errors.sort(key=lambda x: x["confidence"], reverse=True)
    print(f"\nTop {min(n_errors, len(errors))} Confident Misclassifications:")
    print("-" * 70)
    for err in errors[:n_errors]:
        print(f"  Text: \"{err['text']}\"")
        print(f"  True: {err['true_emotion']} → Predicted: {err['predicted_emotion']} "
              f"(conf: {err['confidence']:.3f})")
        print()

    return errors[:n_errors]


def save_results(y_true, y_pred, accuracy, report, errors, low_conf, save_dir="../../Results"):
    """Save evaluation results."""
    os.makedirs(save_dir, exist_ok=True)

    results = {
        "model": "Text (BiLSTM)",
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

    with open(os.path.join(save_dir, "text_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(save_dir, "text_classification_report.txt"), "w") as f:
        f.write("Text Emotion Recognition - Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write(report)

    print(f"\nResults saved to {save_dir}/")


if __name__ == "__main__":
    set_seed(SEED)

    print("=" * 60)
    print("Text Emotion Recognition - Testing")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Load vocabulary
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    # Load data and get Whisper transcripts
    file_paths, labels, words = load_tess_data(DATASET_PATH)
    texts = get_transcripts_for_files(file_paths)

    # Index-based splitting (consistent with train.py and fusion pipeline)
    indices = list(range(len(file_paths)))
    _, temp_idx = train_test_split(
        indices, test_size=0.3, random_state=SEED, stratify=labels
    )
    _, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=SEED,
        stratify=[labels[i] for i in temp_idx]
    )

    test_texts = [texts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    test_dataset = TESSTextDataset(test_texts, test_labels, vocab)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_texts)}")

    # Load model
    checkpoint = torch.load("text_model_best.pth", map_location=DEVICE, weights_only=True)
    model = TextEmotionModel(vocab_size=len(vocab)).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded text_model_best.pth")

    # Evaluate
    y_pred, y_true, y_probs = evaluate_model(model, test_loader)

    accuracy = accuracy_score(y_true, y_pred)
    emotion_names = [EMOTION_LABELS[e] for e in EMOTIONS]
    report = classification_report(y_true, y_pred, target_names=emotion_names)

    print(f"\nOverall Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, "Text Model - Confusion Matrix",
                          "../../Results/plots/text_confusion_matrix.png", cmap='Greens')

    # Error analysis
    errors = analyze_errors(test_texts, y_true, y_pred, y_probs)

    # Low-confidence correct predictions
    low_conf = analyze_low_confidence_correct(y_true, y_pred, y_probs, test_texts)

    # t-SNE visualization
    features, feat_labels = extract_features_for_visualization(model, test_loader, DEVICE, modality="text")
    visualize_tsne(
        features, feat_labels,
        "Text Contextual Modelling - Emotion Clusters (t-SNE)",
        "../../Results/plots/text_contextual_tsne.png"
    )

    # Save results
    save_results(y_true, y_pred, accuracy, report, errors, low_conf)

    print("\nTesting complete!")

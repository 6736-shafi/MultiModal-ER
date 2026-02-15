"""
Multimodal Emotion Recognition - Fusion Testing Pipeline
Evaluates the trained fusion model and generates comparative results.
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Add parent directory for common imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common import (
    DEVICE, EMOTIONS, EMOTION_LABELS, EMOTION_MAP, SEED, BATCH_SIZE, DATASET_PATH,
    set_seed, load_tess_data, visualize_tsne, plot_confusion_matrix,
    extract_features_for_visualization, analyze_low_confidence_correct,
    get_transcripts_for_files
)

from train import MultimodalFusionModel, TESSMultimodalDataset


def evaluate_model(model, test_loader):
    """Evaluate fusion model on test set."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for audio_features, token_ids, labels in test_loader:
            audio_features = audio_features.to(DEVICE)
            token_ids = token_ids.to(DEVICE)
            outputs = model(audio_features, token_ids)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def analyze_errors(file_paths, texts, y_true, y_pred, y_probs, n_errors=5):
    """Analyze misclassified samples."""
    errors = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            errors.append({
                "file": os.path.basename(file_paths[i]) if i < len(file_paths) else f"sample_{i}",
                "text": texts[i] if i < len(texts) else "",
                "true_emotion": EMOTION_LABELS[EMOTIONS[y_true[i]]],
                "predicted_emotion": EMOTION_LABELS[EMOTIONS[y_pred[i]]],
                "confidence": float(y_probs[i][y_pred[i]])
            })

    errors.sort(key=lambda x: x["confidence"], reverse=True)
    print(f"\nTop {min(n_errors, len(errors))} Confident Misclassifications:")
    print("-" * 70)
    for err in errors[:n_errors]:
        print(f"  File: {err['file']}, Text: \"{err['text']}\"")
        print(f"  True: {err['true_emotion']} → Predicted: {err['predicted_emotion']} "
              f"(conf: {err['confidence']:.3f})")
        print()

    return errors[:n_errors]


def save_results(y_true, y_pred, accuracy, report, errors, low_conf, save_dir="../../Results"):
    """Save evaluation results."""
    os.makedirs(save_dir, exist_ok=True)

    results = {
        "model": "Fusion (Speech BiLSTM + Text BiLSTM)",
        "accuracy": float(accuracy),
        "per_class_metrics": {},
        "error_analysis": errors,
        "low_confidence_correct": low_conf
    }

    emotion_names = [EMOTION_LABELS[e] for e in EMOTIONS]
    cm = confusion_matrix(y_true, y_pred)
    for i, name in enumerate(emotion_names):
        tp = cm[i, i]
        total = cm[i].sum()
        results["per_class_metrics"][name] = {
            "accuracy": float(tp / total) if total > 0 else 0.0,
            "support": int(total)
        }

    with open(os.path.join(save_dir, "fusion_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(save_dir, "fusion_classification_report.txt"), "w") as f:
        f.write("Multimodal Fusion Emotion Recognition - Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write(report)

    print(f"\nResults saved to {save_dir}/")


def generate_comparison_table(save_dir="../../Results"):
    """Generate comparison table across all 3 model variants."""
    models = ["speech", "text", "fusion"]
    model_names = ["Speech (BiLSTM)", "Text (BiLSTM)", "Fusion (Multimodal)"]

    print("\n" + "=" * 80)
    print("COMPARISON: All 3 Model Variants")
    print("=" * 80)

    comparison = {}
    for model_key, model_name in zip(models, model_names):
        result_file = os.path.join(save_dir, f"{model_key}_results.json")
        if os.path.exists(result_file):
            with open(result_file) as f:
                data = json.load(f)
            comparison[model_name] = data

    if not comparison:
        print("No results found. Run all three pipelines first.")
        return

    # Print accuracy comparison
    print(f"\n{'Model':<30} {'Overall Accuracy':>20}")
    print("-" * 52)
    for name, data in comparison.items():
        print(f"{name:<30} {data['accuracy']:>20.4f}")

    # Per-class comparison
    emotion_names = [EMOTION_LABELS[e] for e in EMOTIONS]
    print(f"\n{'Emotion':<20}", end="")
    for name in comparison:
        print(f" {name:>20}", end="")
    print()
    print("-" * (20 + 20 * len(comparison) + len(comparison)))

    for emotion in emotion_names:
        print(f"{emotion:<20}", end="")
        for name, data in comparison.items():
            if emotion in data.get("per_class_metrics", {}):
                acc = data["per_class_metrics"][emotion]["accuracy"]
                print(f" {acc:>20.4f}", end="")
            else:
                print(f" {'N/A':>20}", end="")
        print()

    # Save comparison table
    comparison_file = os.path.join(save_dir, "comparison_results.json")
    with open(comparison_file, "w") as f:
        json.dump(comparison, f, indent=2)

    # Plot comparison bar chart
    if len(comparison) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        # Overall accuracy comparison
        names = list(comparison.keys())
        accs = [comparison[n]["accuracy"] for n in names]
        colors = ['#3498db', '#2ecc71', '#9b59b6']
        ax1.bar(names, accs, color=colors[:len(names)])
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Overall Accuracy Comparison")
        ax1.set_ylim(0, 1.0)
        for i, acc in enumerate(accs):
            ax1.text(i, acc + 0.01, f"{acc:.4f}", ha='center', fontweight='bold')

        # Per-class comparison
        x = np.arange(len(emotion_names))
        width = 0.25
        for i, (name, data) in enumerate(comparison.items()):
            class_accs = []
            for emotion in emotion_names:
                if emotion in data.get("per_class_metrics", {}):
                    class_accs.append(data["per_class_metrics"][emotion]["accuracy"])
                else:
                    class_accs.append(0)
            ax2.bar(x + i * width, class_accs, width, label=name, color=colors[i])

        ax2.set_xlabel("Emotion")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Per-Class Accuracy Comparison")
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(emotion_names, rotation=45, ha='right')
        ax2.legend()
        ax2.set_ylim(0, 1.0)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "plots", "model_comparison.png"), dpi=150)
        plt.close()
        print(f"\nComparison plot saved to {save_dir}/plots/model_comparison.png")


if __name__ == "__main__":
    set_seed(SEED)

    print("=" * 60)
    print("Multimodal Emotion Recognition - Fusion Testing")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Load vocabulary
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    # Load data and get Whisper transcripts
    file_paths, labels, words = load_tess_data(DATASET_PATH)
    texts = get_transcripts_for_files(file_paths)

    # Index-based splitting (consistent with train.py)
    indices = list(range(len(file_paths)))
    _, temp_idx = train_test_split(
        indices, test_size=0.3, random_state=SEED, stratify=labels
    )
    _, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=SEED,
        stratify=[labels[i] for i in temp_idx]
    )

    test_files = [file_paths[i] for i in test_idx]
    test_texts = [texts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    test_dataset = TESSMultimodalDataset(test_files, test_texts, test_labels, vocab)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_files)}")

    # Load model
    checkpoint = torch.load("fusion_model_best.pth", map_location=DEVICE, weights_only=True)
    model = MultimodalFusionModel(vocab_size=len(vocab)).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded fusion_model_best.pth")

    # Evaluate
    y_pred, y_true, y_probs = evaluate_model(model, test_loader)

    accuracy = accuracy_score(y_true, y_pred)
    emotion_names = [EMOTION_LABELS[e] for e in EMOTIONS]
    report = classification_report(y_true, y_pred, target_names=emotion_names)

    print(f"\nOverall Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, "Fusion Model - Confusion Matrix",
                          "../../Results/plots/fusion_confusion_matrix.png", cmap='Purples')

    # Error analysis
    errors = analyze_errors(test_files, test_texts, y_true, y_pred, y_probs)

    # Low-confidence correct predictions
    low_conf = analyze_low_confidence_correct(y_true, y_pred, y_probs, test_files)

    # t-SNE visualizations
    fused_feat, speech_feat, text_feat, feat_labels = extract_features_for_visualization(
        model, test_loader, DEVICE, modality="fusion"
    )
    visualize_tsne(
        speech_feat, feat_labels,
        "Fusion - Speech Temporal Features (t-SNE)",
        "../../Results/plots/fusion_speech_temporal_tsne.png"
    )
    visualize_tsne(
        text_feat, feat_labels,
        "Fusion - Text Contextual Features (t-SNE)",
        "../../Results/plots/fusion_text_contextual_tsne.png"
    )
    visualize_tsne(
        fused_feat, feat_labels,
        "Fusion - Combined Representation (t-SNE)",
        "../../Results/plots/fusion_combined_tsne.png"
    )

    # Save results
    save_results(y_true, y_pred, accuracy, report, errors, low_conf)

    # Generate comparison across all models
    generate_comparison_table()

    print("\nTesting complete!")

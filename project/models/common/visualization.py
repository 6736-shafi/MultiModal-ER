"""Shared visualization utilities for all pipelines."""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from .constants import EMOTIONS, EMOTION_LABELS


def visualize_tsne(features, labels, title, save_path):
    """Visualize emotion clusters using t-SNE."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    emotion_names = [EMOTION_LABELS[e] for e in EMOTIONS]
    for i, emotion in enumerate(emotion_names):
        mask = labels == i
        if mask.sum() > 0:
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], label=emotion, alpha=0.6, s=20)

    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"t-SNE plot saved to {save_path}")


def plot_training_history(history, model_name, save_dir="../../Results/plots"):
    """Plot and save training curves."""
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} Model - Loss Curves")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train Acc")
    ax2.plot(history["val_acc"], label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{model_name} Model - Accuracy Curves")
    ax2.legend()

    plt.tight_layout()
    filename = f"{model_name.lower()}_training_curves.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close()
    print(f"Training curves saved to {save_dir}/{filename}")


def plot_confusion_matrix(y_true, y_pred, title, save_path, cmap='Blues'):
    """Plot and save confusion matrix."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    emotion_names = [EMOTION_LABELS[e] for e in EMOTIONS]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=emotion_names, yticklabels=emotion_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def extract_features_for_visualization(model, dataloader, device, modality="speech"):
    """Extract learned representations for t-SNE visualization.

    Args:
        model: trained model with return_features=True support
        dataloader: data loader
        device: torch device
        modality: "speech", "text", or "fusion"

    Returns:
        For speech/text: (features, labels)
        For fusion: (fused_features, speech_features, text_features, labels)
    """
    model.eval()
    all_labels = []

    if modality == "fusion":
        all_fused, all_speech, all_text = [], [], []
        with torch.no_grad():
            for batch in dataloader:
                audio_features, token_ids, labels = batch
                audio_features = audio_features.to(device)
                token_ids = token_ids.to(device)
                fused, speech_repr, text_repr = model(
                    audio_features, token_ids, return_features=True
                )
                all_fused.append(fused.cpu().numpy())
                all_speech.append(speech_repr.cpu().numpy())
                all_text.append(text_repr.cpu().numpy())
                all_labels.append(labels.numpy())
        return (np.concatenate(all_fused),
                np.concatenate(all_speech),
                np.concatenate(all_text),
                np.concatenate(all_labels))
    else:
        all_features = []
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch[0], batch[-1]
                inputs = inputs.to(device)
                features = model(inputs, return_features=True)
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())
        return np.concatenate(all_features), np.concatenate(all_labels)


def analyze_low_confidence_correct(y_true, y_pred, y_probs, identifiers, n=5):
    """Find correctly classified samples with lowest confidence.

    Args:
        y_true: true labels
        y_pred: predicted labels
        y_probs: prediction probabilities
        identifiers: list of sample identifiers (file paths or texts)
        n: number of results to return

    Returns:
        list of dicts with low-confidence correct predictions
    """
    results = []
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            conf = float(y_probs[i][y_pred[i]])
            results.append({
                "identifier": identifiers[i] if i < len(identifiers) else f"sample_{i}",
                "true_emotion": EMOTION_LABELS[EMOTIONS[y_true[i]]],
                "confidence": conf,
                "runner_up": EMOTION_LABELS[EMOTIONS[
                    int(np.argsort(y_probs[i])[-2])
                ]],
                "runner_up_confidence": float(sorted(y_probs[i])[-2])
            })

    results.sort(key=lambda x: x["confidence"])
    if results[:n]:
        print(f"\nTop {min(n, len(results))} Low-Confidence Correct Predictions:")
        print("-" * 70)
        for r in results[:n]:
            ident = os.path.basename(r['identifier']) if '/' in r['identifier'] else r['identifier']
            print(f"  {ident}")
            print(f"  Correct: {r['true_emotion']} (conf: {r['confidence']:.3f}), "
                  f"Runner-up: {r['runner_up']} ({r['runner_up_confidence']:.3f})")
            print()

    return results[:n]

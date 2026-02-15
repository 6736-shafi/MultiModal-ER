"""
Speech Emotion Recognition - Training Pipeline
Architecture: MFCC Feature Extraction → BiLSTM Temporal Modelling → FC Classifier
Dataset: TESS (Toronto Emotional Speech Set)
"""

import os
import sys
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Add parent directory for common imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common import (
    EMOTIONS, EMOTION_MAP, EMOTION_LABELS, DEVICE, SEED,
    SAMPLE_RATE, N_MFCC, MAX_AUDIO_LEN, BATCH_SIZE, EPOCHS,
    LEARNING_RATE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, DATASET_PATH,
    set_seed, load_tess_data, visualize_tsne, plot_training_history,
    extract_features_for_visualization
)


# ========================= Dataset =========================
class TESSAudioDataset(Dataset):
    """TESS dataset loader for speech emotion recognition."""

    def __init__(self, file_paths, labels, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, max_len=MAX_AUDIO_LEN):
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.max_len = max_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]

        # Preprocessing: Load, resample, trim silence
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate)
        waveform, _ = librosa.effects.trim(waveform, top_db=25)

        # Feature Extraction: MFCC (time_steps × features)
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=self.n_mfcc)
        # Add delta and delta-delta features
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.concatenate([mfcc, delta, delta2], axis=0)  # (3*n_mfcc, time)
        features = features.T  # (time, 3*n_mfcc)

        # Pad or truncate to fixed length
        if features.shape[0] < self.max_len:
            pad_width = self.max_len - features.shape[0]
            features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
        else:
            features = features[:self.max_len, :]

        features = torch.FloatTensor(features)
        label = torch.LongTensor([label])
        return features, label.squeeze()


# ========================= Model =========================
class SpeechEmotionModel(nn.Module):
    """
    Speech Emotion Recognition Model
    - Feature Extraction: MFCC + Delta + Delta-Delta (done in dataset)
    - Temporal Modelling: Bidirectional LSTM
    - Classifier: Fully Connected layers
    """

    def __init__(self, input_size=N_MFCC * 3, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, num_classes=7, dropout=DROPOUT):
        super(SpeechEmotionModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Temporal Modelling: BiLSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

        # Attention mechanism for weighted temporal pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Classifier: FC layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x, return_features=False):
        # x: (batch, time_steps, features)
        lstm_out, _ = self.lstm(x)  # (batch, time, hidden*2)
        lstm_out = self.layer_norm(lstm_out)

        # Attention-based pooling
        attn_weights = self.attention(lstm_out)  # (batch, time, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*2)
        context = self.dropout(context)

        if return_features:
            return context  # Return temporal features for visualization

        logits = self.classifier(context)
        return logits


# ========================= Data Loading =========================
def get_dataloaders(dataset_path):
    """Create train/val/test dataloaders."""
    file_paths, labels, words = load_tess_data(dataset_path)
    print(f"Total samples: {len(file_paths)}")
    print(f"Emotion distribution: {dict(zip(EMOTIONS, np.bincount(labels)))}")

    # Split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        file_paths, labels, test_size=0.3, random_state=SEED, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )

    train_dataset = TESSAudioDataset(X_train, y_train)
    val_dataset = TESSAudioDataset(X_val, y_val)
    test_dataset = TESSAudioDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# ========================= Training =========================
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE):
    """Train the speech emotion model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "speech_model_best.pth")

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")

    history = {
        "train_loss": train_losses, "val_loss": val_losses,
        "train_acc": train_accs, "val_acc": val_accs
    }
    return history


# ========================= Main =========================
if __name__ == "__main__":
    set_seed(SEED)

    print("=" * 60)
    print("Speech Emotion Recognition - Training")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(DATASET_PATH)

    # Initialize model
    model = SpeechEmotionModel().to(DEVICE)
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    history = train_model(model, train_loader, val_loader)

    # Plot training history
    plot_training_history(history, "Speech")

    # Load best model and extract features for visualization
    model.load_state_dict(torch.load("speech_model_best.pth", map_location=DEVICE, weights_only=True))

    features, labels = extract_features_for_visualization(model, test_loader, DEVICE, modality="speech")
    visualize_tsne(
        features, labels,
        "Speech Temporal Modelling - Emotion Clusters (t-SNE)",
        "../../Results/plots/speech_temporal_tsne.png"
    )

    # Save split info for test.py
    file_paths, all_labels, words = load_tess_data(DATASET_PATH)
    _, X_temp, _, y_temp = train_test_split(
        file_paths, all_labels, test_size=0.3, random_state=SEED, stratify=all_labels
    )
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )
    np.savez("test_split.npz", file_paths=X_test, labels=y_test)

    print("\nTraining complete! Model saved as speech_model_best.pth")

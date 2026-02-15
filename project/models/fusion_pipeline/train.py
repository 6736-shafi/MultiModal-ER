"""
Multimodal Emotion Recognition - Fusion Training Pipeline
Architecture: Speech BiLSTM + Text BiLSTM → Late Fusion (Concatenation) → FC Classifier
Dataset: TESS (Toronto Emotional Speech Set)
"""

import os
import sys
import pickle
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
    SAMPLE_RATE, N_MFCC, MAX_AUDIO_LEN, MAX_SEQ_LEN,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, EMBEDDING_DIM,
    HIDDEN_SIZE, NUM_LAYERS, DROPOUT, DATASET_PATH,
    set_seed, Vocabulary, load_tess_data, visualize_tsne,
    plot_training_history, extract_features_for_visualization,
    get_transcripts_for_files
)


# ========================= Dataset =========================
class TESSMultimodalDataset(Dataset):
    """TESS dataset for multimodal (speech + text) emotion recognition."""

    def __init__(self, file_paths, texts, labels, vocab,
                 sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC,
                 max_audio_len=MAX_AUDIO_LEN, max_seq_len=MAX_SEQ_LEN):
        self.file_paths = file_paths
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.max_audio_len = max_audio_len
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        text = self.texts[idx]
        label = self.labels[idx]

        # === Speech Preprocessing & Feature Extraction ===
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate)
        waveform, _ = librosa.effects.trim(waveform, top_db=25)

        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=self.n_mfcc)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        audio_features = np.concatenate([mfcc, delta, delta2], axis=0).T

        if audio_features.shape[0] < self.max_audio_len:
            pad_width = self.max_audio_len - audio_features.shape[0]
            audio_features = np.pad(audio_features, ((0, pad_width), (0, 0)), mode='constant')
        else:
            audio_features = audio_features[:self.max_audio_len, :]

        # === Text Preprocessing ===
        token_ids = self.vocab.encode(text, self.max_seq_len)

        audio_features = torch.FloatTensor(audio_features)
        token_ids = torch.LongTensor(token_ids)
        label = torch.LongTensor([label])

        return audio_features, token_ids, label.squeeze()


# ========================= Model =========================
class SpeechEncoder(nn.Module):
    """Speech branch: BiLSTM for temporal modelling."""

    def __init__(self, input_size=N_MFCC * 3, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super(SpeechEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return context


class TextEncoder(nn.Module):
    """Text branch: Embedding + BiLSTM for contextual modelling."""

    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM,
                 hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super(TextEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.layer_norm(lstm_out)
        attn_weights = self.attention(lstm_out)
        mask = (x != 0).unsqueeze(-1).float()
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return context


class MultimodalFusionModel(nn.Module):
    """
    Multimodal Emotion Recognition Model
    - Speech Branch: MFCC → BiLSTM (Temporal Modelling)
    - Text Branch: Embeddings → BiLSTM (Contextual Modelling)
    - Fusion: Late fusion via concatenation
    - Classifier: FC layers
    """

    def __init__(self, vocab_size, hidden_size=HIDDEN_SIZE, num_classes=7, dropout=DROPOUT):
        super(MultimodalFusionModel, self).__init__()

        self.speech_encoder = SpeechEncoder(hidden_size=hidden_size)
        self.text_encoder = TextEncoder(vocab_size, hidden_size=hidden_size)

        self.dropout = nn.Dropout(dropout)

        # Fusion layer: project concatenated features
        fusion_input_size = hidden_size * 2 + hidden_size * 2  # speech + text (both bidirectional)
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_size * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, audio_features, token_ids, return_features=False):
        speech_repr = self.speech_encoder(audio_features)  # (batch, hidden*2)
        text_repr = self.text_encoder(token_ids)            # (batch, hidden*2)

        # Late Fusion: Concatenation
        fused = torch.cat([speech_repr, text_repr], dim=1)  # (batch, hidden*4)
        fused = self.fusion_layer(fused)                     # (batch, hidden*2)

        if return_features:
            return fused, speech_repr, text_repr

        logits = self.classifier(fused)
        return logits


# ========================= Training =========================
def get_dataloaders(dataset_path):
    """Create train/val/test dataloaders for multimodal pipeline."""
    file_paths, labels, words = load_tess_data(dataset_path)
    print(f"Total samples: {len(file_paths)}")

    # Get Whisper transcripts
    texts = get_transcripts_for_files(file_paths)

    # Build vocabulary
    vocab = Vocabulary()
    vocab.build(texts)
    print(f"Vocabulary size: {len(vocab)}")

    # Split using index-based approach for consistency
    indices = list(range(len(file_paths)))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, random_state=SEED,
        stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=SEED,
        stratify=[labels[i] for i in temp_idx]
    )

    def subset(indices):
        return ([file_paths[i] for i in indices],
                [texts[i] for i in indices],
                [labels[i] for i in indices])

    train_data = subset(train_idx)
    val_data = subset(val_idx)
    test_data = subset(test_idx)

    train_dataset = TESSMultimodalDataset(*train_data, vocab)
    val_dataset = TESSMultimodalDataset(*val_data, vocab)
    test_dataset = TESSMultimodalDataset(*test_data, vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, vocab


def train_model(model, train_loader, val_loader, vocab, epochs=EPOCHS, lr=LEARNING_RATE):
    """Train the multimodal fusion model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for audio_features, token_ids, labels in train_loader:
            audio_features = audio_features.to(DEVICE)
            token_ids = token_ids.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(audio_features, token_ids)
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

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for audio_features, token_ids, labels in val_loader:
                audio_features = audio_features.to(DEVICE)
                token_ids = token_ids.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(audio_features, token_ids)
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
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': len(vocab)
            }, "fusion_model_best.pth")

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
    print("Multimodal Emotion Recognition - Fusion Training")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Load data
    train_loader, val_loader, test_loader, vocab = get_dataloaders(DATASET_PATH)

    # Save vocabulary
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    # Initialize model
    model = MultimodalFusionModel(vocab_size=len(vocab)).to(DEVICE)
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    history = train_model(model, train_loader, val_loader, vocab)

    # Plot training history
    plot_training_history(history, "Fusion")

    # Load best model and visualize
    checkpoint = torch.load("fusion_model_best.pth", map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    fused_feat, speech_feat, text_feat, labels = extract_features_for_visualization(
        model, test_loader, DEVICE, modality="fusion"
    )

    # Visualize all three representation spaces
    visualize_tsne(
        speech_feat, labels,
        "Fusion Model - Speech Temporal Features (t-SNE)",
        "../../Results/plots/fusion_speech_temporal_tsne.png"
    )
    visualize_tsne(
        text_feat, labels,
        "Fusion Model - Text Contextual Features (t-SNE)",
        "../../Results/plots/fusion_text_contextual_tsne.png"
    )
    visualize_tsne(
        fused_feat, labels,
        "Fusion Model - Fused Representation (t-SNE)",
        "../../Results/plots/fusion_combined_tsne.png"
    )

    print("\nTraining complete! Model saved as fusion_model_best.pth")

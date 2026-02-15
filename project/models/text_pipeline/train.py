"""
Text Emotion Recognition - Training Pipeline
Architecture: Tokenization → Word Embeddings → BiLSTM Contextual Modelling → FC Classifier
Dataset: TESS (Toronto Emotional Speech Set) - Whisper ASR transcripts
"""

import os
import sys
import pickle
import numpy as np
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
    MAX_SEQ_LEN, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, DATASET_PATH,
    set_seed, Vocabulary, load_tess_data, visualize_tsne,
    plot_training_history, extract_features_for_visualization,
    transcribe_dataset, get_transcripts_for_files
)


# ========================= Dataset =========================
class TESSTextDataset(Dataset):
    """TESS dataset loader for text emotion recognition."""

    def __init__(self, texts, labels, vocab, max_len=MAX_SEQ_LEN):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Preprocessing: tokenize and encode
        token_ids = self.vocab.encode(text, self.max_len)
        token_ids = torch.LongTensor(token_ids)
        label = torch.LongTensor([label])

        return token_ids, label.squeeze()


# ========================= Model =========================
class TextEmotionModel(nn.Module):
    """
    Text Emotion Recognition Model
    - Feature Extraction: Learned Word Embeddings (tokens × features)
    - Contextual Modelling: Bidirectional LSTM
    - Classifier: Fully Connected layers
    """

    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM,
                 hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                 num_classes=7, dropout=DROPOUT):
        super(TextEmotionModel, self).__init__()

        # Feature Extraction: Word Embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Contextual Modelling: BiLSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x, return_features=False):
        # x: (batch, seq_len) - token indices
        embedded = self.embedding(x)  # (batch, seq_len, emb_dim)

        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden*2)
        lstm_out = self.layer_norm(lstm_out)

        # Attention-based pooling
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        # Mask padding tokens
        mask = (x != 0).unsqueeze(-1).float()  # (batch, seq_len, 1)
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*2)
        context = self.dropout(context)

        if return_features:
            return context  # Return contextual features for visualization

        logits = self.classifier(context)
        return logits


# ========================= Training =========================
def get_dataloaders(dataset_path):
    """Create train/val/test dataloaders using Whisper transcripts."""
    file_paths, labels, words = load_tess_data(dataset_path)
    print(f"Total samples: {len(file_paths)}")

    # Get Whisper transcripts
    texts = get_transcripts_for_files(file_paths)
    print(f"Sample transcripts: {texts[:3]}")

    # Build vocabulary from transcripts
    vocab = Vocabulary()
    vocab.build(texts)
    print(f"Vocabulary size: {len(vocab)}")

    # Index-based splitting for consistency across pipelines
    indices = list(range(len(file_paths)))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, random_state=SEED, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=SEED,
        stratify=[labels[i] for i in temp_idx]
    )

    def subset_texts(idxs):
        return [texts[i] for i in idxs], [labels[i] for i in idxs]

    train_texts, train_labels = subset_texts(train_idx)
    val_texts, val_labels = subset_texts(val_idx)
    test_texts, test_labels = subset_texts(test_idx)

    train_dataset = TESSTextDataset(train_texts, train_labels, vocab)
    val_dataset = TESSTextDataset(val_texts, val_labels, vocab)
    test_dataset = TESSTextDataset(test_texts, test_labels, vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, vocab


def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE):
    """Train the text emotion model."""
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

        for token_ids, labels in train_loader:
            token_ids, labels = token_ids.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(token_ids)
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
            for token_ids, labels in val_loader:
                token_ids, labels = token_ids.to(DEVICE), labels.to(DEVICE)
                outputs = model(token_ids)
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
                'vocab_size': len(model.embedding.weight)
            }, "text_model_best.pth")

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
    print("Text Emotion Recognition - Training")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Load data
    train_loader, val_loader, test_loader, vocab = get_dataloaders(DATASET_PATH)

    # Save vocabulary for test.py
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    # Initialize model
    model = TextEmotionModel(vocab_size=len(vocab)).to(DEVICE)
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    history = train_model(model, train_loader, val_loader)

    # Plot training history
    plot_training_history(history, "Text")

    # Load best model and visualize
    checkpoint = torch.load("text_model_best.pth", map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    features, labels = extract_features_for_visualization(model, test_loader, DEVICE, modality="text")
    visualize_tsne(
        features, labels,
        "Text Contextual Modelling - Emotion Clusters (t-SNE)",
        "../../Results/plots/text_contextual_tsne.png"
    )

    print("\nTraining complete! Model saved as text_model_best.pth")

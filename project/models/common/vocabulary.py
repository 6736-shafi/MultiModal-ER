"""Vocabulary class for text tokenization and encoding."""

import re
from collections import Counter


class Vocabulary:
    """Simple vocabulary builder for tokenization."""

    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.word_count = Counter()

    def build(self, texts):
        """Build vocabulary from list of texts."""
        for text in texts:
            tokens = self.tokenize(text)
            self.word_count.update(tokens)

        for word, count in self.word_count.most_common():
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def tokenize(self, text):
        """Clean and tokenize text."""
        text = text.lower().strip()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.split()

    def encode(self, text, max_len):
        """Convert text to token indices with padding."""
        tokens = self.tokenize(text)
        indices = [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]

        # Pad or truncate
        if len(indices) < max_len:
            indices = indices + [self.word2idx["<PAD>"]] * (max_len - len(indices))
        else:
            indices = indices[:max_len]

        return indices

    def __len__(self):
        return len(self.word2idx)

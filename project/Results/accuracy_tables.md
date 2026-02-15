# Accuracy Tables - All 3 Model Variants

## 1. Overall Model Comparison

| Model | Architecture | Parameters | Test Accuracy |
|-------|-------------|-----------|---------------|
| Speech-only | MFCC + BiLSTM + Attention | 718,600 | **100.00%** |
| Text-only | Embeddings + BiLSTM + Attention | 684,488 | **15.00%** |
| Multimodal Fusion | Speech BiLSTM + Text BiLSTM + Concatenation | 1,501,129 | **99.76%** |

## 2. Per-Class Test Accuracy (%)

| Emotion | Speech-only | Text-only | Fusion |
|---------|------------|-----------|--------|
| Angry | 100.00 | 0.00 | 100.00 |
| Disgust | 100.00 | 0.00 | 98.33 |
| Fear | 100.00 | 0.00 | 100.00 |
| Happy | 100.00 | 5.00 | 100.00 |
| Neutral | 100.00 | 0.00 | 100.00 |
| Pleasant Surprise | 100.00 | 100.00* | 100.00 |
| Sad | 100.00 | 0.00 | 100.00 |

*\*Text model predicts Pleasant Surprise for most samples (class collapse)*

## 3. Speech-only Model - Detailed Classification Report

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Angry | 1.00 | 1.00 | 1.00 | 120 |
| Disgust | 1.00 | 1.00 | 1.00 | 120 |
| Fear | 1.00 | 1.00 | 1.00 | 120 |
| Happy | 1.00 | 1.00 | 1.00 | 120 |
| Neutral | 1.00 | 1.00 | 1.00 | 120 |
| Pleasant Surprise | 1.00 | 1.00 | 1.00 | 120 |
| Sad | 1.00 | 1.00 | 1.00 | 120 |
| **Macro Avg** | **1.00** | **1.00** | **1.00** | **840** |

## 4. Text-only Model - Detailed Classification Report

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Angry | 0.00 | 0.00 | 0.00 | 120 |
| Disgust | 0.00 | 0.00 | 0.00 | 120 |
| Fear | 0.00 | 0.00 | 0.00 | 120 |
| Happy | 0.60 | 0.05 | 0.09 | 120 |
| Neutral | 0.00 | 0.00 | 0.00 | 120 |
| Pleasant Surprise | 0.14 | 1.00 | 0.25 | 120 |
| Sad | 0.00 | 0.00 | 0.00 | 120 |
| **Macro Avg** | **0.11** | **0.15** | **0.05** | **840** |

## 5. Fusion Model - Detailed Classification Report

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Angry | 1.00 | 1.00 | 1.00 | 120 |
| Disgust | 1.00 | 0.98 | 0.99 | 120 |
| Fear | 1.00 | 1.00 | 1.00 | 120 |
| Happy | 1.00 | 1.00 | 1.00 | 120 |
| Neutral | 1.00 | 1.00 | 1.00 | 120 |
| Pleasant Surprise | 1.00 | 1.00 | 1.00 | 120 |
| Sad | 0.98 | 1.00 | 0.99 | 120 |
| **Macro Avg** | **1.00** | **1.00** | **1.00** | **840** |

## 6. Training History Summary

| Model | Best Epoch | Final Train Loss | Final Val Loss | Final Val Acc |
|-------|-----------|-----------------|----------------|---------------|
| Speech-only | 10 | 0.0003 | 0.0000 | 100.00% |
| Text-only | 25 | 1.9457 | 1.9458 | 14.88% |
| Fusion | 10 | 0.0006 | 0.0001 | 100.00% |

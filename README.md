# MultiModal-ER: Multimodal Emotion Recognition

A **Multimodal Emotion Recognition** system that classifies 7 emotions using **Speech-only**, **Text-only**, and **Multimodal Fusion** (Speech + Text) pipelines. Built with PyTorch using BiLSTM + Attention architectures on the [TESS dataset](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess).

> **Assignment 2** — IIIT Hyderabad

---

## Results

| Model | Architecture | Test Accuracy |
|-------|-------------|---------------|
| **Speech-only** | MFCC + BiLSTM + Attention | **100.00%** |
| **Text-only** | Whisper ASR + Embeddings + BiLSTM + Attention | **15.00%** |
| **Multimodal Fusion** | Late Fusion (Speech + Text) + Classifier | **99.76%** |

> See [`project/Report.md`](project/Report.md) for detailed architecture decisions, experimental analysis, error analysis, and t-SNE visualizations.
> See [`project/Results/accuracy_tables.md`](project/Results/accuracy_tables.md) for per-class accuracy and classification reports.

---

## Project Structure

```
project/
├── models/
│   ├── common/                    # Shared utilities
│   │   ├── constants.py           # Hyperparameters, emotion labels, device config
│   │   ├── vocabulary.py          # Vocabulary class for text tokenization
│   │   ├── data_loading.py        # TESS dataset loader
│   │   ├── visualization.py       # t-SNE, confusion matrix, training curves
│   │   ├── seed.py                # Reproducibility seeds
│   │   └── transcribe.py          # Whisper ASR transcription with JSON caching
│   ├── speech_pipeline/
│   │   ├── train.py               # Train: MFCC → BiLSTM → Attention → Classifier
│   │   └── test.py                # Evaluate + generate plots
│   ├── text_pipeline/
│   │   ├── train.py               # Train: Whisper ASR → Embeddings → BiLSTM → Classifier
│   │   └── test.py                # Evaluate + generate plots
│   └── fusion_pipeline/
│       ├── train.py               # Train: Speech + Text → Concat → Classifier
│       └── test.py                # Evaluate + comparison + all t-SNE plots
│
├── Results/
│   ├── accuracy_tables.md         # All 3 model accuracy tables
│   ├── plots/                     # Training curves, confusion matrices, t-SNE plots
│   ├── *_results.json             # Per-model metrics
│   └── *_classification_report.txt
│
├── Report.md                      # Full report (Architecture, Experiments, Analysis)
├── README.md                      # Project-level README with detailed instructions
└── requirements.txt               # Python dependencies
```

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- pip

### 1. Clone the Repository
```bash
git clone https://github.com/6736-shafi/MultiModal-ER.git
cd MultiModal-ER/project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

This installs: PyTorch, librosa, OpenAI Whisper, scikit-learn, matplotlib, seaborn, and other dependencies.

### 3. Download the TESS Dataset
Download from [Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) and extract into the `project/` directory:

```bash
# Using Kaggle CLI:
kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess
unzip toronto-emotional-speech-set-tess.zip -d "TESS Toronto emotional speech set data"
```

The dataset folder should be at:
```
project/TESS Toronto emotional speech set data/TESS Toronto emotional speech set data/
```

---

## Running the Pipelines

Run all 3 pipelines from the `project/` directory:

### 1. Speech-Only Pipeline
```bash
cd models/speech_pipeline
python train.py    # ~10 min on CPU
python test.py     # Generates confusion matrix, t-SNE, metrics
```

### 2. Text-Only Pipeline
```bash
cd ../text_pipeline
python train.py    # First run: Whisper transcription (~30 min), then training (~2 min)
python test.py     # Generates confusion matrix, t-SNE, metrics
```

> **Note:** The first run transcribes all 5,600 audio files using Whisper. Transcriptions are cached in `transcriptions.json` for subsequent runs.

### 3. Multimodal Fusion Pipeline
```bash
cd ../fusion_pipeline
python train.py    # ~15 min on CPU
python test.py     # Generates comparison table, all t-SNE plots
```

All results (plots, accuracy tables, classification reports) are saved to `Results/`.

---

## Technical Details

| Component | Choice |
|-----------|--------|
| **Framework** | PyTorch |
| **Speech Features** | MFCC (40 coefficients) + Delta + Delta-Delta |
| **Text Transcription** | OpenAI Whisper (base model) |
| **Text Features** | Learned Embeddings (64-d) |
| **Temporal/Contextual Model** | Bidirectional LSTM (2-layer, 128 hidden) |
| **Attention** | Self-Attention Pooling |
| **Fusion Strategy** | Late Fusion (Feature Concatenation) |
| **Classifier** | 2-layer Fully Connected Network |
| **Optimizer** | Adam (lr=0.001, weight_decay=1e-4) |
| **Data Split** | 70% train / 15% val / 15% test (stratified) |
| **Reproducibility** | Seeded (seed=42) — torch, numpy, random, cudnn |

---

## Dataset

**TESS (Toronto Emotional Speech Set)**

| Property | Details |
|----------|---------|
| Total Samples | 5,600 audio files (WAV) |
| Speakers | 2 female (older + younger) |
| Words | 200 target words per speaker |
| Emotions | 7: Angry, Disgust, Fear, Happy, Neutral, Pleasant Surprise, Sad |
| Balance | 800 samples per emotion (perfectly balanced) |
| Source | [Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) |

---

## Key Findings

1. **Speech-only achieves 100% accuracy** — MFCC features with BiLSTM are highly effective on TESS's clean, studio-recorded emotional speech.
2. **Text-only achieves only 15%** — TESS uses the carrier phrase "Say the word X" for all emotions, so text carries no emotional information.
3. **Fusion achieves 99.76%** — Late fusion works well but slightly underperforms speech-only because the uninformative text branch introduces minor noise (2 Disgust→Sad misclassifications).

---

## References

1. Dupuis, K., & Pichora-Fuller, M. K. (2010). *Toronto Emotional Speech Set (TESS)*. University of Toronto.
2. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
3. Radford, A. et al. (2023). *Robust Speech Recognition via Large-Scale Weak Supervision*. ICML.

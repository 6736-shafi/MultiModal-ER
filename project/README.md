# MultiModal-ER: Multimodal Emotion Recognition

**Assignment 2** — Build a system that recognizes emotions using Speech-only, Text-only, and Multimodal (Speech + Text) inputs.

## Project Structure

```
project/
├── models/
│   ├── common/                   # Shared utilities (constants, vocab, visualization, etc.)
│   │   ├── constants.py          # Hyperparameters, emotion labels, device config
│   │   ├── vocabulary.py         # Vocabulary class for text tokenization
│   │   ├── data_loading.py       # TESS dataset loader
│   │   ├── visualization.py      # t-SNE, confusion matrix, training curves
│   │   ├── seed.py               # Reproducibility (torch, numpy, random, cudnn)
│   │   └── transcribe.py         # Whisper ASR transcription with JSON cache
│   ├── speech_pipeline/
│   │   ├── train.py              # Speech: MFCC → BiLSTM → Classifier
│   │   └── test.py               # Evaluation, confusion matrix, t-SNE
│   ├── text_pipeline/
│   │   ├── train.py              # Text: Whisper ASR → Embeddings → BiLSTM → Classifier
│   │   └── test.py               # Evaluation, confusion matrix, t-SNE
│   └── fusion_pipeline/
│       ├── train.py              # Fusion: Speech + Text → Concat → Classifier
│       └── test.py               # Evaluation, comparison table, all t-SNE
│
├── transcriptions.json              # Whisper ASR cache (auto-generated on first text/fusion run)
├── Results/
│   ├── accuracy_tables.md        # All 3 model variants accuracy tables
│   ├── comparison_results.json   # JSON comparison data
│   ├── speech_results.json       # Speech model metrics
│   ├── text_results.json         # Text model metrics
│   ├── fusion_results.json       # Fusion model metrics
│   ├── *_classification_report.txt
│   └── plots/
│       ├── speech_training_curves.png
│       ├── text_training_curves.png
│       ├── fusion_training_curves.png
│       ├── speech_confusion_matrix.png
│       ├── text_confusion_matrix.png
│       ├── fusion_confusion_matrix.png
│       ├── speech_temporal_tsne.png        # Temporal Modelling block
│       ├── text_contextual_tsne.png        # Contextual Modelling block
│       ├── fusion_combined_tsne.png        # Fusion block
│       ├── fusion_speech_temporal_tsne.png
│       ├── fusion_text_contextual_tsne.png
│       └── model_comparison.png
│
├── Report.md                     # Detailed report (Architecture, Experiments, Analysis)
├── README.md                     # This file (Setup & Instructions)
└── requirements.txt
```

## Dataset

**TESS (Toronto Emotional Speech Set)** — [Kaggle Link](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

| Property | Details |
|----------|---------|
| Samples | 5,600 audio files (WAV) |
| Speakers | 2 female (older + younger) |
| Words | 200 target words per speaker |
| Emotions | 7: Angry, Disgust, Fear, Happy, Neutral, Pleasant Surprise, Sad |
| Balance | 800 samples per emotion (perfectly balanced) |

## Setup & Installation

### Prerequisites
- Python 3.8+
- pip

### Step 1: Clone the Repository
```bash
git clone <repo-url>
cd project
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

This installs PyTorch, librosa, OpenAI Whisper (for ASR transcription), scikit-learn, and visualization libraries.

### Step 3: Download the Dataset
Download from [Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) and extract into the project root:

```bash
# Using Kaggle CLI (if configured):
kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess
unzip toronto-emotional-speech-set-tess.zip -d "TESS Toronto emotional speech set data"
```

Your folder should look like:
```
project/
├── TESS Toronto emotional speech set data/
│   ├── TESS Toronto emotional speech set data/
│   │   ├── OAF_angry/
│   │   ├── OAF_disgust/
│   │   ├── ... (14 folders total)
│   │   └── YAF_sad/
```

Alternatively, set a custom dataset path:
```bash
export TESS_DATASET_PATH="/your/path/to/TESS Toronto emotional speech set data/TESS Toronto emotional speech set data"
```

## Running the Pipelines

**Run all 3 pipelines in order** (each train.py must finish before its test.py):

### 1. Speech-Only Pipeline
```bash
cd models/speech_pipeline
python train.py    # Trains speech model (~10 min on CPU)
python test.py     # Generates results, confusion matrix, t-SNE
```

### 2. Text-Only Pipeline
```bash
cd ../text_pipeline
python train.py    # Transcribes audio via Whisper on first run (~30 min), then trains (~2 min)
python test.py     # Generates results, confusion matrix, t-SNE
```

> **Note:** The first run of the text or fusion pipeline transcribes all 5,600 audio files using Whisper (base model). Transcriptions are cached to `project/transcriptions.json` so subsequent runs are instant.

### 3. Multimodal Fusion Pipeline
```bash
cd ../fusion_pipeline
python train.py    # Trains fusion model (~15 min on CPU)
python test.py     # Generates results, comparison table, all t-SNE plots
```

After running all pipelines, check `Results/` for accuracy tables, plots, and classification reports.

## Results Summary

| Model | Test Accuracy |
|-------|--------------|
| Speech-only (BiLSTM) | **100.00%** |
| Text-only (BiLSTM) | **15.00%** |
| Multimodal Fusion | **99.76%** |

See `Report.md` for detailed architecture decisions, experimental analysis, error analysis, and t-SNE visualizations.

## Technical Details

| Component | Choice |
|-----------|--------|
| Framework | PyTorch |
| Speech Features | MFCC (40) + Delta + Delta-Delta |
| Text Transcription | OpenAI Whisper (base model) |
| Text Features | Learned Embeddings (64-d) |
| Temporal/Contextual Model | Bidirectional LSTM (2-layer, 128 hidden) |
| Fusion Strategy | Late Fusion (Feature-level Concatenation) |
| Classifier | 2-layer FC |
| Optimizer | Adam (lr=0.001) |
| Data Split | 70/15/15 (train/val/test, stratified) |
| Reproducibility | Seeded (seed=42) across torch, numpy, random, cudnn |

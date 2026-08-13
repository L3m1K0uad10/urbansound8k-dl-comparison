# UrbanSound8K: Comparative Study of Deep Learning Architectures for Environmental Sound Classification

## Description

This project investigates different deep learning approaches for environmental sound classification using the UrbanSound8K dataset.

Three classification pipelines are explored:

1. **Vision-based classification** using Log-Mel spectrograms and a 2D ResNet18 CNN.
2. **Temporal classification** using raw waveform signals and the M5 1D CNN architecture.
3. **Attention-based classification** using Log-Mel spectrograms and a Mini Audio Spectrogram Transformer (Mini-AST).

The models are compared across multiple dimensions:

- Accuracy
- Precision
- Recall
- F1-score
- Number of parameters
- Model size
- Training time
- Inference latency
- Confusion matrices
- Error patterns and misclassifications

The best-performing model is subsequently deployed through a Streamlit demonstration application.

## Project Goal

### Main Goal

Learn audio machine learning by understanding how environmental sounds can be represented and classified using different neural network architectures.

### Specific Objectives

- Understand digital audio signals.
- Learn audio preprocessing techniques.
- Generate spectrogram representations.
- Build 2D CNN audio classifiers.
- Build 1D waveform classifiers.
- Explore Transformer-based audio classification.
- Compare multiple architectures and audio representations.
- Analyze classification errors and confusion patterns.
- Evaluate computational characteristics of the models.
- Deploy the best-performing model through a demonstration application.

## Dataset

### UrbanSound8K

UrbanSound8K contains 8,732 labeled audio clips belonging to 10 environmental sound classes.

| Class |
| --- |
| Air Conditioner |
| Car Horn |
| Children Playing |
| Dog Bark |
| Drilling |
| Engine Idling |
| Gun Shot |
| Jackhammer |
| Siren |
| Street Music |

## Overall Architecture

```text
                         Raw Audio (.wav)
                               |
                       Audio Preprocessing
            (Mono, Resample, Pad/Truncate, Normalize)
                               |
          ------------------------------------------------
          |                     |                        |
          |                     |                        |
    Pipeline 1             Pipeline 2              Pipeline 3
      Vision                Temporal                Attention
          |                     |                        |
   Log-Mel Spectrogram      Raw Waveform         Log-Mel Spectrogram
          |                     |                        |
      ResNet18                 M5 CNN          Mini-AST (4 Transformer
          |                     |               Encoder Layers)
          |                     |                        |
          ------------------------------------------------
                               |
                    Evaluation & Comparison
        (Accuracy, Precision, Recall, F1, Latency,
              Parameters, Model Size, Error Analysis)
                               |
                       Demo Application
```

## Pipeline 1 — Vision
```
Audio Waveform
      ↓
Preprocessing
      ↓
Log-Mel Spectrogram
      ↓
ResNet18
      ↓
10-Class Prediction
```

The ResNet18 model operates on the Log-Mel spectrogram representation of the audio signal.


## Pipeline 2 — Temporal
```
Audio Waveform
      ↓
Preprocessing
      ↓
Raw Waveform
      ↓
M5 1D CNN
      ↓
10-Class Prediction
```

The M5 architecture operates directly on the temporal waveform representation.

## Pipeline 3 — Attention
```
Audio Waveform
      ↓
Preprocessing
      ↓
Log-Mel Spectrogram
      ↓
Patch Extraction
      ↓
Mini-AST
      ↓
Transformer Encoder
      ↓
10-Class Prediction
```

The Mini-AST model uses Transformer-based attention mechanisms to model relationships between spectrogram patches.

## Demo Application Architecture

The best-performing model is deployed through a Streamlit application.
```
Upload Audio
     │
     ▼
Preprocessing
     │
     ├── Mono conversion
     ├── Resampling to 16 kHz
     ├── Padding / truncation to 4 seconds
     └── Waveform normalization
     │
     ▼
Log-Mel Spectrogram
     │
     ▼
ResNet18
     │
     ▼
Prediction
     │
     ├── Predicted class
     ├── Confidence
     ├── Top-3 predictions
     └── Class probability distribution
     │
     ▼
Visualizations
     ├── Waveform
     └── Log-Mel Spectrogram
```

## Final Preprocessing Specification

The preprocessing pipeline used by the deployed model is fixed to ensure that inference data follows the same representation used during training.
```
Audio
 ↓
Mono
 ↓
16,000 Hz
 ↓
4 seconds
 ↓
64,000 samples
 ↓
Mel Spectrogram
 ↓
128 Mel bands
 ↓
Log transformation
 ↓
Input shape: 1 × 128 × 126
 ↓
ResNet18
```

## Mel-Spectrogram Configuration
```
Sample rate  : 16,000 Hz
n_fft        : 1024
hop_length   : 512
n_mels       : 128
Duration     : 4 seconds
Samples      : 64,000
```

See `app/preprocessing.py` for the implementation of the preprocessing pipeline.

## Deployed Model

The demonstration application uses the trained ResNet18 checkpoint.
```
ResNet18 Architecture
        ↓
Load Trained Checkpoint
        ↓
model.eval()
        ↓
Inference
        ↓
10-Class Prediction
```

Model checkpoint:
```
models/best_resnet18.pth
```

## Project Structure
```
Environmental sound classification/
│
├── app/
│   ├── preprocessing.py
│   └── streamlit_app.py
│
├── models/
│   └── best_resnet18.pth
│
├── notebooks/
│   └── ...
│
├── test_preprocessing.py
├── test_model.py
├── test_inference.py
└── README.md
```

## Results

The three pipelines are evaluated using both predictive performance and computational characteristics.

The final comparison includes:
- Accuracy
- Precision
- Recall
- F1-score
- Number of parameters
- Model size
- Training time
- Inference latency
- Confusion matrices
- Misclassification analysis

The three pipelines were evaluated using predictive performance, computational characteristics, confusion matrices, and error analysis. ResNet18 achieved the strongest validation performance among the evaluated models.

[__see notebook/UrbanSound8K_Comparative_Study_audio_classification.ipynb__]

## Demo

The trained model can be interacted with through the Streamlit demonstration application.

The application allows users to:
- Upload an audio file.
- Preview the uploaded audio.
- Inspect the waveform.
- Inspect the Log-Mel spectrogram.
- Obtain the predicted environmental sound class.
- View prediction confidence.
- View the Top-3 predictions.
- Inspect the probability distribution across all 10 classes.
- Review information about the deployed model.
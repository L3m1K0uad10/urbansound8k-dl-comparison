# UrbanSound8K: Comparative Study of CNNs and Transformers for Environmental Sound Classification

## Description

This project investigates different deep learning approaches for environmental sound classification using the UrbanSound8K dataset. Three pipelines are explored:

1. Image-based classification using Mel-spectrograms and 2D CNNs.
2. Temporal modeling using raw waveform signals and 1D CNN architectures.
3. Attention-based models using Audio Spectrogram Transformers.

The models are compared based on:

- Classification accuracy
- Inference speed
- Model size

The best model is deployed through a demonstration application.

## Project Goal

### Main Goal

Learn audio machine learning by understanding how sounds can be represented and classified using different neural network architectures.

### Specific Objectives

- Understand digital audio signals.
- Learn audio preprocessing techniques.
- Generate spectrogram representations.
- Build 2D CNN audio classifiers.
- Build 1D waveform classifiers.
- Explore Transformer-based audio classification.
- Compare multiple architectures.
- Build a Demo App with the best model

## Dataset

### UrbanSound8K

Contains 8732 audio clips belonging to 10 classes:

| Class            |
| ---------------- |
| Air Conditioner  |
| Car Horn         |
| Children Playing |
| Dog Bark         |
| Drilling         |
| Engine Idling    |
| Gun Shot         |
| Jackhammer       |
| Siren            |
| Street Music     |


## Overall Architecure

```

                        Raw Audio (.wav)
                             |
                     Common Preprocessing
      (Mono, Resample, Pad/Truncate, Normalize)
                             |
          ------------------------------------------------
          |                     |                        |
          |                     |                        |
    Pipeline 1             Pipeline 2              Pipeline 3
      Vision                Temporal                Attention
          |                     |                        |
    Mel Spectrogram         Raw Waveform          Log-Mel Spectrogram
          |                     |                        |
     Simple CNN               M5 CNN                     AST
          |                     |                        |
        ResNet18            (optional)             Transformer Encoder
          ------------------------------------------------
                             |
                  Evaluation & Comparison
          (Accuracy, Latency, Parameters, F1)
                             |
                      Demo Application
```
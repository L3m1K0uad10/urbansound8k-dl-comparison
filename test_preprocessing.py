import torch
import torchaudio

from app.preprocessing import preprocess_audio


waveform, sample_rate = torchaudio.load(
    "siren_test.wav"
)

print("Original waveform:", waveform.shape)
print("Original sample rate:", sample_rate)

processed_waveform, log_mel_spec = preprocess_audio(
    waveform,
    sample_rate
)

print("Processed waveform:", processed_waveform.shape)
print("Log-Mel spectrogram:", log_mel_spec.shape)
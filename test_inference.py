import torch
import torchaudio

from app.preprocessing import preprocess_audio
from models.resnet18_model import create_resnet18


# configuration
DEVICE = torch.device("cpu")

AUDIO_PATH = "siren_test.wav"

CLASS_NAMES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]


# Load model
model = create_resnet18()

checkpoint = torch.load(
    "models/best_resnet18.pth",
    map_location = DEVICE
)

model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()


# Load audio
waveform, sample_rate = torchaudio.load(
    AUDIO_PATH
)

print("Original waveform:", waveform.shape)
print("Original sample rate:", sample_rate)


# Preprocess audio
processed_waveform, log_mel_spec = preprocess_audio(
    waveform,
    sample_rate
)

print("Processed waveform:", processed_waveform.shape)
print("Log-Mel spectrogram:", log_mel_spec.shape)


# Prepare model input

# Add batch dimension
model_input = log_mel_spec.unsqueeze(0)

model_input = model_input.to(DEVICE)

print("Model input:", model_input.shape)


# Inference
with torch.no_grad():

    logits = model(model_input)

    probabilities = torch.softmax(
        logits,
        dim = 1
    )

    predicted_index = torch.argmax(
        probabilities,
        dim = 1
    ).item()


# Results
predicted_class = CLASS_NAMES[predicted_index]

confidence = probabilities[
    0,
    predicted_index
].item()


print()
print("Prediction:", predicted_class)
print(f"Confidence: {confidence:.4f}")
print(f"Confidence: {confidence * 100:.2f}%")
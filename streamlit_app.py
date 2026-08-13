import os

import torch
import streamlit as st
import torchaudio

from app.preprocessing import preprocess_audio
from app.model import load_model, CLASS_NAMES


# Configuration
MODEL_PATH = "models/best_resnet18.pth"

DEVICE = torch.device("cpu")


# Page configuration
st.set_page_config(
    page_title = "Environmental Sound Classifier",
    page_icon = "🎧",
    layout = "centered"
)


# Model loading
@st.cache_resource
def load_trained_model():

    model = load_model(
        MODEL_PATH,
        DEVICE
    )

    return model


# Application
st.title("🎧 Environmental Sound Classifier")

st.write(
    """
    Upload an audio recording and the trained ResNet18 model
    will classify it into one of ten environmental sound categories.
    """
)


# Upload audio
uploaded_file = st.file_uploader(
    "Upload an audio file",
    type = ["wav", "mp3", "ogg", "flac", "m4a"]
)


if uploaded_file is not None:

    # Save uploaded audio temporarily
    temp_path = "temp_audio"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Audio player
    st.audio(
        uploaded_file
    )

    # Run classification
    if st.button("🔍 Classify Audio"):

        with st.spinner("Analyzing audio..."):

            waveform, sample_rate = torchaudio.load(
                temp_path
            )

            # Preprocess
            processed_waveform, log_mel = preprocess_audio(
                waveform,
                sample_rate
            )

            # Add batch dimension
            model_input = log_mel.unsqueeze(0)

            # Load model
            model = load_trained_model()

            # Inference
            with torch.no_grad():

                outputs = model(
                    model_input.to(DEVICE)
                )

                probabilities = torch.softmax(
                    outputs,
                    dim = 1
                )[0]


            # Prediction
            top_probability, top_index = torch.max(
                probabilities,
                dim = 0
            )

            predicted_class = CLASS_NAMES[
                top_index.item()
            ]

            confidence = top_probability.item()

            # Display result
            st.subheader("Prediction")

            st.success(
                predicted_class.replace("_", " ").title()
            )

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

            # Confidence warning
            if confidence < 0.50:

                st.warning(
                    "The model is uncertain about this prediction."
                )

            elif confidence < 0.75:

                st.info(
                    "The model has moderate confidence in this prediction."
                )

            else:

                st.success(
                    "The model is highly confident in this prediction."
                )


            # Top predictions
            st.subheader("Top Predictions")

            top_k = min(5, len(CLASS_NAMES))

            top_probs, top_indices = torch.topk(
                probabilities,
                top_k
            )

            for probability, index in zip(
                top_probs,
                top_indices
            ):

                class_name = CLASS_NAMES[
                    index.item()
                ]

                st.write(
                    f"**{class_name.replace('_', ' ').title()}**"
                )

                st.progress(
                    float(probability)
                )

                st.caption(
                    f"{probability.item() * 100:.2f}%"
                )

            # Technical information
            with st.expander(
                "Technical Details"
            ):

                st.write(
                    "Original sample rate:",
                    f"{sample_rate} Hz"
                )

                st.write(
                    "Original waveform shape:",
                    tuple(waveform.shape)
                )

                st.write(
                    "Processed waveform shape:",
                    tuple(processed_waveform.shape)
                )

                st.write(
                    "Log-Mel spectrogram shape:",
                    tuple(log_mel.shape)
                )

                st.write(
                    "Model input shape:",
                    tuple(model_input.shape)
                )

                st.write(
                    "Device:",
                    str(DEVICE)
                )


    # Clean temporary file
    if os.path.exists(temp_path):

        os.remove(temp_path)
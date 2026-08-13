import os
import sys

import streamlit as st
import torch
import torchaudio
import pandas as pd


# PATH SETUP
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing import preprocess_audio



# PAGE CONFIGURATION
st.set_page_config(
    page_title = "Environmental Sound Classifier",
    page_icon = "🔊",
    layout = "wide",
    initial_sidebar_state = "expanded"
)


# ADDING CUSTOM CSS
st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    .prediction-card {
        background-color: #f5f7fa;
        border: 1px solid #d9dee7;
        border-radius: 14px;
        padding: 24px;
        margin: 10px 0 20px 0;
        text-align: center;
    }

    .prediction-label {
        color: #555555;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .prediction-value {
        color: #111111;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .prediction-confidence {
        color: #444444;
        font-size: 18px;
        font-weight: 500;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 1rem;
        margin-bottom: 0.7rem;
    }

    .info-card {
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background: #fafafa;
    }

    </style>
    """,
    unsafe_allow_html = True
)


# CLASS LABELS
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


# MODEL CONFIGURATION
MODEL_NAME = "ResNet18"

MODEL_PARAMETERS = "11,172,810"

TARGET_SAMPLE_RATE = 16000
TARGET_DURATION = 4
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512


# MODEL ARCHITECTURE
def create_model():

    from torchvision.models import resnet18

    model = resnet18(weights = None)

    # Same architecture used during training
    model.conv1 = torch.nn.Conv2d(
        in_channels = 1,
        out_channels = 64,
        kernel_size = 3,
        stride = 1,
        padding = 1,
        bias = False
    )

    model.maxpool = torch.nn.Identity()

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        10
    )

    return model


# LOADING THE MODEL
@st.cache_resource
def load_model():

    model = create_model()

    checkpoint_path = os.path.join(
        PROJECT_ROOT,
        "models",
        "best_resnet18.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location = torch.device("cpu")
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.eval()

    return model


# SETTING TITLE AND SUBTITLE
st.markdown(
    '<div class="main-title">🔊 Environmental Sound Classifier</div>',
    unsafe_allow_html = True
)

st.markdown(
    """
    <div class="subtitle">
    AI-powered environmental sound classification using a
    ResNet18 convolutional neural network trained on UrbanSound8K.
    </div>
    """,
    unsafe_allow_html = True
)


# SIDEBAR
with st.sidebar:

    st.header("Model Information")

    st.write(f"**Architecture:** {MODEL_NAME}")
    st.write(f"**Parameters:** {MODEL_PARAMETERS}")
    st.write("**Dataset:** UrbanSound8K")
    st.write("**Classes:** 10")
    st.write("**Input representation:** Log-Mel spectrogram")
    st.write(f"**Sample rate:** {TARGET_SAMPLE_RATE:,} Hz")
    st.write(f"**Audio duration:** {TARGET_DURATION} seconds")

    st.divider()

    st.caption(
        "The model classifies environmental sounds "
        "into one of ten UrbanSound8K categories."
    )


# MAIN TABS
tab_classify, tab_pipeline, tab_about = st.tabs(
    [
        "🎧 Classify Audio",
        "⚙️ Preprocessing Pipeline",
        "📘 About the Model"
    ]
)


# TAB 1 — CLASSIFICATION
with tab_classify:

    st.markdown(
        '<div class="section-title">Upload an audio recording</div>',
        unsafe_allow_html = True
    )

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type = [
            "wav",
            "mp3",
            "ogg",
            "flac",
            "m4a"
        ],
        help = "Upload an environmental sound recording."
    )


    if uploaded_file is None:
        st.info("Upload an audio file to begin classification.")
    else:
        # audio player
        st.audio(uploaded_file)

        st.write(f"**File:** `{uploaded_file.name}`")

        st.divider()


        # classification button
        classify = st.button(
            "🔍 Classify Audio",
            type = "primary",
            use_container_width = True
        )

        if classify:
            temp_audio_path = os.path.join(
                PROJECT_ROOT,
                "_temp_uploaded_audio.wav"
            )

            try:
                # saving uploaded audio to a temporary file
                with open(temp_audio_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # loading audio
                waveform, sample_rate = torchaudio.load(temp_audio_path)

                # defining and setting audio information metrics
                st.subheader("Audio Information")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Sample Rate",f"{sample_rate:,} Hz")
                with col2:
                    st.metric("Channels", waveform.shape[0])
                with col3:
                    duration = (waveform.shape[1] / sample_rate)

                    st.metric("Duration", f"{duration:.2f} sec")

                # preprocessing
                waveform, log_mel_spec = preprocess_audio(
                    waveform,
                    sample_rate
                )

                # model input
                model_input = log_mel_spec.unsqueeze(0)

                # loading the model 
                model = load_model()

                # inference
                with torch.no_grad():
                    logits = model(model_input)

                    probabilities = torch.softmax(logits, dim=1)[0]

                # prediction
                predicted_index = torch.argmax(probabilities).item()

                predicted_class = CLASS_NAMES[predicted_index]

                confidence = probabilities[predicted_index].item()

                # results
                st.divider()

                st.subheader("Classification Result")

                result_col1, result_col2 = st.columns([1.4, 1])

                # prediction card
                with result_col1:

                    st.markdown(
                        f"""
                        <div class="prediction-card">
                            <div class="prediction-label">Predicted Sound</div>
                            <div class="prediction-value">{predicted_class.replace("_", " ").title()}</div>
                            <div class="prediction-confidence">Confidence: {confidence:.2%}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                # confidence
                with result_col2:
                    st.metric("Confidence", f"{confidence * 100:.2f}%")

                    st.progress(confidence)


                # TOP 3
                st.subheader("Top 3 Predictions")

                top_values, top_indices = torch.topk(probabilities, k = 3)

                top_col1, top_col2, top_col3 = st.columns(3)

                columns = [
                    top_col1,
                    top_col2,
                    top_col3
                ]

                for rank, (value, index) in enumerate(zip(top_values, top_indices), start = 1):
                    class_name = CLASS_NAMES[index.item()]

                    with columns[rank - 1]:
                        st.metric(
                            f"#{rank}",
                            class_name.replace(
                                "_",
                                " "
                            ).title(),
                            f"{value.item() * 100:.2f}%"
                        )

                # probability distribution
                st.subheader("Class Probability Distribution")

                sorted_probabilities = sorted(
                    zip(CLASS_NAMES, probabilities.tolist()),
                    key = lambda x: x[1],
                    reverse = True
                )

                for class_name, probability in sorted_probabilities:
                    display_name = class_name.replace("_", " ").title()

                    st.write(f"**{display_name}** — {probability:.2%}")

                    st.progress(float(probability))


                # technical details
                with st.expander("Technical inference details"):
                    st.write(
                        "**Processed waveform:** "
                        f"`{tuple(waveform.shape)}`"
                    )

                    st.write(
                        "**Log-Mel spectrogram:** "
                        f"`{tuple(log_mel_spec.shape)}`"
                    )

                    st.write(
                        "**Model input:** "
                        f"`{tuple(model_input.shape)}`"
                    )

                    st.write("**Model:** ResNet18")

                    st.write("**Device:** CPU")


            except Exception as e:
                st.error("An error occurred during classification.")

                st.exception(e)
            finally:
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)


# TAB 2 — preprocessing pipeline
with tab_pipeline:
    st.header("Audio Preprocessing Pipeline")

    st.write(
        """
        The uploaded audio is transformed into the same
        representation used during model training.
        This ensures that the distribution of inputs during
        inference matches the distribution seen during training.
        """
    )

    st.markdown(
        """
        ### 1. Stereo → Mono

        Multi-channel audio is converted into a single
        mono waveform.

        **Output:** 1 audio channel
        """
    )

    st.markdown(
        """
        ### 2. Resampling

        Audio is resampled to:

        **16,000 Hz**
        """
    )

    st.markdown(
        """
        ### 3. Duration Normalization

        Each recording is standardized to:

        **4 seconds → 64,000 samples**

        Shorter recordings are padded and longer recordings
        are truncated.
        """
    )

    st.markdown(
        """
        ### 4. Waveform Normalization

        The waveform amplitude is normalized to prevent
        excessively large amplitude values.
        """
    )


    st.markdown(
        """
        ### 5. Mel-Spectrogram

        The normalized waveform is converted into a
        128-bin Mel-spectrogram.

        - FFT size: **1024**
        - Hop length: **512**
        - Mel bins: **128**
        """
    )


    st.markdown(
        """
        ### 6. Log-Mel Transformation

        A logarithmic transformation is applied to the
        Mel-spectrogram before it is passed to ResNet18.
        """
    )


    st.divider()

    st.success("Final model input: [1, 1, 128, 126]")


# TAB 3 — about the model
with tab_about:

    st.header("About the Model")

    st.write(
        """
        This application demonstrates an environmental sound
        classification system developed using the UrbanSound8K
        dataset.
        """
    )


    st.subheader("Model Architecture")

    st.write(
        """
        The deployed classifier is a modified ResNet18
        convolutional neural network.

        Instead of processing raw audio directly, the model
        receives a log-Mel spectrogram representation of the
        audio.
        """
    )


    st.subheader("Why a Log-Mel Spectrogram?")

    st.write(
        """
        Audio is a time-domain signal, but environmental sounds
        contain important frequency characteristics.

        The Mel-spectrogram provides a time-frequency
        representation that allows the convolutional network
        to learn patterns associated with different sounds.
        """
    )


    st.subheader("Training Dataset")

    st.write(
        """
        **UrbanSound8K**

        The dataset contains environmental recordings belonging
        to ten sound categories.
        """
    )

    st.subheader("Sound Categories")

    for class_name in CLASS_NAMES:
        st.write(f"• {class_name.replace('_', ' ').title()}")


    st.subheader("Deployment Configuration")

    st.write(
        f"""
        - Architecture: **{MODEL_NAME}**
        - Parameters: **{MODEL_PARAMETERS}**
        - Sample rate: **{TARGET_SAMPLE_RATE:,} Hz**
        - Duration: **{TARGET_DURATION} seconds**
        - Mel bins: **{N_MELS}**
        - FFT size: **{N_FFT}**
        - Hop length: **{HOP_LENGTH}**
        - Input shape: **[1, 1, 128, 126]**
        """
    )


    st.info(
        """
        This demo is intended as a practical deployment
        demonstration of the trained environmental sound
        classification model.
        """
    )


# footer
st.divider()

st.caption(
    "Environmental Sound Classification • UrbanSound8K • ResNet18"
)
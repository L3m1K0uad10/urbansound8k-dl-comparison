import os
import sys

import streamlit as st
import torch
import torchaudio


# ============================================================
# PATH SETUP
# ============================================================
# Adding the project root to Python's import path.
# This allows us to import files from the app directory
# regardless of how Streamlit launches this script.

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# importting preprocessing from the same app directory.
from preprocessing import preprocess_audio



# configuration
st.set_page_config(
    page_title = "Environmental Sound Classifier",
    page_icon = "🔊",
    layout = "centered"
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


# model definition
def create_model():
    """
    Create the same ResNet18 architecture used during training.
    """

    from torchvision.models import resnet18

    model = resnet18(weights = None)

    # matching the architecture used during training
    model.conv1 = torch.nn.Conv2d(
        in_channels = 1,
        out_channels = 64,
        kernel_size = 3,
        stride = 1,
        padding = 1,
        bias = False
    )

    # removing the original max pooling layer
    model.maxpool = torch.nn.Identity()

    # 10 UrbanSound8K classes
    model.fc = torch.nn.Linear(
        model.fc.in_features,
        10
    )

    return model


# load model
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

    # Handling both possibilities:
    #
    # 1. checkpoint is directly model.state_dict()
    # 2. checkpoint is a dictionary containing state_dict

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


# application title and description
st.title("🔊 Environmental Sound Classifier")

st.write(
    """
    Upload an environmental audio recording and the trained
    ResNet18 model will classify it into one of the 10
    UrbanSound8K sound categories.
    """
)


st.divider()


# audio upload widget
uploaded_file = st.file_uploader(
    "Upload an audio file",
    type = [
        "wav",
        "mp3",
        "ogg",
        "flac",
        "m4a"
    ]
)


# processing audio
if uploaded_file is not None:

    st.subheader("Uploaded Audio")

    # playing audio directly in Streamlit.
    st.audio(uploaded_file)

    st.write(f"**File:** {uploaded_file.name}")

    st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")

    # running classification
    if st.button("Classify Audio", type = "primary"):
        try:
            # saving uploaded file temporarily
            temp_audio_path = os.path.join(PROJECT_ROOT, "_temp_uploaded_audio.wav")

            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # loading audio
            waveform, sample_rate = torchaudio.load(temp_audio_path)

            # displaying original information
            st.subheader("Audio Information")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Sample Rate", f"{sample_rate} Hz")
            with col2:
                st.metric("Channels", waveform.shape[0])


            # preprocessing audio
            waveform, log_mel_spec = preprocess_audio(waveform, sample_rate)


            st.subheader("Preprocessing")

            st.write(
                f"Processed waveform shape: "
                f"`{tuple(waveform.shape)}`"
            )

            st.write(
                f"Log-Mel spectrogram shape: "
                f"`{tuple(log_mel_spec.shape)}`"
            )


            # ------------------------------------------------
            # Preparing model input
            # ------------------------------------------------
            #
            # Current shape:
            #
            # [1, 128, 126]
            #
            # Model expects:
            #
            # [batch, channel, 128, 126]
            #
            # Therefore:
            #
            # [1, 1, 128, 126]

            model_input = log_mel_spec.unsqueeze(0)


            st.write(
                f"Model input shape: "
                f"`{tuple(model_input.shape)}`"
            )


            # loading model
            model = load_model()


            # inferencing
            with torch.no_grad():
                logits = model(model_input)

                probabilities = torch.softmax(logits, dim = 1)[0]

            # getting prediction
            predicted_index = torch.argmax(probabilities).item()

            predicted_class = CLASS_NAMES[predicted_index]

            confidence = probabilities[predicted_index].item()


            # results
            st.divider()

            st.subheader("Classification Result")


            st.success(f"Prediction: **{predicted_class.replace('_', ' ').title()}**")


            st.metric("Confidence", f"{confidence * 100:.2f}%")


            # probability distribution
            st.subheader("Class Probabilities")


            probability_data = {
                CLASS_NAMES[i].replace(
                    "_",
                    " "
                ).title(): float(
                    probabilities[i]
                )
                for i in range(
                    len(CLASS_NAMES)
                )
            }


            # sorting from highest to lowest.
            probability_data = dict(
                sorted(
                    probability_data.items(),
                    key = lambda item: item[1],
                    reverse = True
                )
            )


            # displaying as percentages.
            for class_name, probability in probability_data.items():

                st.write(
                    f"**{class_name}** — "
                    f"{probability * 100:.2f}%"
                )

                st.progress(probability)


            # Top 3 predictions
            st.subheader("Top 3 Predictions")

            top_values, top_indices = torch.topk(probabilities, k = 3)


            for rank, (value, index) in enumerate(
                zip(
                    top_values,
                    top_indices
                ),
                start=1
            ):

                class_name = CLASS_NAMES[index.item()]

                st.write(
                    f"**{rank}. "
                    f"{class_name.replace('_', ' ').title()}** — "
                    f"{value.item() * 100:.2f}%"
                )


            # Cleanup
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)


        except Exception as e:
            st.error(
                "An error occurred while processing "
                "the audio file."
            )

            st.exception(e)
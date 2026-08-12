"""  
the Mel transform is:

mel_transform = MelSpectrogram(
    sample_rate = 16000,
    n_fft = 1024,
    hop_length = 512,
    n_mels = 128
)

This is very important. 

The Streamlit application cannot suddenly use:

n_mels = 64

or:

sample_rate = 22050

because then the input distribution is different from what the model learned.
"""
import torch
import torch.nn.functional as F

import torchaudio
from torchaudio.transforms import (
    MelSpectrogram,
    Resample
)


TARGET_SAMPLE_RATE = 16000
TARGET_DURATION = 4
TARGET_NUM_SAMPLES = (
    TARGET_SAMPLE_RATE * TARGET_DURATION
)


mel_transform = MelSpectrogram(
    sample_rate = TARGET_SAMPLE_RATE,
    n_fft = 1024,
    hop_length = 512,
    n_mels = 128
)


def preprocess_waveform(
    waveform,
    sample_rate
):
    """  
    Preprocess the waveform to match the model's expected input format.
    one small improvement over the notebook
        if max_value > 0:
    That's because completely silent audio has:
        waveform.abs().max() == 0
    and we don't want a division-by-zero situation.
    """

    # Stereo → mono
    if waveform.shape[0] > 1:

        waveform = waveform.mean(
            dim=0,
            keepdim=True
        )

    # Resample
    if sample_rate != TARGET_SAMPLE_RATE:

        resampler = Resample(
            sample_rate,
            TARGET_SAMPLE_RATE
        )

        waveform = resampler(waveform)

    # Pad / truncate
    if waveform.shape[1] < TARGET_NUM_SAMPLES:

        padding = (
            TARGET_NUM_SAMPLES
            - waveform.shape[1]
        )

        waveform = F.pad(
            waveform,
            (0, padding)
        )

    else:

        waveform = waveform[
            :, :TARGET_NUM_SAMPLES
        ]

    # Normalize
    max_value = waveform.abs().max()

    if max_value > 0:
        waveform = waveform / max_value

    return waveform



def waveform_to_log_mel(
    waveform
):
    """
    Convert the waveform to a log-mel spectrogram.  
    """

    mel_spec = mel_transform(
        waveform
    )

    log_mel_spec = torch.log(
        mel_spec + 1e-9
    )

    return log_mel_spec



def preprocess_audio(
    waveform,
    sample_rate
):
    """ 
    Preprocess the audio waveform and convert it to a log-mel spectrogram. 
    """

    waveform = preprocess_waveform(
        waveform,
        sample_rate
    )

    log_mel_spec = waveform_to_log_mel(
        waveform
    )

    return waveform, log_mel_spec
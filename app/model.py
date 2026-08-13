import torch
import torch.nn as nn
from torchvision import models


NUM_CLASSES = 10

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
    "street_music",
]


def create_model():
    """
    Create the ResNet18 architecture used during training.
    """

    model = models.resnet18(weights=None)

    # Match the architecture used during training
    model.conv1 = nn.Conv2d(
        1,
        64,
        kernel_size = 3,
        stride = 1,
        padding = 1,
        bias = False
    )

    # Remove max pooling
    model.maxpool = nn.Identity()

    # Replace classifier
    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    return model


def load_model(checkpoint_path, device):
    """
    Create the model and load the trained checkpoint.
    """

    model = create_model()

    checkpoint = torch.load(
        checkpoint_path,
        map_location = device
    )

    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model
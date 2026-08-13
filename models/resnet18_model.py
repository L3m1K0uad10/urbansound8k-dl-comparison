import torch.nn as nn
from torchvision.models import resnet18


NUM_CLASSES = 10


def create_resnet18():
    model = resnet18(weights = None)

    # Adapt ResNet18 for Log-Mel spectrogram input
    model.conv1 = nn.Conv2d(
        in_channels = 1,
        out_channels = 64,
        kernel_size = 3,
        stride = 1,
        padding = 1,
        bias = False
    )

    # Remove the original ImageNet max pooling
    model.maxpool = nn.Identity()

    # Replace classifier
    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    return model
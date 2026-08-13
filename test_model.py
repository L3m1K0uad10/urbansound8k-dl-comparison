import torch

from models.resnet18_model import create_resnet18


DEVICE = torch.device("cpu")


model = create_resnet18()

checkpoint = torch.load(
    "models/best_resnet18.pth",
    map_location = DEVICE
)

model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()


print("Model loaded successfully!")
print("Device:", DEVICE)
print("Number of parameters:", sum(p.numel() for p in model.parameters()))
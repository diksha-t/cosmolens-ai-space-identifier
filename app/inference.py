import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# class names
CLASS_NAMES = [
    "asteroid",
    "black_hole",
    "earth",
    "galaxy",
    "jupiter",
    "mars",
    "mercury",
    "neptune",
    "pluto",
    "saturn",
    "uranus",
    "venus"
]

# CNN architecture
class SimpleCNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        return x

# load model
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = SimpleCNN(
    num_classes=len(CLASS_NAMES)
)

model.load_state_dict(
    torch.load(
        "../models/simple_cnn.pth",
        map_location=device
    )
)

model.to(device)

model.eval()

# prediction function
def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image)

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

        top5_prob, top5_idx = torch.topk(
            probabilities,
            k=5
        )

    result = {

        "prediction": CLASS_NAMES[prediction.item()],

        "confidence": confidence.item(),

        "top5": [

            (
                CLASS_NAMES[idx.item()],
                prob.item()
            )

            for prob, idx in zip(
                top5_prob[0],
                top5_idx[0]
            )
        ]
    }

    return result
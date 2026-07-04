import torch
import torch.nn as nn
from torchvision import models


class ImageClassifier:

    def __init__(self, num_classes):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load pretrained ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze pretrained layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace final layer
        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

        self.model = self.model.to(self.device)

    def get_model(self):
        return self.model

    def get_device(self):
        return self.device


if __name__ == "__main__":

    model = ImageClassifier(num_classes=10)

    print(model.get_model())

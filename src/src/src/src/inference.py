import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# -----------------------------
# Load Model
# -----------------------------
def load_model(model_path, num_classes):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = models.resnet18(
        weights=None
    )

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model, device


# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Predict Image
# -----------------------------
def predict(image_path, model_path, class_names):

    model, device = load_model(
        model_path,
        len(class_names)
    )

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            1
        )

    predicted_class = class_names[
        prediction.item()
    ]

    print("=" * 50)
    print("Prediction Result")
    print("=" * 50)

    print(f"Predicted Class : {predicted_class}")

    print(
        f"Confidence      : {confidence.item()*100:.2f}%"
    )

    return predicted_class


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path to image"
    )

    parser.add_argument(
        "--model",
        default="../models/best_model.pth",
        help="Path to trained model"
    )

    args = parser.parse_args()

    # Replace these with your dataset class names
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]

    predict(
        args.image,
        args.model,
        class_names
    )

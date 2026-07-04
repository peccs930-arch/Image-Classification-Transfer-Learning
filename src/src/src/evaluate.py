import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


class ModelEvaluator:

    def __init__(self, model, test_loader, device, class_names):

        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names

    def evaluate(self):

        self.model.eval()

        predictions = []
        labels = []

        with torch.no_grad():

            for images, targets in self.test_loader:

                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)

                _, predicted = torch.max(outputs, 1)

                predictions.extend(predicted.cpu().numpy())
                labels.extend(targets.cpu().numpy())

        accuracy = accuracy_score(labels, predictions)

        precision = precision_score(
            labels,
            predictions,
            average="weighted"
        )

        recall = recall_score(
            labels,
            predictions,
            average="weighted"
        )

        f1 = f1_score(
            labels,
            predictions,
            average="weighted"
        )

        print("=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print("\nClassification Report\n")

        print(
            classification_report(
                labels,
                predictions,
                target_names=self.class_names
            )
        )

        return labels, predictions

    def plot_confusion_matrix(self, labels, predictions):

        cm = confusion_matrix(labels, predictions)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.class_names
        )

        fig, ax = plt.subplots(figsize=(8, 8))

        disp.plot(
            cmap="Blues",
            xticks_rotation=45,
            ax=ax
        )

        plt.title("Confusion Matrix")

        plt.tight_layout()

        plt.savefig(
            "../images/confusion_matrix.png",
            dpi=300
        )

        plt.show()

    def plot_training_history(self, history):

        epochs = range(
            1,
            len(history["train_loss"]) + 1
        )

        # Loss Curve
        plt.figure(figsize=(8,5))

        plt.plot(
            epochs,
            history["train_loss"],
            label="Training Loss"
        )

        plt.plot(
            epochs,
            history["val_loss"],
            label="Validation Loss"
        )

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.title("Training Loss Curve")

        plt.legend()

        plt.grid(True)

        plt.savefig(
            "../images/training_curve.png",
            dpi=300
        )

        plt.show()

        # Accuracy Curve
        plt.figure(figsize=(8,5))

        plt.plot(
            epochs,
            history["train_accuracy"],
            label="Training Accuracy"
        )

        plt.plot(
            epochs,
            history["val_accuracy"],
            label="Validation Accuracy"
        )

        plt.xlabel("Epoch")

        plt.ylabel("Accuracy")

        plt.title("Accuracy Curve")

        plt.legend()

        plt.grid(True)

        plt.savefig(
            "../images/accuracy_curve.png",
            dpi=300
        )

        plt.show()


if __name__ == "__main__":

    print("Evaluation module loaded successfully.")

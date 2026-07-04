import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class ImageDataset:

    def __init__(
        self,
        train_dir,
        valid_dir,
        test_dir,
        batch_size=32
    ):

        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.batch_size = batch_size

        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    def get_dataloaders(self):

        train_dataset = datasets.ImageFolder(
            self.train_dir,
            transform=self.train_transform
        )

        valid_dataset = datasets.ImageFolder(
            self.valid_dir,
            transform=self.test_transform
        )

        test_dataset = datasets.ImageFolder(
            self.test_dir,
            transform=self.test_transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

        return (
            train_loader,
            valid_loader,
            test_loader,
            train_dataset.classes
        )


if __name__ == "__main__":

    dataset = ImageDataset(
        train_dir="../dataset/train",
        valid_dir="../dataset/validation",
        test_dir="../dataset/test"
    )

    train_loader, valid_loader, test_loader, classes = dataset.get_dataloaders()

    print("Classes:", classes)
    print("Training Images:", len(train_loader.dataset))
    print("Validation Images:", len(valid_loader.dataset))
    print("Testing Images:", len(test_loader.dataset))

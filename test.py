import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MatteaNet


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="./checkpoints/best.pth")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-dir", type=str, default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_set = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = MatteaNet().to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        predicted = outputs.argmax(dim=1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = 100.0 * correct / total
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Recorded best accuracy: {checkpoint.get('best_acc', 'unknown')}")
    print(f"Test accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()

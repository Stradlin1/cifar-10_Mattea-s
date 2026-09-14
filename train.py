import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import MatteaNet


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10
VAL_PER_CLASS = 500


def build_stratified_indices(targets):
    """Create a fixed stratified split without random numbers or a seed.

    For each CIFAR-10 class, the first 500 samples encountered are assigned to
    validation and the remaining 4500 samples are assigned to training.
    """
    val_counts = [0] * NUM_CLASSES
    train_indices = []
    val_indices = []

    for index, label in enumerate(targets):
        if val_counts[label] < VAL_PER_CLASS:
            val_indices.append(index)
            val_counts[label] += 1
        else:
            train_indices.append(index)

    expected_val = NUM_CLASSES * VAL_PER_CLASS
    if len(val_indices) != expected_val:
        raise RuntimeError(
            f"Expected {expected_val} validation samples, got {len(val_indices)}"
        )

    return train_indices, val_indices


def build_loaders(data_dir: str, batch_size: int, workers: int):
    """Build fixed train/validation loaders from CIFAR-10's training set.

    The official CIFAR-10 test set is intentionally not loaded here. It is used
    only by test.py after model selection is finished.
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Two views of the same 50,000 training images are used so that training
    # samples can use augmentation while validation samples stay deterministic.
    train_full_aug = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    train_full_eval = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=eval_transform,
    )

    train_indices, val_indices = build_stratified_indices(train_full_aug.targets)

    train_set = Subset(train_full_aug, train_indices)
    val_set = Subset(train_full_eval, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )

    return train_loader, val_loader


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = build_loaders(
        args.data_dir,
        args.batch_size,
        args.workers,
    )
    print(
        f"Dataset split: train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}"
    )
    print(
        "Split rule: fixed stratified split, "
        f"{VAL_PER_CLASS} validation samples per class and "
        f"{5000 - VAL_PER_CLASS} training samples per class."
    )
    print("Official CIFAR-10 test set is reserved for test.py only.")

    model = MatteaNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.2f}% | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                    "split": "fixed_stratified",
                    "val_per_class": VAL_PER_CLASS,
                },
                save_dir / "best.pth",
            )
            print(f"Saved new best checkpoint: val_acc={best_val_acc:.2f}%")

    print(f"Training finished. Best validation accuracy: {best_val_acc:.2f}%")
    print("Run `python test.py` once for the final official test accuracy.")


if __name__ == "__main__":
    main()

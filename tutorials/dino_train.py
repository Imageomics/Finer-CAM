import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a linear classifier on top of a frozen DINOv2 backbone."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Dataset root. Supports train/valid, train/val, or a single ImageFolder root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs/dinov2_classifier_cli"),
        help="Directory for checkpoints and class_names.json.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="CUDA device index to use.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--model-repo",
        type=str,
        default="facebookresearch/dinov2",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="dinov2_vitb14",
    )
    return parser.parse_args()


def ensure_supported_python():
    if sys.version_info <= (3, 10):
        raise RuntimeError(
            "This script requires Python > 3.10 because the official DINOv2 "
            "torch.hub code uses modern type hint syntax."
        )


def get_device(gpu_id):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for training. Install a CUDA-enabled PyTorch build "
            "and run this script in a GPU-enabled environment."
        )
    if gpu_id >= torch.cuda.device_count():
        raise ValueError(
            f"Requested --gpu-id={gpu_id}, but only {torch.cuda.device_count()} "
            "CUDA device(s) are visible."
        )
    return torch.device(f"cuda:{gpu_id}")


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224, interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    valid_transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, valid_transform


def find_split_files(data_root):
    candidates = [data_root, data_root.parent]
    for root in candidates:
        images_txt = root / "images.txt"
        split_txt = root / "train_test_split.txt"
        if images_txt.is_file() and split_txt.is_file():
            return images_txt, split_txt
    return None, None


def load_metadata_split_indices(data_root):
    images_txt, split_txt = find_split_files(data_root)
    if images_txt is None or split_txt is None:
        return None, None

    image_id_to_path = {}
    with open(images_txt, "r") as f:
        for line in f:
            image_id, rel_path = line.strip().split(" ", 1)
            image_id_to_path[int(image_id)] = rel_path

    dataset = datasets.ImageFolder(root=str(data_root))
    dataset_index_by_relpath = {
        os.path.relpath(path, str(data_root)): idx
        for idx, (path, _) in enumerate(dataset.samples)
    }

    train_indices = []
    valid_indices = []
    with open(split_txt, "r") as f:
        for line in f:
            image_id, is_train = line.strip().split()
            rel_path = image_id_to_path[int(image_id)]
            dataset_idx = dataset_index_by_relpath.get(rel_path)
            if dataset_idx is None:
                continue
            if int(is_train) == 1:
                train_indices.append(dataset_idx)
            else:
                valid_indices.append(dataset_idx)

    return train_indices, valid_indices


def build_datasets(data_root, train_transform, valid_transform, random_seed):
    train_root = data_root / "train"
    valid_root = data_root / "valid"
    val_root = data_root / "val"

    if train_root.is_dir() and valid_root.is_dir():
        train_dataset = datasets.ImageFolder(
            root=str(train_root), transform=train_transform
        )
        valid_dataset = datasets.ImageFolder(
            root=str(valid_root), transform=valid_transform
        )
        split_mode = "explicit train/valid folders"
        class_names = train_dataset.classes
        return train_dataset, valid_dataset, class_names, split_mode

    if train_root.is_dir() and val_root.is_dir():
        train_dataset = datasets.ImageFolder(
            root=str(train_root), transform=train_transform
        )
        valid_dataset = datasets.ImageFolder(
            root=str(val_root), transform=valid_transform
        )
        split_mode = "explicit train/val folders"
        class_names = train_dataset.classes
        return train_dataset, valid_dataset, class_names, split_mode

    base_dataset = datasets.ImageFolder(root=str(data_root))
    train_indices, valid_indices = load_metadata_split_indices(data_root)

    if train_indices and valid_indices:
        split_mode = "metadata-defined split"
    else:
        num_samples = len(base_dataset)
        num_valid = max(1, int(0.2 * num_samples))
        generator = torch.Generator().manual_seed(random_seed)
        indices = torch.randperm(num_samples, generator=generator).tolist()
        valid_indices = indices[:num_valid]
        train_indices = indices[num_valid:]
        split_mode = "deterministic random 80/20 split"

    train_dataset = Subset(
        datasets.ImageFolder(root=str(data_root), transform=train_transform),
        train_indices,
    )
    valid_dataset = Subset(
        datasets.ImageFolder(root=str(data_root), transform=valid_transform),
        valid_indices,
    )
    class_names = base_dataset.classes
    return train_dataset, valid_dataset, class_names, split_mode


def save_class_names(output_dir, class_names):
    class_path = output_dir / "class_names.json"
    class_path.write_text(json.dumps(class_names, indent=2))
    return class_path


def extract_features(backbone, images):
    patch_tokens = backbone.forward_features(images)["x_norm_patchtokens"]
    return patch_tokens.mean(dim=1)


def train_one_epoch(backbone, classifier, loader, criterion, optimizer, device):
    classifier.train()
    running_loss = 0.0
    total = 0
    correct = 0

    progress = tqdm(loader, desc="Training", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.no_grad():
            features = extract_features(backbone, images)

        logits = classifier(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100.0 * correct / total:.2f}%",
        )

    return {
        "loss": running_loss / len(loader.dataset),
        "accuracy": 100.0 * correct / total,
    }


def evaluate(backbone, classifier, loader, device):
    classifier.eval()
    total = 0
    correct = 0

    progress = tqdm(loader, desc="Validating", leave=False)
    with torch.no_grad():
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            features = extract_features(backbone, images)
            logits = classifier(features)
            preds = logits.argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()
            progress.set_postfix(acc=f"{100.0 * correct / total:.2f}%")

    return {
        "accuracy": 100.0 * correct / total,
    }


def main():
    ensure_supported_python()
    args = parse_args()

    data_root = args.data_root.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"--data-root does not exist: {data_root}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.gpu_id)
    train_transform, valid_transform = build_transforms()

    train_dataset, valid_dataset, class_names, split_mode = build_datasets(
        data_root,
        train_transform,
        valid_transform,
        random_seed=args.random_seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    class_names_path = save_class_names(output_dir, class_names)

    print(f"Device: {device}")
    print(f"GPU name: {torch.cuda.get_device_name(device)}")
    print(f"Data root: {data_root}")
    print(f"Split mode: {split_mode}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    print(f"Num classes: {len(class_names)}")
    print(f"Class names saved to: {class_names_path}")

    backbone = torch.hub.load(
        args.model_repo, args.model_name, pretrained=True
    ).to(device)
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    classifier = nn.Linear(backbone.embed_dim, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr)

    best_accuracy = 0.0
    best_path = output_dir / "best_classifier.pth"
    final_path = output_dir / "final_classifier.pth"

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            backbone, classifier, train_loader, criterion, optimizer, device
        )
        valid_metrics = evaluate(backbone, classifier, valid_loader, device)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.2f}% | "
            f"valid_acc={valid_metrics['accuracy']:.2f}%"
        )

        if valid_metrics["accuracy"] > best_accuracy:
            best_accuracy = valid_metrics["accuracy"]
            torch.save(classifier.state_dict(), best_path)
            print(f"Saved new best classifier to: {best_path}")

    torch.save(classifier.state_dict(), final_path)
    print(f"Saved final classifier to: {final_path}")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()

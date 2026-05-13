"""Train the compact lip-reading word classifier."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from model import LipReadingModel
from utils import VOCABULARY, labels_to_maps, normalize_clip


class LipClipDataset(Dataset):
    """Dataset of .npy clips saved by collect_data.py."""

    def __init__(self, data_dir: str | Path, frames_per_clip: int, img_size: int, labels: list[str]) -> None:
        self.data_dir = Path(data_dir)
        self.frames_per_clip = frames_per_clip
        self.img_size = img_size
        self.labels = labels
        self.label_to_index, _ = labels_to_maps(labels)
        self.samples: list[tuple[Path, int]] = []

        for label in self.labels:
            label_dir = self.data_dir / label
            if not label_dir.exists():
                continue
            for path in sorted(label_dir.glob("*.npy")):
                self.samples.append((path, self.label_to_index[label]))

        if not self.samples:
            raise RuntimeError(
                f"No .npy clips found in {self.data_dir}. "
                "Run collect_data.py first, for example: python3 collect_data.py --label hello"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        clip = np.load(path)

        if clip.shape[0] != self.frames_per_clip:
            raise ValueError(
                f"{path} has {clip.shape[0]} frames, expected {self.frames_per_clip}. "
                "Use matching --frames_per_clip for collection and training."
            )
        if clip.shape[1] != self.img_size or clip.shape[2] != self.img_size:
            raise ValueError(
                f"{path} has image size {clip.shape[1:3]}, expected {(self.img_size, self.img_size)}."
            )

        clip = normalize_clip(clip)
        return torch.from_numpy(clip), torch.tensor(label, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lip-reading classifier.")
    parser.add_argument("--data_dir", default="data/raw", help="Directory containing label subfolders.")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--frames_per_clip", type=int, default=30, help="Frames per clip.")
    parser.add_argument("--img_size", type=int, default=96, help="Mouth crop size.")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation fraction.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Training device. Auto uses CUDA when available, otherwise CPU. MPS is opt-in.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels to train. Defaults to all folders in data_dir, or project vocabulary if empty.",
    )
    return parser.parse_args()


def get_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    for clips, labels in tqdm(loader, leave=False):
        clips = clips.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(clips)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    args = parse_args()
    random.seed(42)
    torch.manual_seed(42)

    data_dir = Path(args.data_dir)
    if args.labels:
        labels = args.labels
    else:
        folder_labels = sorted(path.name for path in data_dir.iterdir() if path.is_dir()) if data_dir.exists() else []
        labels = folder_labels or VOCABULARY

    dataset = LipClipDataset(data_dir, args.frames_per_clip, args.img_size, labels)
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise RuntimeError("Need at least two clips to create train/validation split.")

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = get_device(args.device)
    print(f"Device: {device}")
    if torch.backends.mps.is_available() and device.type == "cpu":
        print("MPS available but CPU selected for training stability with GRU backward.")
    print(f"Samples: train={len(train_dataset)}, val={len(val_dataset)}")

    model = LipReadingModel(num_classes=len(labels), img_size=args.img_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    checkpoint_path = Path("models/lip_reader.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocabulary": labels,
                    "frames_per_clip": args.frames_per_clip,
                    "img_size": args.img_size,
                    "best_val_acc": best_val_acc,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

    print(f"Best validation accuracy: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()

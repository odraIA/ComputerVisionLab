import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MASK_TOKENS = ("segmentation", "mask", "groundtruth", "ground_truth")
SMOOTH = 1e-6


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Using CPU instead.")
        return torch.device("cpu")
    return torch.device(device_arg)


def parse_image_size(value):
    value = str(value).lower().replace("x", ",")
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) == 1:
        size = int(parts[0])
        return size, size
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    raise argparse.ArgumentTypeError("Use an integer like 256 or a size like 256x256.")


def is_mask_file(path):
    stem = path.stem.lower()
    if any(stem.endswith(f"_{token}") or stem.endswith(f"-{token}") for token in MASK_TOKENS):
        return True

    mask_folder_names = {
        "ground_truth",
        "groundtruth",
        "mask",
        "masks",
        "segmentation",
        "segmentations",
        "segmentation_masks",
    }
    return any(part.lower() in mask_folder_names for part in path.parent.parts)


def normalized_pair_key(path):
    stem = path.stem.lower()
    suffixes = [
        "_segmentation",
        "-segmentation",
        " segmentation",
        "_mask",
        "-mask",
        " mask",
        "_groundtruth",
        "-groundtruth",
        " groundtruth",
        "_ground_truth",
        "-ground_truth",
        " ground_truth",
    ]
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True
    return re.sub(r"[^a-z0-9]+", "", stem)


def list_image_files(data_dir):
    return sorted(
        path
        for path in Path(data_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_image_mask_pairs(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            "Download the ISIC melanoma segmentation zip from the README link, "
            "extract it, and pass the extracted folder with --data-dir."
        )

    files = list_image_files(data_dir)
    if not files:
        raise FileNotFoundError(
            f"No jpg/jpeg/png files found under {data_dir}.\n"
            "Check that the ISIC segmentation dataset was extracted correctly."
        )

    mask_files = [path for path in files if is_mask_file(path)]
    mask_set = set(mask_files)
    image_files = [path for path in files if path not in mask_set]
    if not mask_files:
        raise FileNotFoundError(
            "No mask files were found. Expected filenames or folders containing "
            "'segmentation', 'mask', or 'groundtruth'."
        )

    masks_by_key = defaultdict(list)
    for mask_path in mask_files:
        masks_by_key[normalized_pair_key(mask_path)].append(mask_path)

    pairs = []
    used_masks = set()
    for image_path in image_files:
        key = normalized_pair_key(image_path)
        candidates = [mask for mask in masks_by_key.get(key, []) if mask not in used_masks]
        if not candidates:
            continue
        candidates = sorted(
            candidates,
            key=lambda mask: (
                0 if mask.parent == image_path.parent else 1,
                len(str(mask.relative_to(data_dir))),
            ),
        )
        mask_path = candidates[0]
        pairs.append((image_path, mask_path))
        used_masks.add(mask_path)

    if not pairs:
        raise FileNotFoundError(
            "No image/mask pairs could be matched. Images should be named like "
            "ISIC_0000000.jpg and masks like ISIC_0000000_segmentation.png, "
            "or placed in clearly named image and mask folders."
        )
    return sorted(pairs, key=lambda item: str(item[0]))


class ISICSegmentationDataset(Dataset):
    def __init__(self, pairs, image_size=(256, 256), augment=False):
        self.pairs = list(pairs)
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        width, height = self.image_size[1], self.image_size[0]
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        mask = mask.resize((width, height), Image.Resampling.NEAREST)

        if self.augment:
            image, mask = self.apply_augmentations(image, mask)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        return image_tensor, mask_tensor

    @staticmethod
    def apply_augmentations(image, mask):
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            angle = random.uniform(-15.0, 15.0)
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
            mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
        return image, mask


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(
            x,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = torch.sum(probs * targets, dim=dims)
        denominator = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
        dice = (2.0 * intersection + SMOOTH) / (denominator + SMOOTH)
        dice_loss = 1.0 - dice.mean()
        return bce_loss + dice_loss


def update_metric_sums(metric_sums, logits, targets):
    preds = (torch.sigmoid(logits) > 0.5).float()
    targets = targets.float()
    metric_sums["intersection"] += torch.sum(preds * targets).item()
    metric_sums["pred_sum"] += torch.sum(preds).item()
    metric_sums["target_sum"] += torch.sum(targets).item()
    metric_sums["union"] += torch.sum((preds + targets) > 0).item()
    metric_sums["correct"] += torch.sum(preds == targets).item()
    metric_sums["pixels"] += targets.numel()


def compute_metrics(metric_sums):
    intersection = metric_sums["intersection"]
    pred_sum = metric_sums["pred_sum"]
    target_sum = metric_sums["target_sum"]
    union = metric_sums["union"]
    correct = metric_sums["correct"]
    pixels = metric_sums["pixels"]
    return {
        "dice": float((2.0 * intersection + SMOOTH) / (pred_sum + target_sum + SMOOTH)),
        "iou": float((intersection + SMOOTH) / (union + SMOOTH)),
        "pixel_accuracy": float(correct / max(pixels, 1)),
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    metric_sums = defaultdict(float)

    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        update_metric_sums(metric_sums, logits.detach(), masks)

    metrics = compute_metrics(metric_sums)
    metrics["loss"] = float(total_loss / max(total_samples, 1))
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="eval"):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    metric_sums = defaultdict(float)

    for images, masks in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        update_metric_sums(metric_sums, logits, masks)

    metrics = compute_metrics(metric_sums)
    metrics["loss"] = float(total_loss / max(total_samples, 1))
    return metrics


def deterministic_split(pairs, seed, val_ratio):
    pairs = list(pairs)
    if len(pairs) < 2:
        raise ValueError("At least two image/mask pairs are required for an 80/20 split.")

    rng = random.Random(seed)
    rng.shuffle(pairs)

    test_count = max(1, int(len(pairs) * 0.2))
    test_pairs = pairs[:test_count]
    train_val_pairs = pairs[test_count:]
    if not train_val_pairs:
        raise ValueError("Training split is empty. Add more image/mask pairs.")

    val_count = 0
    if val_ratio > 0 and len(train_val_pairs) > 1:
        val_count = max(1, int(len(train_val_pairs) * val_ratio))
        val_count = min(val_count, len(train_val_pairs) - 1)

    val_pairs = train_val_pairs[:val_count]
    train_pairs = train_val_pairs[val_count:]
    return train_pairs, val_pairs, test_pairs


def make_loader(dataset, batch_size, shuffle, num_workers, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def save_curves(history, out_dir):
    epochs = [entry["epoch"] for entry in history]

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, [entry["train_loss"] for entry in history], label="train")
    if "val_loss" in history[0]:
        plt.plot(epochs, [entry["val_loss"] for entry in history], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, [entry["train_dice"] for entry in history], label="train")
    if "val_dice" in history[0]:
        plt.plot(epochs, [entry["val_dice"] for entry in history], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Dice coefficient")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "dice_curve.png", dpi=150)
    plt.close()


@torch.no_grad()
def save_prediction_examples(model, dataset, device, examples_dir, max_examples):
    examples_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    for index in range(min(max_examples, len(dataset))):
        image_tensor, mask_tensor = dataset[index]
        logits = model(image_tensor.unsqueeze(0).to(device))
        pred = (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5).astype(np.float32)

        image = image_tensor.permute(1, 2, 0).numpy()
        mask = mask_tensor[0].numpy()
        overlay = image.copy()
        overlay[..., 0] = np.maximum(overlay[..., 0], pred)
        overlay[..., 1] = overlay[..., 1] * (1.0 - 0.35 * pred)
        overlay[..., 2] = overlay[..., 2] * (1.0 - 0.35 * pred)

        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        axes[0].imshow(image)
        axes[0].set_title("Image")
        axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground truth")
        axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Prediction")
        axes[3].imshow(np.clip(overlay, 0.0, 1.0))
        axes[3].set_title("Overlay")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(examples_dir / f"example_{index:03d}.png", dpi=150)
        plt.close(fig)


def save_metrics(metrics, out_dir):
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    lines = [
        "ISIC U-Net segmentation test metrics",
        f"test_loss: {metrics['test_loss']:.6f}",
        f"dice: {metrics['dice']:.6f}",
        f"iou: {metrics['iou']:.6f}",
        f"pixel_accuracy: {metrics['pixel_accuracy']:.6f}",
        "",
        f"train_images: {metrics['train_images']}",
        f"val_images: {metrics['val_images']}",
        f"test_images: {metrics['test_images']}",
        f"best_epoch: {metrics['best_epoch']}",
        f"device: {metrics['device']}",
    ]
    (out_dir / "metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def serializable_args(args):
    values = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            values[key] = str(value)
        elif isinstance(value, tuple):
            values[key] = list(value)
        else:
            values[key] = value
    return values


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple U-Net for ISIC lesion segmentation.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Extracted ISIC dataset folder.")
    parser.add_argument("--out-dir", type=Path, default=Path("results_isic_unet"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=parse_image_size, default=(256, 256))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-examples", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.image_size[0] < 16 or args.image_size[1] < 16:
        raise ValueError("--image-size must be at least 16 pixels in each dimension.")
    if args.val_ratio < 0 or args.val_ratio >= 1:
        raise ValueError("--val-ratio must be in the range [0, 1).")

    set_seed(args.seed)
    device = get_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_image_mask_pairs(args.data_dir)
    train_pairs, val_pairs, test_pairs = deterministic_split(pairs, args.seed, args.val_ratio)

    split_info = {
        "data_dir": str(args.data_dir),
        "total_pairs": len(pairs),
        "train_images": len(train_pairs),
        "val_images": len(val_pairs),
        "test_images": len(test_pairs),
        "seed": args.seed,
    }
    (args.out_dir / "split.json").write_text(json.dumps(split_info, indent=2) + "\n", encoding="utf-8")

    train_dataset = ISICSegmentationDataset(train_pairs, args.image_size, augment=True)
    val_dataset = ISICSegmentationDataset(val_pairs, args.image_size, augment=False)
    test_dataset = ISICSegmentationDataset(test_pairs, args.image_size, augment=False)

    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, args.seed)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers, args.seed)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers, args.seed)

    model = UNet().to(device)
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_dice = -1.0
    best_epoch = 0
    best_model_path = args.out_dir / "best_unet.pt"
    history = []

    print(f"Found {len(pairs)} image/mask pairs.")
    print(
        f"Split: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test. "
        f"Device: {device}"
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
        }

        score_metrics = train_metrics
        if val_loader is not None and len(val_dataset) > 0:
            val_metrics = evaluate(model, val_loader, criterion, device, desc="val")
            record["val_loss"] = val_metrics["loss"]
            record["val_dice"] = val_metrics["dice"]
            score_metrics = val_metrics

        history.append(record)
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={record['train_loss']:.4f} train_dice={record['train_dice']:.4f}"
            + (
                f" val_loss={record['val_loss']:.4f} val_dice={record['val_dice']:.4f}"
                if "val_loss" in record
                else ""
            )
        )

        if score_metrics["dice"] > best_dice:
            best_dice = score_metrics["dice"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_dice": best_dice,
                    "args": serializable_args(args),
                },
                best_model_path,
            )

    if not best_model_path.exists():
        raise RuntimeError("No model checkpoint was saved. Check that --epochs is greater than 0.")

    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    save_curves(history, args.out_dir)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device, desc="test")

    final_metrics = {
        "test_loss": test_metrics["loss"],
        "dice": test_metrics["dice"],
        "iou": test_metrics["iou"],
        "pixel_accuracy": test_metrics["pixel_accuracy"],
        "train_images": len(train_pairs),
        "val_images": len(val_pairs),
        "test_images": len(test_pairs),
        "best_epoch": best_epoch,
        "best_validation_dice": best_dice,
        "device": str(device),
        "image_size": list(args.image_size),
        "seed": args.seed,
    }
    save_metrics(final_metrics, args.out_dir)
    save_prediction_examples(model, test_dataset, device, args.out_dir / "examples", args.num_examples)

    print(f"Saved best model to {best_model_path}")
    print(f"Saved metrics and figures to {args.out_dir}")


if __name__ == "__main__":
    main()

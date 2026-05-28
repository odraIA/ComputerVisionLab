import argparse
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from timm.data import resolve_model_data_config
from torchvision import transforms
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_layers(layers_text):
    layers = []
    for value in layers_text.split(","):
        value = value.strip()
        if value:
            layers.append(int(value))
    if not layers:
        raise ValueError("At least one layer index must be provided.")
    return layers


def get_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Using CPU instead.")
        return torch.device("cpu")
    return torch.device(device_arg)


def list_images(image_dir):
    image_dir = Path(image_dir)
    if not image_dir.exists():
        return []
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def default_image_dir():
    repo_root = Path(__file__).resolve().parents[1]
    image_dir = repo_root / "images"
    return image_dir if list_images(image_dir) else None


def prepare_image(path, input_size, mean, std):
    image = Image.open(path).convert("RGB")
    width, height = input_size
    display = ImageOps.fit(image, (width, height), method=Image.Resampling.BICUBIC)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    tensor = transform(display).unsqueeze(0)
    return display, tensor


def normalize_map(attention_map):
    attention_map = attention_map.astype(np.float32)
    min_value = float(attention_map.min())
    max_value = float(attention_map.max())
    if max_value - min_value < 1e-8:
        return np.zeros_like(attention_map, dtype=np.float32)
    return (attention_map - min_value) / (max_value - min_value)


def attention_to_image_map(attention, grid_size, image_size, heads):
    if heads == "mean":
        cls_attention = attention[0, :, 0, :].mean(dim=0)
    else:
        head_idx = int(heads)
        if head_idx < 0 or head_idx >= attention.shape[1]:
            raise ValueError(
                f"Head index {head_idx} is invalid for a model with {attention.shape[1]} heads."
            )
        cls_attention = attention[0, head_idx, 0, :]

    num_patches = grid_size[0] * grid_size[1]
    patch_attention = cls_attention[-num_patches:].reshape(1, 1, grid_size[0], grid_size[1])
    patch_attention = F.interpolate(
        patch_attention,
        size=(image_size[1], image_size[0]),
        mode="bilinear",
        align_corners=False,
    )
    return normalize_map(patch_attention.squeeze().cpu().numpy())


def attention_rollout(attentions, grid_size, image_size):
    rollout = None
    for attention in attentions:
        attention = attention[0].mean(dim=0)
        identity = torch.eye(attention.shape[-1], device=attention.device)
        attention = attention + identity
        attention = attention / attention.sum(dim=-1, keepdim=True)
        rollout = attention if rollout is None else attention @ rollout

    num_patches = grid_size[0] * grid_size[1]
    cls_rollout = rollout[0, -num_patches:].reshape(1, 1, grid_size[0], grid_size[1])
    cls_rollout = F.interpolate(
        cls_rollout,
        size=(image_size[1], image_size[0]),
        mode="bilinear",
        align_corners=False,
    )
    return normalize_map(cls_rollout.squeeze().cpu().numpy())


def overlay_attention(image, attention_map, alpha=0.45, cmap_name="jet"):
    image_array = np.asarray(image).astype(np.float32) / 255.0
    cmap = plt.get_cmap(cmap_name)
    heatmap = cmap(attention_map)[..., :3]
    overlay = (1.0 - alpha) * image_array + alpha * heatmap
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def save_heatmap(attention_map, output_path):
    plt.imsave(output_path, attention_map, cmap="jet", vmin=0.0, vmax=1.0)


def save_grid(image, entries, output_path):
    columns = 1 + len(entries)
    fig, axes = plt.subplots(1, columns, figsize=(4 * columns, 4))
    if columns == 1:
        axes = [axes]
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    for ax, (title, overlay) in zip(axes[1:], entries):
        ax.imshow(overlay)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def get_grid_size(model, attention):
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "grid_size"):
        grid_size = model.patch_embed.grid_size
        return int(grid_size[0]), int(grid_size[1])

    num_tokens = attention.shape[-1]
    num_prefix_tokens = getattr(model, "num_prefix_tokens", 1)
    num_patches = num_tokens - num_prefix_tokens
    side = int(math.sqrt(num_patches))
    if side * side != num_patches:
        raise ValueError("Could not infer a square patch grid from the attention tensor.")
    return side, side


def disable_fused_attention(model):
    for module in model.modules():
        if hasattr(module, "fused_attn"):
            module.fused_attn = False


def register_attention_hooks(model, capture_layers, attention_store):
    handles = []
    for layer_idx in capture_layers:
        block = model.blocks[layer_idx]
        if not hasattr(block.attn, "attn_drop"):
            raise RuntimeError(
                "This timm model does not expose attn_drop. Try another ViT model."
            )

        def hook(_module, _inputs, output, layer_idx=layer_idx):
            attention_store[layer_idx] = output.detach().cpu()

        handles.append(block.attn.attn_drop.register_forward_hook(hook))
    return handles


def build_report(out_dir, args, image_summaries, selected_layers, rollout_enabled):
    report_path = out_dir / "report_vit_attention.md"
    lines = [
        "# Visualizing attention maps in Vision Transformers",
        "",
        f"- Model: `{args.model_name}`",
        f"- Selected layers: `{','.join(str(layer) for layer in selected_layers)}`",
        f"- Heads: `{args.heads}`",
        f"- Images processed: `{len(image_summaries)}`",
        f"- Attention rollout: `{'enabled' if rollout_enabled else 'disabled'}`",
        "",
        "Earlier ViT layers usually attend to local texture, edges and small parts of the object. "
        "Later layers tend to concentrate more on semantically relevant regions because information "
        "has mixed across many transformer blocks. Attention rollout accumulates attention through "
        "the network, so it is often smoother and more global than a single-layer CLS-token map.",
        "",
        "## Outputs",
        "",
    ]

    for summary in image_summaries:
        lines.extend(
            [
                f"### {summary['name']}",
                "",
                f"- Top-1 class index: `{summary['top1_index']}`",
                f"- Top-1 confidence: `{summary['top1_confidence']:.4f}`",
                f"- Original image: `{summary['original']}`",
                f"- Grid comparison: `{summary['grid']}`",
            ]
        )
        for layer, path in summary["layers"]:
            lines.append(f"- Layer {layer} overlay: `{path}`")
        if summary.get("rollout"):
            lines.append(f"- Attention rollout overlay: `{summary['rollout']}`")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize CLS-token attention maps from pretrained timm Vision Transformers."
    )
    parser.add_argument("--image-dir", default="", help="Directory with input images.")
    parser.add_argument("--out-dir", default="results_vit_attention", help="Output directory.")
    parser.add_argument("--model-name", default="vit_tiny_patch16_224", help="timm ViT model name.")
    parser.add_argument("--layers", default="0,3,6,11", help="Comma-separated ViT block indices.")
    parser.add_argument("--heads", default="mean", help="Use 'mean' or a specific attention head index.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--max-images", type=int, default=4, help="Maximum number of images to process.")
    parser.add_argument("--rollout", action="store_true", help="Also save attention rollout overlays.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    selected_layers = parse_layers(args.layers)

    image_dir = Path(args.image_dir) if args.image_dir else default_image_dir()
    if image_dir is None or not list_images(image_dir):
        raise FileNotFoundError(
            "No input images were found. Pass --image-dir with jpg/png images, for example "
            "`python src/vit_attention_maps.py --image-dir images`."
        )

    image_paths = list_images(image_dir)[: args.max_images]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = timm.create_model(args.model_name, pretrained=True)
    except Exception as exc:
        raise RuntimeError(
            "Could not create the pretrained timm model. Check that `timm` is installed and "
            "that pretrained weights can be downloaded or are already cached."
        ) from exc

    if not hasattr(model, "blocks"):
        raise ValueError("The selected model does not look like a standard timm Vision Transformer.")

    num_layers = len(model.blocks)
    invalid_layers = [layer for layer in selected_layers if layer < 0 or layer >= num_layers]
    if invalid_layers:
        raise ValueError(f"Invalid layer indices {invalid_layers}. Model has {num_layers} blocks.")

    disable_fused_attention(model)
    model.eval().to(device)

    data_config = resolve_model_data_config(model)
    input_height, input_width = data_config["input_size"][-2:]
    mean = data_config["mean"]
    std = data_config["std"]

    capture_layers = set(selected_layers)
    if args.rollout:
        capture_layers.update(range(num_layers))
    capture_layers = sorted(capture_layers)

    image_summaries = []
    for image_path in tqdm(image_paths, desc="Processing images"):
        attention_store = {}
        handles = register_attention_hooks(model, capture_layers, attention_store)
        try:
            display_image, tensor = prepare_image(
                image_path,
                input_size=(input_width, input_height),
                mean=mean,
                std=std,
            )
            tensor = tensor.to(device)
            with torch.no_grad():
                logits = model(tensor)
                probabilities = logits.softmax(dim=-1)
                top1_confidence, top1_index = probabilities.max(dim=-1)
        finally:
            for handle in handles:
                handle.remove()

        missing_layers = [layer for layer in capture_layers if layer not in attention_store]
        if missing_layers:
            raise RuntimeError(
                f"Attention was not captured for layers {missing_layers}. "
                "The model may still be using fused attention."
            )

        image_out_dir = out_dir / image_path.stem
        image_out_dir.mkdir(parents=True, exist_ok=True)

        original_path = image_out_dir / "original.png"
        display_image.save(original_path)

        grid_entries = []
        layer_outputs = []
        first_attention = attention_store[selected_layers[0]]
        grid_size = get_grid_size(model, first_attention)

        for layer_idx in selected_layers:
            attention_map = attention_to_image_map(
                attention_store[layer_idx],
                grid_size=grid_size,
                image_size=display_image.size,
                heads=args.heads,
            )
            overlay = overlay_attention(display_image, attention_map)
            overlay_path = image_out_dir / f"layer_{layer_idx:02d}_attention_overlay.png"
            heatmap_path = image_out_dir / f"layer_{layer_idx:02d}_attention_heatmap.png"
            overlay.save(overlay_path)
            save_heatmap(attention_map, heatmap_path)
            grid_entries.append((f"Layer {layer_idx}", overlay))
            layer_outputs.append((layer_idx, overlay_path.relative_to(out_dir).as_posix()))

        rollout_path = None
        if args.rollout:
            rollout_map = attention_rollout(
                [attention_store[layer] for layer in range(num_layers)],
                grid_size=grid_size,
                image_size=display_image.size,
            )
            rollout_overlay = overlay_attention(display_image, rollout_map)
            rollout_path = image_out_dir / "attention_rollout_overlay.png"
            rollout_heatmap_path = image_out_dir / "attention_rollout_heatmap.png"
            rollout_overlay.save(rollout_path)
            save_heatmap(rollout_map, rollout_heatmap_path)
            grid_entries.append(("Rollout", rollout_overlay))

        grid_path = image_out_dir / "attention_grid.png"
        save_grid(display_image, grid_entries, grid_path)

        image_summaries.append(
            {
                "name": image_path.name,
                "top1_index": int(top1_index.item()),
                "top1_confidence": float(top1_confidence.item()),
                "original": original_path.relative_to(out_dir).as_posix(),
                "grid": grid_path.relative_to(out_dir).as_posix(),
                "layers": layer_outputs,
                "rollout": rollout_path.relative_to(out_dir).as_posix()
                if rollout_path is not None
                else None,
            }
        )

    report_path = build_report(out_dir, args, image_summaries, selected_layers, args.rollout)
    print(f"Saved attention visualizations to: {out_dir}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()

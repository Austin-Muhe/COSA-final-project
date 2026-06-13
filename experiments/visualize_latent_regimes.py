#!/usr/bin/env python
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external" / "COSA_ICLR2026"
sys.path.insert(0, str(EXTERNAL_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_cfg_defaults, get_norm_module_cfg
from datasets.build import update_cfg_from_dataset
from datasets.loader import get_val_dataloader
from models.build import build_model, build_norm_module, load_best_model
from tta.cosa import DynamicRegimeAdapter
from utils.misc import prepare_inputs, set_devices, set_seeds


DATA_FILES = {
    "ETTh1": "ETTh1.csv",
    "weather": "weather.csv",
    "exchange_rate": "exchange_rate.csv",
}
SHIFT_TYPES = ("Normal", "Level", "Variance", "Spike")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize learned latent regime factors z.")
    parser.add_argument("--dataset", default="weather")
    parser.add_argument("--model", default="DLinear")
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--regime_checkpoint", required=True)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--projection", choices=["tsne", "pca"], default="tsne")
    parser.add_argument("--max_batches", type=int, default=30)
    parser.add_argument("--max_points_per_regime", type=int, default=2000)
    parser.add_argument("--level_scale", type=float, default=1.0)
    parser.add_argument("--variance_scale", type=float, default=1.8)
    parser.add_argument("--spike_scale", type=float, default=4.0)
    parser.add_argument("--spike_fraction", type=float, default=0.08)
    parser.add_argument("--output_dir", default="./results/regime_adapter_latents")
    parser.add_argument("--visible_devices", default="0")
    return parser.parse_args()


def build_cfg(args):
    cfg = get_cfg_defaults()
    cfg.VISIBLE_DEVICES = args.visible_devices
    cfg.DATA.NAME = args.dataset
    cfg.DATA.BASE_DIR = str(EXTERNAL_ROOT / "data")
    cfg.DATA.PRED_LEN = args.pred_len
    cfg.MODEL.NAME = args.model
    cfg.MODEL.pred_len = args.pred_len
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = False
    cfg.TTA.ENABLE = False
    cfg.VAL.BATCH_SIZE = args.batch_size
    cfg.VAL.SHUFFLE = False
    cfg.VAL.DROP_LAST = False

    update_cfg_from_dataset(cfg, args.dataset)

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = EXTERNAL_ROOT / "checkpoints" / args.model / f"{args.dataset}_{args.pred_len}"
    cfg.TRAIN.CHECKPOINT_DIR = str(checkpoint_dir)
    return cfg


def preflight_inputs(cfg, args):
    data_file = DATA_FILES.get(cfg.DATA.NAME, f"{cfg.DATA.NAME}.csv")
    data_path = Path(cfg.DATA.BASE_DIR) / cfg.DATA.NAME / data_file
    backbone_checkpoint = Path(cfg.TRAIN.CHECKPOINT_DIR) / "checkpoint_best.pth"
    regime_checkpoint = Path(args.regime_checkpoint)

    missing = []
    if not data_path.exists():
        missing.append(f"data file: {data_path}")
    if not backbone_checkpoint.exists():
        missing.append(f"backbone checkpoint: {backbone_checkpoint}")
    if not regime_checkpoint.exists():
        missing.append(f"regime checkpoint: {regime_checkpoint}")
    if missing:
        raise FileNotFoundError("Missing required input(s):\n  - " + "\n  - ".join(missing))


def build_dynamic_adapter(cfg, args, device):
    payload = torch.load(args.regime_checkpoint, map_location="cpu")
    args.latent_dim = int(payload.get("latent_dim", args.latent_dim))
    args.hidden_dim = int(payload.get("hidden_dim", args.hidden_dim))
    adapter = DynamicRegimeAdapter(
        feature_dim=cfg.DATA.N_VAR,
        horizon=cfg.DATA.PRED_LEN,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    adapter.load_state_dict(payload.get("adapter_state", payload), strict=True)
    adapter.eval()
    return adapter


def make_level_shift(context_window, scale):
    context_std = context_window.std(dim=1, keepdim=True).clamp_min(1e-6)
    direction = torch.sign(torch.randn(
        context_window.shape[0], 1, context_window.shape[2],
        device=context_window.device, dtype=context_window.dtype,
    ))
    return context_window + direction * context_std * scale


def make_variance_shift(context_window, scale):
    center = context_window.mean(dim=1, keepdim=True)
    return center + (context_window - center) * scale


def make_spike_shift(context_window, scale, fraction):
    shifted = context_window.clone()
    batch_size, context_len, feature_dim = shifted.shape
    num_spikes = max(1, int(context_len * fraction))
    context_std = shifted.std(dim=1, keepdim=True).clamp_min(1e-6)

    for sample_idx in range(batch_size):
        spike_idx = torch.randperm(context_len, device=shifted.device)[:num_spikes]
        spike_sign = torch.sign(torch.randn(num_spikes, feature_dim, device=shifted.device, dtype=shifted.dtype))
        shifted[sample_idx, spike_idx, :] += spike_sign * context_std[sample_idx, 0, :] * scale
    return shifted


def build_shifted_contexts(context_window, args):
    return {
        "Normal": context_window,
        "Level": make_level_shift(context_window, args.level_scale),
        "Variance": make_variance_shift(context_window, args.variance_scale),
        "Spike": make_spike_shift(context_window, args.spike_scale, args.spike_fraction),
    }


@torch.no_grad()
def collect_latents(cfg, adapter, args, device):
    val_loader = get_val_dataloader(cfg)
    z_by_regime = {name: [] for name in SHIFT_TYPES}
    g_by_regime = {name: [] for name in SHIFT_TYPES}

    for batch_idx, inputs in enumerate(val_loader):
        if args.max_batches and batch_idx >= args.max_batches:
            break
        enc_window, _, _, _ = prepare_inputs(inputs)
        enc_window = enc_window.to(device)

        shifted_contexts = build_shifted_contexts(enc_window, args)
        for regime, context in shifted_contexts.items():
            g, z = adapter(context)
            z_by_regime[regime].append(z.detach().cpu())
            g_by_regime[regime].append(g.detach().cpu())

    z_list = []
    g_list = []
    labels = []
    label_ids = []
    for label_id, regime in enumerate(SHIFT_TYPES):
        if not z_by_regime[regime]:
            continue
        z = torch.cat(z_by_regime[regime], dim=0)
        g = torch.cat(g_by_regime[regime], dim=0)
        if len(z) > args.max_points_per_regime:
            keep = torch.randperm(len(z))[:args.max_points_per_regime]
            z = z[keep]
            g = g[keep]
        z_list.append(z.numpy())
        g_list.append(g.numpy())
        labels.extend([regime] * len(z))
        label_ids.extend([label_id] * len(z))

    if not z_list:
        raise RuntimeError("No latent vectors were collected from validation data.")

    return np.concatenate(z_list, axis=0), np.concatenate(g_list, axis=0), np.array(labels), np.array(label_ids)


def project_latents(z, method):
    if method == "pca":
        projector = PCA(n_components=2, random_state=0)
        return projector.fit_transform(z)

    perplexity = min(30, max(5, (len(z) - 1) // 3))
    projector = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=0,
    )
    return projector.fit_transform(z)


def plot_projection(points_2d, labels, args, output_path):
    colors = {
        "Normal": "#2b6cb0",
        "Level": "#c53030",
        "Variance": "#2f855a",
        "Spike": "#b7791f",
    }
    plt.figure(figsize=(8, 6), dpi=160)
    for regime in SHIFT_TYPES:
        mask = labels == regime
        if mask.any():
            plt.scatter(
                points_2d[mask, 0], points_2d[mask, 1],
                s=10, alpha=0.72, c=colors[regime], label=regime, linewidths=0,
            )
    plt.title(f"Latent Regime Factors z ({args.projection.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_outputs(run_dir, points_2d, z, g, labels, args):
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "latent_z.npy", z)
    np.save(run_dir / "gate_g.npy", g)
    np.save(run_dir / "projection_2d.npy", points_2d)
    np.save(run_dir / "labels.npy", labels)

    with (run_dir / "projection_2d.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "label"])
        for point, label in zip(points_2d, labels):
            writer.writerow([float(point[0]), float(point[1]), label])

    metadata = {
        "dataset": args.dataset,
        "model": args.model,
        "pred_len": args.pred_len,
        "projection": args.projection,
        "regime_checkpoint": args.regime_checkpoint,
        "num_points": int(len(labels)),
        "shift_types": list(SHIFT_TYPES),
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main():
    args = parse_args()
    cfg = build_cfg(args)
    preflight_inputs(cfg, args)
    set_devices(args.visible_devices)
    set_seeds(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    norm_module = build_norm_module(cfg) if cfg.NORM_MODULE.ENABLE else None
    model = load_best_model(cfg, model)
    if cfg.NORM_MODULE.ENABLE:
        norm_module = load_best_model(get_norm_module_cfg(cfg), norm_module)
    model.eval()
    if norm_module is not None:
        norm_module.eval()

    adapter = build_dynamic_adapter(cfg, args, device)
    z, g, labels, _ = collect_latents(cfg, adapter, args, device)
    points_2d = project_latents(z, args.projection)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    run_dir = output_dir / args.projection / args.model / args.dataset / str(args.pred_len)
    image_path = run_dir / "latent_regimes.png"
    save_outputs(run_dir, points_2d, z, g, labels, args)
    plot_projection(points_2d, labels, args, image_path)

    print(json.dumps({
        "image": str(image_path),
        "points": int(len(labels)),
        "projection": args.projection,
        "labels": {name: int((labels == name).sum()) for name in SHIFT_TYPES},
    }, indent=2))


if __name__ == "__main__":
    main()

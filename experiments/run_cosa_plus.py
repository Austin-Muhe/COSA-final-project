#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external" / "COSA_ICLR2026"
sys.path.insert(0, str(EXTERNAL_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_cfg_defaults, get_norm_module_cfg
from datasets.build import update_cfg_from_dataset
from models.build import build_model, build_norm_module, load_best_model
from utils.misc import set_devices, set_seeds

from experiments.tta.cosa_plus import COSAPlusAdapter


DATA_FILES = {
    "ETTh1": "ETTh1.csv",
    "weather": "weather.csv",
    "exchange_rate": "exchange_rate.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run COSA+ variants using the official COSA pipeline.")
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--model", default="DLinear")
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument(
        "--variant",
        default="original",
        choices=["original", "vec_gate", "rich_ctx", "ctx_std_only", "cosa_plus"],
    )
    parser.add_argument("--output_dir", default="./results/ablation/")
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--ctx_len", type=int, default=10)
    parser.add_argument("--base_lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--checkpoint_dir", default=None)
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
    cfg.TTA.ENABLE = True
    cfg.TTA.SOLVER.BASE_LR = args.base_lr
    cfg.TTA.SOLVER.WEIGHT_DECAY = args.weight_decay
    cfg.TTA.COSA.BATCH_SIZE = args.batch_size
    cfg.TTA.COSA.STEPS = args.steps
    cfg.TTA.COSA.BUFFER_CONTEXT_SIZE = args.ctx_len
    cfg.TTA.COSA.FAST_ADAPTATION = True
    cfg.TTA.COSA.ADAPTIVE_LR = True
    cfg.TTA.COSA.PER_BATCH_LR_RESET = True
    cfg.TTA.COSA.PAAS = False

    update_cfg_from_dataset(cfg, args.dataset)

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = EXTERNAL_ROOT / "checkpoints" / args.model / f"{args.dataset}_{args.pred_len}"
    cfg.TRAIN.CHECKPOINT_DIR = str(checkpoint_dir)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    cfg.RESULT_DIR = str(output_dir / args.variant / args.model / args.dataset / str(args.pred_len))
    os.makedirs(cfg.RESULT_DIR, exist_ok=True)
    return cfg


def preflight_inputs(cfg):
    data_file = DATA_FILES.get(cfg.DATA.NAME, f"{cfg.DATA.NAME}.csv")
    data_path = Path(cfg.DATA.BASE_DIR) / cfg.DATA.NAME / data_file
    checkpoint_path = Path(cfg.TRAIN.CHECKPOINT_DIR) / "checkpoint_best.pth"

    missing = []
    if not data_path.exists():
        missing.append(f"data file: {data_path}")
    if not checkpoint_path.exists():
        missing.append(f"checkpoint: {checkpoint_path}")
    if missing:
        message = "Missing required input(s):\n  - " + "\n  - ".join(missing)
        raise FileNotFoundError(message)


def main():
    args = parse_args()
    cfg = build_cfg(args)
    preflight_inputs(cfg)
    set_devices(cfg.VISIBLE_DEVICES)
    set_seeds(cfg.SEED)

    model = build_model(cfg)
    norm_module = build_norm_module(cfg) if cfg.NORM_MODULE.ENABLE else None
    model = load_best_model(cfg, model)
    if cfg.NORM_MODULE.ENABLE:
        norm_module = load_best_model(get_norm_module_cfg(cfg), norm_module)

    adapter = COSAPlusAdapter(cfg, model, norm_module=norm_module, variant=args.variant)
    results = adapter.adapt()
    gate_path = adapter.save_gate(cfg.RESULT_DIR, args.dataset, args.model, args.pred_len)
    adapter.print_results()
    print(f"Gate: {gate_path}")

    with open(Path(cfg.RESULT_DIR) / "metrics.txt", "w", encoding="utf-8") as handle:
        handle.write(f"variant: {args.variant}\n")
        handle.write(f"MSE: {results['test_mse']:.6f}\n")
        handle.write(f"MAE: {results['test_mae']:.6f}\n")
        handle.write(f"Gate: {gate_path}\n")


if __name__ == "__main__":
    main()

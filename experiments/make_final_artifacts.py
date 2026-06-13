#!/usr/bin/env python
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "results" / "final_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _parse_metric_log(path: Path) -> dict:
    text = path.read_text(errors="ignore")
    mse = re.search(r"MSE:\s*([0-9.]+)", text)
    mae = re.search(r"MAE:\s*([0-9.]+)", text)
    if not mse:
        raise ValueError(f"No MSE found in {path}")
    return {
        "mse": float(mse.group(1)),
        "mae": float(mae.group(1)) if mae else np.nan,
    }


def to_markdown_table(df: pd.DataFrame) -> str:
    cols = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                if value.is_integer() and abs(value) >= 1:
                    values.append(str(int(value)))
                else:
                    values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def load_clean_results() -> pd.DataFrame:
    rows = []
    specs = [
        (PROJECT_ROOT / "results" / "ablation_etth1_dlinear", "ETTh1"),
        (PROJECT_ROOT / "results" / "ablation_reduced", "ETTh1"),
        (PROJECT_ROOT / "results" / "ablation_weather_dlinear", "weather"),
    ]
    for result_dir, dataset in specs:
        for path in sorted(result_dir.glob(f"*_{dataset}_DLinear_*.txt")):
            horizon = int(path.stem.rsplit("_", 1)[1])
            if result_dir.name == "ablation_reduced" and horizon != 96:
                continue
            variant = path.stem.replace(f"_{dataset}_DLinear_{horizon}", "")
            metrics = _parse_metric_log(path)
            rows.append(
                {
                    "dataset": dataset,
                    "model": "DLinear",
                    "horizon": horizon,
                    "variant": variant,
                    **metrics,
                }
            )
    return pd.DataFrame(rows).drop_duplicates(
        subset=["dataset", "model", "horizon", "variant"],
        keep="first",
    )


def load_blackswan(result_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(result_dir.glob("*.json")):
        with open(path, encoding="utf-8") as handle:
            rows.append(json.load(handle))
    return pd.DataFrame(rows)


def save_tables(clean_df: pd.DataFrame):
    clean_df.sort_values(["dataset", "horizon", "variant"]).to_csv(OUT_DIR / "clean_results_long.csv", index=False)

    main_rows = clean_df[clean_df["variant"].isin(["original", "cosa_plus"])]
    main_table = main_rows.pivot_table(
        index=["dataset", "horizon"],
        columns="variant",
        values="mse",
        aggfunc="first",
    ).reset_index()
    main_table.to_csv(OUT_DIR / "table1_clean_main.csv", index=False)
    (OUT_DIR / "table1_clean_main.md").write_text(to_markdown_table(main_table), encoding="utf-8")

    ablation = clean_df[clean_df["dataset"].eq("ETTh1")].pivot_table(
        index="horizon",
        columns="variant",
        values="mse",
        aggfunc="first",
    )
    ablation = ablation.reindex(columns=["original", "vec_gate", "rich_ctx", "ctx_std_only", "cosa_plus"])
    ablation.to_csv(OUT_DIR / "table2_etth1_ablation.csv")
    (OUT_DIR / "table2_etth1_ablation.md").write_text(to_markdown_table(ablation.reset_index()), encoding="utf-8")


def plot_etth1_blackswan():
    df = load_blackswan(PROJECT_ROOT / "results" / "blackswan_mag3_720")
    shift_types = ["level", "variance", "trend", "spike"]
    variants = ["no_tta", "original", "cosa_plus"]
    colors = {"no_tta": "#6E6E6E", "original": "#0072B2", "cosa_plus": "#D55E00"}
    linestyles = {"no_tta": "--", "original": "-", "cosa_plus": "-"}

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.0), sharex=True)
    window_before = 100
    window_after = 120

    for ax, shift_type in zip(axes.flat, shift_types):
        subset = df[df["shift_type"].eq(shift_type)]
        for variant in variants:
            row = subset[subset["variant"].eq(variant)]
            if row.empty:
                continue
            record = row.iloc[0]
            rolling = np.asarray(record["rolling_mse"], dtype=float)
            t_shift = int(record["t_shift"])
            start = max(0, t_shift - window_before)
            end = min(len(rolling), t_shift + window_after)
            x = np.arange(start, end) - t_shift
            ax.plot(
                x,
                rolling[start:end],
                label=variant,
                color=colors[variant],
                linestyle=linestyles[variant],
                linewidth=1.8,
            )
        ax.axvline(0, color="black", linestyle=":", linewidth=1.2)
        ax.set_title(shift_type)
        ax.set_xlabel("Steps from shift")
        ax.set_ylabel("Rolling MSE")
        ax.grid(alpha=0.2)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("ETTh1 / DLinear / Horizon 720: Black Swan Recovery (magnitude=3.0)", y=0.985)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT_DIR / "figure1_etth1_h720_blackswan_recovery.png", dpi=200)
    fig.savefig(OUT_DIR / "figure1_etth1_h720_blackswan_recovery.pdf")
    plt.close(fig)

    summary = df.pivot_table(index="shift_type", columns="variant", values="mse_after_50", aggfunc="first")
    summary = summary.reindex(index=shift_types, columns=variants)
    summary.to_csv(OUT_DIR / "table_blackswan_etth1_h720_mse_after50.csv")
    (OUT_DIR / "table_blackswan_etth1_h720_mse_after50.md").write_text(
        to_markdown_table(summary.reset_index()),
        encoding="utf-8",
    )


def plot_ablation(clean_df: pd.DataFrame):
    variants = ["vec_gate", "rich_ctx", "ctx_std_only", "cosa_plus"]
    labels = ["VecGate", "RichCtx", "StdOnly", "COSA+"]
    colors = ["#8C8C8C", "#F4A261", "#D95F5F", "#F2C879"]
    etth1 = clean_df[clean_df["dataset"].eq("ETTh1")]
    horizons = [96, 192, 336, 720]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(horizons))
    width = 0.17

    for idx, (variant, label, color) in enumerate(zip(variants, labels, colors)):
        values = []
        for horizon in horizons:
            subset = etth1[etth1["horizon"].eq(horizon)].set_index("variant")
            original = float(subset.loc["original", "mse"])
            candidate = float(subset.loc[variant, "mse"])
            values.append((original - candidate) * 1000.0)
        offset = (idx - (len(variants) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=color,
            alpha=0.72,
            edgecolor=color,
        )
        for bar, value in zip(bars, values):
            if value >= 0:
                y = value + 0.03
                va = "bottom"
            else:
                y = value - 0.03
                va = "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{value:.2f}",
                ha="center",
                va=va,
                fontsize=9,
            )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L={horizon}" for horizon in horizons])
    ax.set_ylabel(r"MSE reduction vs Original COSA ($\times 10^{-3}$)")
    ax.set_title("ETTh1 / DLinear Ablation: Improvement Over Original COSA")
    ax.legend(loc="upper left", ncol=4, frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(min(ymin, -0.65), max(ymax, 2.45))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure2_etth1_ablation_bar.png", dpi=200)
    fig.savefig(OUT_DIR / "figure2_etth1_ablation_bar.pdf")
    plt.close(fig)


def plot_gate_heatmap():
    candidates = [
        PROJECT_ROOT / "results" / "ablation_etth1_dlinear" / "cosa_plus" / "DLinear" / "ETTh1" / "720" / "gate_cosa_plus_ETTh1_DLinear_720.npy",
        PROJECT_ROOT / "results" / "ablation_reduced" / "cosa_plus" / "DLinear" / "ETTh1" / "720" / "gate_cosa_plus_ETTh1_DLinear_720.npy",
    ]
    gate_path = next((path for path in candidates if path.exists()), None)
    if gate_path is None:
        return
    gate = np.load(gate_path).reshape(-1)
    fig, ax = plt.subplots(figsize=(10, 2.4))
    im = ax.imshow(gate.reshape(1, -1), aspect="auto", cmap="coolwarm")
    ax.set_yticks([])
    ax.set_xlabel("Forecast step")
    ax.set_title("COSA+ Learned Vector Gate, ETTh1 / DLinear / L=720")
    fig.colorbar(im, ax=ax, orientation="vertical", label="tanh(g)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure3_gate_heatmap_etth1_h720.png", dpi=200)
    fig.savefig(OUT_DIR / "figure3_gate_heatmap_etth1_h720.pdf")
    plt.close(fig)


def plot_weather_blackswan_optional():
    result_dir = PROJECT_ROOT / "results" / "blackswan_weather_mag3_720"
    if not result_dir.exists():
        return
    df = load_blackswan(result_dir)
    variants = ["no_tta", "original", "rich_ctx", "cosa_plus"]
    shift_types = ["level", "variance", "trend", "spike"]
    summary = df.pivot_table(index="shift_type", columns="variant", values="mse_after_50", aggfunc="first")
    summary = summary.reindex(index=shift_types, columns=variants)
    summary.to_csv(OUT_DIR / "table_weather_blackswan_h720_mse_after50.csv")
    (OUT_DIR / "table_weather_blackswan_h720_mse_after50.md").write_text(
        to_markdown_table(summary.reset_index()),
        encoding="utf-8",
    )


def main():
    clean_df = load_clean_results()
    save_tables(clean_df)
    plot_etth1_blackswan()
    plot_ablation(clean_df)
    plot_gate_heatmap()
    plot_weather_blackswan_optional()
    print(f"Saved final artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()

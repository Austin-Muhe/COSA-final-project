import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SHIFT_TYPES = ["level", "variance", "trend", "spike"]

ABLATION_METHODS = {
    "original": "Original COSA",
    "vec_gate": "VecGate",
    "rich_ctx": "RichCtx",
    "cosa_plus": "COSA+",
}

SWEEP_METHODS = {
    "no_tta": "No-TTA",
    "original": "Original COSA",
    "cosa_plus": "COSA+",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build small black-swan result tables and plots.")
    parser.add_argument("--input_dir", action="append", required=True)
    parser.add_argument("--output_dir", default="./results/final_figures/")
    parser.add_argument("--experiment", choices=["ablation", "magnitude"], required=True)
    return parser.parse_args()


def load_rows(input_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for input_dir in input_dirs:
        for path in sorted(input_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as handle:
                item = json.load(handle)
            rows.append(
                {
                    "dataset": item["dataset"],
                    "model": item["model"],
                    "horizon": int(item["pred_len"]),
                    "shift_type": item["shift_type"],
                    "magnitude": float(item["magnitude"]),
                    "method": item["variant"],
                    "mse_after_50": float(item["mse_after_50"]),
                    "source_file": str(path),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No JSON result files found in {input_dirs}")
    df = pd.DataFrame(rows)
    return df.sort_values(["dataset", "model", "horizon", "magnitude", "shift_type", "method", "source_file"])


def to_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "model", "horizon", "shift_type", "magnitude", "method"]
    return df.drop_duplicates(subset=keys, keep="last")


def available_path(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(3, 100):
        candidate = path.with_name(f"{path.stem}_v{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"No available non-overwriting name for {path}")


def write_ablation(df: pd.DataFrame, output_dir: Path) -> None:
    methods = list(ABLATION_METHODS)
    df = df[
        df["dataset"].eq("ETTh1")
        & df["model"].eq("DLinear")
        & df["horizon"].eq(720)
        & df["magnitude"].eq(3.0)
        & df["method"].isin(methods)
    ].copy()
    df = dedupe_latest(df)
    df["method_label"] = df["method"].map(ABLATION_METHODS)

    required = {(shift, method) for shift in SHIFT_TYPES for method in methods}
    actual = set(zip(df["shift_type"], df["method"]))
    missing = sorted(required - actual)
    if missing:
        raise RuntimeError(f"Missing ablation rows: {missing}")

    long_cols = ["dataset", "model", "horizon", "shift_type", "magnitude", "method", "mse_after_50"]
    long_path = available_path(output_dir / "table_etth1_blackswan_ablation_h720_mag3_long_v2.csv")
    table_path = available_path(output_dir / "table_etth1_blackswan_ablation_h720_mag3_v2.csv")
    md_path = available_path(output_dir / "table_etth1_blackswan_ablation_h720_mag3_v2.md")
    plot_path = output_dir / "figure_etth1_blackswan_ablation_h720_mag3_bar_v2.png"
    plot_pdf_path = output_dir / "figure_etth1_blackswan_ablation_h720_mag3_bar_v2.pdf"

    df[long_cols].to_csv(long_path, index=False)

    summary = df.pivot_table(index="shift_type", columns="method_label", values="mse_after_50", aggfunc="first")
    summary = summary.reindex(index=SHIFT_TYPES, columns=[ABLATION_METHODS[m] for m in methods])
    summary.to_csv(table_path)
    md_path.write_text(to_markdown_table(summary.reset_index()), encoding="utf-8")

    improvement = summary[[ABLATION_METHODS[m] for m in ["vec_gate", "rich_ctx", "cosa_plus"]]].copy()
    for column in improvement.columns:
        improvement[column] = (summary["Original COSA"] - improvement[column]) * 1000.0

    colors = {
        "VecGate": "#A6A6A6",
        "RichCtx": "#F4B183",
        "COSA+": "#F7D999",
    }
    ax = improvement.plot(
        kind="bar",
        figsize=(8.8, 4.6),
        width=0.72,
        color=[colors[column] for column in improvement.columns],
    )
    ax.set_xlabel("Shift type")
    ax.set_ylabel(r"MSE reduction vs Original COSA ($\times 10^{-3}$)")
    ax.set_title("ETTh1 / DLinear / Horizon 720: Black-Swan Ablation Improvement")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(title="Method", frameon=False, ncol=3, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.savefig(plot_pdf_path)
    plt.close()


def write_magnitude(df: pd.DataFrame, output_dir: Path) -> None:
    methods = list(SWEEP_METHODS)
    df = df[
        df["dataset"].eq("ETTh1")
        & df["model"].eq("DLinear")
        & df["horizon"].eq(720)
        & df["shift_type"].isin(SHIFT_TYPES)
        & df["magnitude"].isin([1.0, 2.0, 3.0])
        & df["method"].isin(methods)
    ].copy()
    df = dedupe_latest(df)
    df["method_label"] = df["method"].map(SWEEP_METHODS)

    required = {(shift, magnitude, method) for shift in SHIFT_TYPES for magnitude in [1.0, 2.0, 3.0] for method in methods}
    actual = set(zip(df["shift_type"], df["magnitude"], df["method"]))
    missing = sorted(required - actual)
    if missing:
        raise RuntimeError(f"Missing magnitude rows: {missing}")

    long_cols = ["dataset", "model", "horizon", "shift_type", "magnitude", "method", "mse_after_50"]
    long_path = available_path(output_dir / "table_etth1_blackswan_magnitude_sweep_h720_long_v2.csv")
    md_path = available_path(output_dir / "table_etth1_blackswan_magnitude_sweep_h720_v2.md")
    plot_path = available_path(output_dir / "figure_etth1_blackswan_magnitude_sweep_h720_v2.png")
    plot_pdf_path = available_path(output_dir / "figure_etth1_blackswan_magnitude_sweep_h720_v2.pdf")

    df[long_cols].sort_values(["shift_type", "magnitude", "method"]).to_csv(long_path, index=False)
    md_path.write_text(to_markdown_table(df[long_cols].sort_values(["shift_type", "magnitude", "method"])), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for ax, shift_type in zip(axes.ravel(), SHIFT_TYPES):
        subset = df[df["shift_type"].eq(shift_type)]
        for method, label in SWEEP_METHODS.items():
            series = subset[subset["method"].eq(method)].sort_values("magnitude")
            if series.empty:
                continue
            ax.plot(series["magnitude"], series["mse_after_50"], marker="o", label=label)
        ax.set_title(shift_type)
        ax.set_xlabel("Magnitude")
        ax.set_ylabel("MSE after 50")
        ax.grid(alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(SWEEP_METHODS), frameon=False)
    fig.suptitle("ETTh1 DLinear H720 magnitude sensitivity")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(plot_path, dpi=200)
    fig.savefig(plot_pdf_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_rows([Path(item) for item in args.input_dir])
    if args.experiment == "ablation":
        write_ablation(df, output_dir)
    else:
        write_magnitude(df, output_dir)


if __name__ == "__main__":
    main()

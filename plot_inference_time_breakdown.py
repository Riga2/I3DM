import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_summary(summary_path):
    with open(summary_path, "r") as f:
        rows = json.load(f)
    if not rows:
        raise ValueError(f"No rows found in {summary_path}")
    return sorted(rows, key=lambda row: row["memory_size_before"])


def infer_generated_frames(rows, args):
    memory_sizes = np.array([row["memory_size_before"] for row in rows], dtype=np.float64)
    clip_ids = ((memory_sizes - args.initial_memory_size) / float(args.memory_add_per_clip)) + 1.0
    clip_ids = np.rint(clip_ids).astype(np.int64)
    return args.first_clip_frames + args.next_clip_frames * (clip_ids - 1)


def select_representative_indices(generated_frames, max_bars, target_frames):
    if target_frames:
        selected = []
        for target in target_frames:
            selected.append(int(np.argmin(np.abs(generated_frames - target))))
        return sorted(set(selected))

    if len(generated_frames) <= max_bars:
        return list(range(len(generated_frames)))
    return sorted(set(np.rint(np.linspace(0, len(generated_frames) - 1, max_bars)).astype(int).tolist()))


def plot_breakdown(args):
    rows = load_summary(args.summary_path)
    generated_frames_all = infer_generated_frames(rows, args)
    selected = select_representative_indices(generated_frames_all, args.max_bars, args.target_frames)
    rows = [rows[i] for i in selected]
    generated_frames = generated_frames_all[selected]

    retrieval = np.array([row["retrieval_time_mean"] for row in rows], dtype=np.float64)
    dit = np.array([row["dit_time_mean"] for row in rows], dtype=np.float64)
    add = np.array([row["add_time_mean"] for row in rows], dtype=np.float64)
    total = np.array([row["total_clip_time_mean"] for row in rows], dtype=np.float64)
    other = np.maximum(total - retrieval - dit, 0.0)
    retrieval_ratio = retrieval / np.maximum(total, 1e-8) * 100.0

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=args.dpi)
    x = np.arange(len(rows))
    width = 0.62

    colors = {
        "dit": "#9aa0a6",
        "retrieval": "#1f77b4",
        "other": "#def0be",
    }

    ax.bar(x, dit, width, color=colors["dit"], label="DiT generation")
    ax.bar(x, retrieval, width, bottom=dit, color=colors["retrieval"], label="Memory retrieval")
    ax.bar(x, other, width, bottom=dit + retrieval, color=colors["other"], label="Other overhead")

    top = dit + retrieval + other
    for i, (bar_top, ratio, ret_time) in enumerate(zip(top, retrieval_ratio, retrieval)):
        ax.annotate(
            f"{ratio:.1f}%\n({ret_time:.1f}s)",
            xy=(x[i], bar_top),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=colors["retrieval"],
            fontsize=8.5,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": colors["retrieval"],
                "linewidth": 0.55,
                "alpha": 0.88,
            },
        )

    ax.set_title(args.title)
    ax.set_xlabel("Generated frames")
    ax.set_ylabel("Per-clip inference time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in generated_frames])
    ax.set_ylim(0, top.max() * 1.18)
    ax.grid(True, axis="y", linestyle="--", alpha=0.28)
    ax.legend(loc="upper left", frameon=False, ncol=3)

    if args.subtitle:
        ax.text(
            0.0,
            -0.18,
            args.subtitle,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="#4d5663",
        )

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    fig.savefig(args.output_path, bbox_inches="tight")
    print(f"Saved {args.output_path}")


def parse_target_frames(value):
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot stacked inference-time breakdown against generated frames.")
    parser.add_argument(
        "--summary-path",
        default="results/memory_retrieval_scaling/memory_scaling_full_optimized_summary.json",
    )
    parser.add_argument(
        "--output-path",
        default="results/memory_retrieval_scaling/inference_time_breakdown.png",
    )
    parser.add_argument("--title", default="Inference-Time Breakdown as Video Length Increases")
    parser.add_argument(
        "--subtitle",
        default="Labels show memory retrieval share of total per-clip inference time.",
    )
    parser.add_argument("--first-clip-frames", type=int, default=77)
    parser.add_argument("--next-clip-frames", type=int, default=76)
    parser.add_argument("--initial-memory-size", type=int, default=4)
    parser.add_argument("--memory-add-per-clip", type=int, default=20)
    parser.add_argument("--max-bars", type=int, default=6)
    parser.add_argument("--target-frames", type=parse_target_frames, default="")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


if __name__ == "__main__":
    plot_breakdown(parse_args())

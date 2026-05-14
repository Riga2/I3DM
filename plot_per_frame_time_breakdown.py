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


def infer_clip_ids(rows, initial_memory_size, memory_add_per_clip):
    memory_sizes = np.array([row["memory_size_before"] for row in rows], dtype=np.float64)
    clip_ids = ((memory_sizes - initial_memory_size) / float(memory_add_per_clip)) + 1.0
    return np.rint(clip_ids).astype(np.int64)


def select_representative_indices(generated_frames, max_bars, target_frames):
    if target_frames:
        selected = []
        for target in target_frames:
            selected.append(int(np.argmin(np.abs(generated_frames - target))))
        return sorted(set(selected))

    if len(generated_frames) <= max_bars:
        return list(range(len(generated_frames)))
    return sorted(set(np.rint(np.linspace(0, len(generated_frames) - 1, max_bars)).astype(int).tolist()))


def plot_per_frame_breakdown(args):
    rows = load_summary(args.summary_path)
    clip_ids_all = infer_clip_ids(rows, args.initial_memory_size, args.memory_add_per_clip)
    generated_frames_all = args.first_clip_frames + args.next_clip_frames * (clip_ids_all - 1)

    selected = select_representative_indices(generated_frames_all, args.max_bars, args.target_frames)
    rows = [rows[i] for i in selected]
    clip_ids = clip_ids_all[selected]
    generated_frames = generated_frames_all[selected]

    effective_frames = np.where(clip_ids == 1, args.first_clip_frames, args.next_clip_frames).astype(np.float64)

    retrieval = np.array([row["retrieval_time_mean"] for row in rows], dtype=np.float64)
    dit = np.array([row["dit_time_mean"] for row in rows], dtype=np.float64)
    total = np.array([row["total_clip_time_mean"] for row in rows], dtype=np.float64)
    other = np.maximum(total - retrieval - dit, 0.0)

    retrieval_pf = retrieval / effective_frames
    dit_pf = dit / effective_frames
    other_pf = other / effective_frames
    total_pf = total / effective_frames
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

    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=args.dpi)
    x = np.arange(len(rows))
    width = 0.62

    colors = {
        "dit": "#9aa0a6",
        "retrieval": "#1f77b4",
        "other": "#f2c14e",
    }

    ax.bar(x, dit_pf, width, color=colors["dit"], label="DiT Generation")
    ax.bar(x, retrieval_pf, width, bottom=dit_pf, color=colors["retrieval"], label="Memory Retrieval")
    ax.bar(x, other_pf, width, bottom=dit_pf + retrieval_pf, color=colors["other"], label="Other Overhead")

    for i, (top, ratio, ret_pf) in enumerate(zip(total_pf, retrieval_ratio, retrieval_pf)):
        ax.annotate(
            f"{ratio:.1f}%\n{ret_pf:.2f}s",
            xy=(x[i], top),
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
                "alpha": 0.9,
            },
        )

    ax.set_title(args.title)
    ax.set_xlabel("Generated frames")
    ax.set_ylabel("Inference time per frame (s)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in generated_frames])
    ax.set_ylim(0, total_pf.max() * 1.2)
    ax.grid(True, axis="y", linestyle="--", alpha=0.28)
    ax.legend(loc="upper left", frameon=False, ncol=3, columnspacing=1.2, handlelength=1.6)

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
    parser = argparse.ArgumentParser(description="Plot per-frame inference-time breakdown with retrieval ratio labels.")
    parser.add_argument(
        "--summary-path",
        default="results/memory_retrieval_scaling/memory_scaling_full_optimized_summary.json",
    )
    parser.add_argument(
        "--output-path",
        default="results/memory_retrieval_scaling/per_frame_time_breakdown.png",
    )
    parser.add_argument("--title", default="Per-Frame Inference Time Cost as Video Length Increases")
    parser.add_argument(
        "--subtitle",
        default="Labels show memory retrieval share of total inference time and per-frame retrieval seconds.",
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
    plot_per_frame_breakdown(parse_args())

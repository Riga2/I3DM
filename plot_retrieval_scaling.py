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


def plot_retrieval_scaling(args):
    rows = load_summary(args.summary_path)
    clip_ids = infer_clip_ids(
        rows,
        initial_memory_size=args.initial_memory_size,
        memory_add_per_clip=args.memory_add_per_clip,
    )
    generated_frames = args.first_clip_frames + args.next_clip_frames * (clip_ids - 1)

    retrieval_time = np.array([row["retrieval_time_mean"] for row in rows], dtype=np.float64)
    retrieval_time_std = np.array([row["retrieval_time_std"] for row in rows], dtype=np.float64)
    total_time = np.array([row["total_clip_time_mean"] for row in rows], dtype=np.float64)
    retrieval_ratio = retrieval_time / np.maximum(total_time, 1e-8) * 100.0

    fig, ax_time = plt.subplots(figsize=(7.2, 4.6), dpi=args.dpi)
    ax_ratio = ax_time.twinx()

    time_color = "#1f77b4"
    ratio_color = "#d62728"

    ax_time.plot(
        generated_frames,
        retrieval_time,
        color=time_color,
        marker="o",
        linewidth=2.0,
        markersize=4.5,
        label="Retrieval time",
    )
    if args.show_std:
        ax_time.fill_between(
            generated_frames,
            np.maximum(retrieval_time - retrieval_time_std, 0.0),
            retrieval_time + retrieval_time_std,
            color=time_color,
            alpha=0.16,
            linewidth=0,
        )

    ax_ratio.plot(
        generated_frames,
        retrieval_ratio,
        color=ratio_color,
        marker="s",
        linestyle="--",
        linewidth=2.0,
        markersize=4.2,
        label="Retrieval ratio",
    )

    ax_time.set_xlabel("Generated frames")
    ax_time.set_ylabel("Retrieval time (s)", color=time_color)
    ax_ratio.set_ylabel("Retrieval / total time (%)", color=ratio_color)
    ax_time.tick_params(axis="y", labelcolor=time_color)
    ax_ratio.tick_params(axis="y", labelcolor=ratio_color)

    ax_time.grid(True, axis="both", linestyle="--", alpha=0.28)
    ax_time.set_title(args.title)
    ax_time.set_xlim(generated_frames.min() - 10, generated_frames.max() + 10)
    ax_time.set_ylim(bottom=0)
    ax_ratio.set_ylim(bottom=0)

    lines_1, labels_1 = ax_time.get_legend_handles_labels()
    lines_2, labels_2 = ax_ratio.get_legend_handles_labels()
    ax_time.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", frameon=False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    fig.savefig(args.output_path, bbox_inches="tight")
    print(f"Saved {args.output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot retrieval time and ratio against generated frames.")
    parser.add_argument(
        "--summary-path",
        default="results/memory_retrieval_scaling/memory_scaling_full_optimized_summary.json",
    )
    parser.add_argument(
        "--output-path",
        default="results/memory_retrieval_scaling/retrieval_scaling.png",
    )
    parser.add_argument("--title", default="Retrieval Cost Growth with Generated Video Length")
    parser.add_argument("--first-clip-frames", type=int, default=77)
    parser.add_argument("--next-clip-frames", type=int, default=76)
    parser.add_argument("--initial-memory-size", type=int, default=4)
    parser.add_argument("--memory-add-per-clip", type=int, default=20)
    parser.add_argument("--show-std", action="store_true")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


if __name__ == "__main__":
    plot_retrieval_scaling(parse_args())


import logging
import os
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from main import Chunk, load_data, Hyperparameters, Params, transient_detector_j

COLOR_PRIMARY = "#5396D0"
COLOR_ACCENT = "#9025B1"


def figure_1_chunk_with_labels(name: str, chunk: Chunk) -> None:
    # Plot chunk audio data and its labels
    fig, ax = plt.subplots(figsize=(12, 4))
    sample_rate = 48000  # FORCE_SAMPLE_RATE from main.py
    time_axis = np.arange(len(chunk.audio)) / sample_rate
    ax.plot(time_axis, chunk.audio, label="Audio", color=COLOR_PRIMARY)
    ax.plot(
        time_axis, chunk.labels * chunk.audio.max(), label="Labels", color=COLOR_ACCENT
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Chunk Audio and Labels")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"./figures/{name}/figure_1.png")


def figure_2_chunk_with_labels_and_inset(name: str, chunk: Chunk) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    sample_rate = 48000
    time_axis = np.arange(len(chunk.audio)) / sample_rate
    ax.plot(time_axis, chunk.audio, label="Audio", color=COLOR_PRIMARY)
    ax.plot(
        time_axis, chunk.labels * chunk.audio.max(), label="Labels", color=COLOR_ACCENT
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Chunk Audio and Labels (Inset)")
    ax.legend()
    # Inset around first transient
    if chunk.transient_times_sec.size > 0:
        t0 = float(chunk.transient_times_sec[0])
        window = 0.05  # seconds to show around transient
        left = max(t0 - window / 2, 0)
        right = min(t0 + window / 2, time_axis[-1])
        idx_left = int(left * sample_rate)
        idx_right = int(right * sample_rate)
        # Draw rectangle on main axis to indicate inset region
        rect = patches.Rectangle(
            (left, ax.get_ylim()[0]),
            right - left,
            ax.get_ylim()[1] - ax.get_ylim()[0],
            linewidth=2,
            edgecolor=COLOR_ACCENT,
            facecolor="none",
            linestyle="--",
            alpha=0.7,
        )
        ax.add_patch(rect)
        inset = inset_axes(ax, width="30%", height="60%", loc="lower right")
        inset.plot(
            time_axis[idx_left:idx_right],
            chunk.audio[idx_left:idx_right],
            color=COLOR_PRIMARY,
        )
        inset.plot(
            time_axis[idx_left:idx_right],
            chunk.labels[idx_left:idx_right] * chunk.audio.max(),
            color=COLOR_ACCENT,
        )
        inset.set_xlim(left, right)
        inset.set_ylim(ax.get_ylim())
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title("Zoom on 1st Transient", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"./figures/{name}/figure_2.png")


def main():
    logging.basicConfig(level=logging.INFO)
    random.seed(42)
    plt.style.use("./mystyle.mplstyle")

    base_parameters = Hyperparameters()

    # Load data
    chunks = load_data(base_parameters)

    random.shuffle(chunks)

    train_chunks = chunks[: base_parameters.train_dataset_size]
    validation_chunks = chunks[
        base_parameters.train_dataset_size : base_parameters.train_dataset_size
        + base_parameters.val_dataset_size
    ]

    print(train_chunks[2])

    for i, chunk in enumerate(train_chunks):
        print(f"{i}/{len(train_chunks)}")
        os.makedirs(f"./figures/train_{i}", exist_ok=True)
        # figure_1_chunk_with_labels(f'train_{i}', chunk)
        figure_2_chunk_with_labels_and_inset(f"train_{i}", chunk)


if __name__ == "__main__":
    main()

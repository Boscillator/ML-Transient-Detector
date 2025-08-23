import json
import logging
import os
from pathlib import Path
import random
import sys
import matplotlib
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from main import (
    Chunk,
    load_data,
    Hyperparameters,
    Params,
    transient_detector_j,
    ResultsSummary,
    EvaluationResult,
)

COLOR_PRIMARY = "#5396D0"
COLOR_ACCENT = "#9025B1"
COLOR_GRAY = "#d8dfe4"


def figure_1_chunk_with_labels(name: str, chunk: Chunk) -> None:
    # Plot chunk audio data and its labels
    fig, ax = plt.subplots(figsize=(12, 4))
    sample_rate = 48000  # FORCE_SAMPLE_RATE from main.py
    time_axis = np.arange(len(chunk.audio)) / sample_rate
    ax.plot(time_axis, chunk.audio, label="Audio", color=COLOR_PRIMARY, linewidth=0.5)
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
    ax.plot(time_axis, chunk.audio, label="Audio", color=COLOR_PRIMARY, linewidth=0.5)
    ax.plot(
        time_axis, chunk.labels * chunk.audio.max(), label="Labels", color=COLOR_ACCENT
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
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
            alpha=0.7,
            linestyle="dotted",
        )
        ax.add_patch(rect)
        inset = inset_axes(ax, width="30%", height="60%", loc="lower right")
        inset.plot(
            time_axis[idx_left:idx_right],
            chunk.audio[idx_left:idx_right],
            color=COLOR_PRIMARY,
            linewidth=0.5,
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


def figure_3_plot_with_prediction(
    name: str, chunk: Chunk, prediction: jnp.ndarray
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    sample_rate = 48000
    time_axis = np.arange(len(chunk.audio)) / sample_rate
    ax.plot(time_axis, chunk.audio, label="Audio", color=COLOR_PRIMARY, linewidth=0.5)
    ax.plot(
        time_axis, chunk.labels * chunk.audio.max(), label="Labels", color=COLOR_ACCENT
    )
    ax.plot(
        time_axis,
        prediction,
        label="Prediction",
        color=COLOR_ACCENT,
        linewidth=1.0,
        linestyle="dotted",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude / Prediction")
    ax.legend()
    # Inset around first transient
    if chunk.transient_times_sec.size > 0:
        t0 = float(chunk.transient_times_sec[0])
        window = 0.1  # seconds to show around transient
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
            alpha=0.7,
            linestyle="dotted",
        )
        ax.add_patch(rect)
        inset = inset_axes(ax, width="30%", height="60%", loc="lower right")
        inset.plot(
            time_axis[idx_left:idx_right],
            chunk.audio[idx_left:idx_right],
            color=COLOR_PRIMARY,
            linewidth=0.5,
        )
        inset.plot(
            time_axis[idx_left:idx_right],
            chunk.labels[idx_left:idx_right] * chunk.audio.max(),
            color=COLOR_ACCENT,
        )
        inset.plot(
            time_axis[idx_left:idx_right],
            prediction[idx_left:idx_right],
            color=COLOR_ACCENT,
            linewidth=1.0,
            linestyle="dotted",
        )
        inset.set_xlim(left, right)
        inset.set_ylim(ax.get_ylim())
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title("Zoom on 1st Transient", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"./figures/{name}/figure_3.png")


def load_summary(path: Path) -> ResultsSummary:
    with open(path, "r") as f:
        summary_dict = json.load(f)

    def from_serializable(cls, obj):
        # Recursively reconstruct dataclasses and arrays
        if cls is Hyperparameters:
            return Hyperparameters(**obj)
        elif cls is Params:
            # Convert lists to jnp arrays for array fields
            return Params(
                window_size_sec=jnp.array(obj["window_size_sec"]),
                weights=jnp.array(obj["weights"]),
                filter_f0s=jnp.array(obj["filter_f0s"]),
                filter_qs=jnp.array(obj["filter_qs"]),
                bias=obj["bias"],
                post_gain=obj["post_gain"],
                post_bias=obj["post_bias"],
                compressor_window_size_sec=obj["compressor_window_size_sec"],
                compressor_gain=obj["compressor_gain"],
            )
        elif cls is EvaluationResult:
            return EvaluationResult(**obj)
        elif cls is ResultsSummary:
            # Only keep EvaluationResult objects in lists
            tr = [
                from_serializable(EvaluationResult, r) for r in obj["training_results"]
            ]
            tr = [r for r in tr if isinstance(r, EvaluationResult)]
            vr = [
                from_serializable(EvaluationResult, r)
                for r in obj["validation_results"]
            ]
            vr = [r for r in vr if isinstance(r, EvaluationResult)]
            return ResultsSummary(
                hyperparameters=from_serializable(
                    Hyperparameters, obj["hyperparameters"]
                ),
                parameters=from_serializable(Params, obj["parameters"]),
                training_results=tr,
                validation_results=vr,
            )
        else:
            return obj

    result = from_serializable(ResultsSummary, summary_dict)
    if not isinstance(result, ResultsSummary):
        raise TypeError("Deserialized object is not a ResultsSummary")
    return result


def main(model_path):
    logging.basicConfig(level=logging.INFO)
    random.seed(42)
    jax.config.update("jax_platform_name", "cpu")

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

    # Load model
    results = load_summary(model_path)
    print(results)

    for i, chunk in enumerate(train_chunks):
        print(f"{i}/{len(train_chunks)}")
        os.makedirs(f"./figures/train_{i}", exist_ok=True)
        figure_1_chunk_with_labels(f"train_{i}", chunk)
        figure_2_chunk_with_labels_and_inset(f"train_{i}", chunk)

        predictions, aux = transient_detector_j(
            results.hyperparameters,
            results.parameters,
            jnp.array(chunk.audio),
            is_training=False,
            return_aux=True,
        )

        figure_3_plot_with_prediction(f"train_{i}", chunk, predictions)


if __name__ == "__main__":
    main(sys.argv[1])

"""
Plotting and visualization routines for transient detection.
"""

from __future__ import annotations

import os
from typing import List, Optional


import logging
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from data import TransientExample
from evaluation import detect_events_from_prediction
from model import transient_detector, ExperimentHyperparameters

logger = logging.getLogger(__name__)


def _plot_audio_label(
    audio: np.ndarray,
    label: np.ndarray,
    sr: int,
    duration_s: float,
    title: Optional[str] = None,
):
    """Create a figure and plot audio and label arrays. Returns (fig, ax1, ax2, t, max_samples)."""
    max_samples = min(len(audio), int(duration_s * sr))
    t = np.arange(max_samples) / sr
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(t, audio[:max_samples], label="Audio", color="C0", linewidth=0.8)
    ax1.set_ylabel("Audio")
    ax1.set_xlabel("Time (s)")
    ax2 = ax1.twinx()
    ax2.plot(
        t, label[:max_samples], label="Label", color="C1", alpha=0.5, linewidth=1.5
    )
    ax2.set_ylabel("Label (0/1)")
    if title:
        ax1.set_title(title)
    ax1.set_xlim(0, duration_s)
    return fig, ax1, ax2, t, max_samples


def _save_figure(fig, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_transient_example(
    example: TransientExample, out_path: str, duration_s: float = 1.0
) -> None:
    """Plot the audio and label array of a TransientExample and save to out_path. Plots up to duration_s seconds."""
    fig, ax1, ax2, t, max_samples = _plot_audio_label(
        example.audio,
        example.label_array,
        example.sample_rate,
        duration_s,
        title=f"Audio and Transient Label (First {duration_s} Second(s))",
    )
    _save_figure(fig, out_path)


def plot_chunk_with_prediction(
    chunk: TransientExample,
    prediction: jnp.ndarray,
    out_path: str = "data/plots/chunk_with_prediction.png",
    duration_s: float = 1.0,
    hyperparams: Optional["ExperimentHyperparameters"] = None,
) -> None:
    """Plot the audio, label, and prediction for a chunk."""
    fig, ax1, ax2, t, max_samples = _plot_audio_label(
        chunk.audio,
        chunk.label_array,
        chunk.sample_rate,
        duration_s,
        title=f"Audio, Label, and Prediction (First {duration_s} Second(s))",
    )
    # Add prediction
    ax2.plot(
        t,
        np.asarray(prediction[:max_samples]),
        label="Prediction",
        color="C3",
        alpha=0.7,
        linewidth=1.5,
    )
    # Plot vertical lines for true transient times (ground truth)
    for i, tt in enumerate(chunk.transient_times):
        if 0 <= tt <= duration_s:
            ax2.axvline(
                tt,
                color="g",
                linestyle="--",
                alpha=0.7,
                label="True Event" if i == 0 else None,
            )
    # Plot vertical lines for predicted transient times
    window_s = hyperparams.window_s if hyperparams is not None else 0.04
    pred_times = detect_events_from_prediction(
        np.asarray(prediction), 0.5, chunk.sample_rate, window_s
    )
    for i, pt in enumerate(pred_times):
        if 0 <= pt <= duration_s:
            ax2.axvline(
                pt,
                color="r",
                linestyle=":",
                alpha=0.7,
                label="Pred Event" if i == 0 else None,
            )
    # Remove duplicate legend labels
    handles, labels = ax2.get_legend_handles_labels()
    from collections import OrderedDict

    by_label = OrderedDict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc="upper right")
    _save_figure(fig, out_path)


def plot_chunks(
    chunks: List[TransientExample], out_dir: str = "data/plots/chunks"
) -> None:
    """Plot the first 5 chunks using plot_transient_example."""
    os.makedirs(out_dir, exist_ok=True)
    for i, chunk in enumerate(chunks[:5]):
        out_path = os.path.join(out_dir, f"chunk_{i + 1}.png")
        plot_transient_example(chunk, out_path)
    logger.info(f"Plotted {min(5, len(chunks))} chunks to {out_dir}")


def plot_predictions(
    chunks,
    params,
    output_dir,
    is_training,
    hyperparams,
    do_debug=True,
    prefix="",
    print_prefix="",
):
    os.makedirs(output_dir, exist_ok=True)
    for i, chunk in enumerate(chunks[:5]):
        audio_jnp = jnp.asarray(chunk.audio)
        pred = transient_detector(
            params,
            audio_jnp,
            chunk.sample_rate,
            do_debug=do_debug,
            is_training=is_training,
            hyperparams=hyperparams,
        )
        out_path = f"{output_dir}/{prefix}chunk_{i + 1}_pred.png"
        plot_chunk_with_prediction(
            chunk,
            pred,
            out_path,
            duration_s=min(5.0, len(chunk.audio) / chunk.sample_rate),
            hyperparams=hyperparams,
        )
        logger.info(f"{print_prefix}Plotted prediction for chunk {i + 1} to {out_path}")

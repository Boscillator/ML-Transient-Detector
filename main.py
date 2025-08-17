import shutil
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import numpy as np
import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Hyperparameters:
    """
    Training configuration data.
    """

    data_dir: Path = field(default_factory=lambda: Path("data/export"))
    """Training directory with .wav files and Label_Tracks.txt"""

    plots_dir: Path = field(default_factory=lambda: Path("data/plots"))
    """Root directory to save plots to"""

    chunk_length_sec: float = 1.0
    """Length of snippets used for training"""

    label_width_sec: float = 0.04
    """Width of pulse generated, centered on a transient"""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Chunk:
    audio: jnp.ndarray
    """Mono audio data, scaled to +/- 1.0"""

    sample_rate: int
    """Sample rate of the audio data"""

    labels: jnp.ndarray
    """Labels for the audio data, 1.0 at transients with width :ref:`label_width_sec`, 0.0 elsewhere"""

    transient_times_sec: jnp.ndarray
    """Times of transients in seconds, relative to the start of the chunk"""


def plot_chunk(
    hyperparameters: Hyperparameters,
    folder: str,
    title: str,
    chunk: Chunk,
    show_labels: bool = True,
    show_transients: bool = True,
):
    """
    Plots a chunk, saves to `{plots_dir}/{folder}/{title}.png`. Shows wav data and optionally other data.
    """

    os.makedirs(hyperparameters.plots_dir / folder, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(chunk.audio, label="Audio")
    if show_labels:
        plt.plot(chunk.labels, label="Labels")
    if show_transients:
        plt.vlines(
            chunk.transient_times_sec * chunk.sample_rate,
            -1,
            1,
            color="r",
            label="Transients",
        )
    plt.title(title)
    plt.legend()
    plt.ylim([-1.1, 1.1])
    plt.savefig(hyperparameters.plots_dir / folder / f"{title}.png")
    plt.close()


def load_data(
    hyperparameters: Hyperparameters, filter: Optional[Set[str]] = None
) -> List[Chunk]:
    """Loads audio data, breaks into chunks and generates transient signal"""
    label_file = hyperparameters.data_dir / "Label_Tracks.txt"
    # Read label file
    labels_dict = {}
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            track_name = parts[0].replace('"', "").replace("_Labels", "")
            transient_time = float(parts[1])
            labels_dict.setdefault(track_name, []).append(transient_time)

    chunks = []
    for track_name, transient_times in labels_dict.items():
        wav_path = hyperparameters.data_dir / f"{track_name}.wav"
        if filter is not None and track_name not in filter:
            # Skip filtered out tracks
            continue
        if not wav_path.exists():
            continue
        sample_rate, audio_np = wavfile.read(wav_path)
        # Convert to float32 and normalize to +/-1.0
        if audio_np.dtype == np.int16:
            audio_np = audio_np.astype(np.float32) / 32768.0
        elif audio_np.dtype == np.int32:
            audio_np = audio_np.astype(np.float32) / 2147483648.0
        elif audio_np.dtype == np.uint8:
            audio_np = (audio_np.astype(np.float32) - 128) / 128.0
        else:
            audio_np = audio_np.astype(np.float32)
        # If stereo, convert to mono
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=1)
        audio = jnp.array(audio_np)

        chunk_len = int(hyperparameters.chunk_length_sec * sample_rate)
        total_len = len(audio)
        num_chunks = (total_len + chunk_len - 1) // chunk_len

        for i in range(num_chunks):
            start = i * chunk_len
            end = min((i + 1) * chunk_len, total_len)
            audio_chunk = audio[start:end]
            # Find transients in this chunk
            chunk_start_sec = start / sample_rate
            chunk_end_sec = end / sample_rate
            transients_in_chunk = [
                t for t in transient_times if chunk_start_sec <= t < chunk_end_sec
            ]
            # Make transients relative to chunk start
            transients_rel = [t - chunk_start_sec for t in transients_in_chunk]
            # Generate label signal
            labels = jnp.zeros(end - start, dtype=jnp.float32)
            label_width = hyperparameters.label_width_sec
            for t_rel in transients_rel:
                center = int(t_rel * sample_rate)
                half_width = int((label_width * sample_rate) / 2)
                left = max(center - half_width, 0)
                right = min(center + half_width, end - start)
                labels = labels.at[left:right].set(1.0)
            chunk = Chunk(
                audio=audio_chunk,
                sample_rate=sample_rate,
                labels=labels,
                transient_times_sec=jnp.array(transients_rel, dtype=jnp.float32),
            )
            chunks.append(chunk)
    return chunks


def main():
    logging.basicConfig(level=logging.INFO)
    hyperparameters = Hyperparameters()

    # Clear out plots folder
    shutil.rmtree(hyperparameters.plots_dir, ignore_errors=True)

    chunks = load_data(hyperparameters, filter={"DarkIllusion_Kick"})
    for i, chunk in enumerate(chunks):
        plot_chunk(
            hyperparameters,
            "chunks",
            f"chunk_{i}",
            chunk,
            show_labels=True,
            show_transients=True,
        )


if __name__ == "__main__":
    main()

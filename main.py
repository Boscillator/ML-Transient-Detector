import shutil
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import numpy as np
import os
import logging
import optax
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

MAX_WINDOW_SIZE = int(0.1 * 48000)


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

    label_width_sec: float = 0.01
    """Width of pulse generated, centered on a transient"""

    num_channels: int = 2
    """Number of channels to use in transient detector architecture"""


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


@jax.tree_util.register_dataclass
@dataclass
class Params:
    window_size_sec: jnp.ndarray
    """Per-channel moving average window size"""

    weights: jnp.ndarray
    """Channel weights"""

    bias: float
    """Channel sum bias"""


def plot_chunk(
    hyperparameters: Hyperparameters,
    folder: str,
    title: str,
    chunk: Chunk,
    show_labels: bool = True,
    show_transients: bool = True,
    predictions: Optional[jnp.ndarray] = None,
    channel_outputs: Optional[jnp.ndarray] = None,
    preactivation: Optional[jnp.ndarray] = None,
):
    """
    Plots a chunk, saves to `{plots_dir}/{folder}/{title}.png`. Shows wav data and optionally other data.
    """

    os.makedirs(hyperparameters.plots_dir / folder, exist_ok=True)

    if channel_outputs is not None:
        fig, (ax_main, ax_channels) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    else:
        fig, ax_main = plt.subplots(1, 1, figsize=(12, 6), sharex=True)

    ax_main.plot(chunk.audio, label="Audio")
    if show_labels:
        ax_main.plot(chunk.labels, label="Labels")
    if show_transients:
        ax_main.vlines(
            chunk.transient_times_sec * chunk.sample_rate,
            -1,
            1,
            color="r",
            linestyles="dotted",
            label="Transients",
        )
    if predictions is not None:
        ax_main.plot(predictions, label="Predictions")
    ax_main.set_title(title)
    ax_main.legend()
    ax_main.set_ylim((-1.1, 1.1))

    if channel_outputs is not None:
        for i in range(channel_outputs.shape[0]):
            ax_channels.plot(channel_outputs[i], label=f"Channel {i}")
        if preactivation is not None:
            ax_channels.plot(
                preactivation, label="Pre-activation", linestyle="--", color="gray"
            )
        ax_channels.set_title("Channel Outputs")
        ax_channels.legend()

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


def moving_average(
    x: jnp.ndarray,
    window_size_s: float,
    sample_rate: int,
    max_kernel_size: int = MAX_WINDOW_SIZE,
    is_training: bool = True,
) -> jnp.ndarray:
    """
    Differentiable causal moving average with a fixed-length kernel.
    Kernel weights depend smoothly on window_size_s.
    """
    window_size_samples = window_size_s * sample_rate

    idx = jnp.arange(-max_kernel_size // 2, max_kernel_size // 2)
    if is_training:
        sharpness = 1
        weights = jnp.where(
            idx <= 0, jax.nn.sigmoid(sharpness * (idx + window_size_samples)), 0.0
        )
        weights = jnp.flip(weights + 1e-8)
    else:
        weights = jnp.where((idx > -window_size_samples) & (idx <= 0), 1.0, 0.0).astype(
            jnp.float32
        )
        weights = jnp.flip(weights)

    conv_result = jnp.convolve(x, weights, mode="same")

    divisor = jnp.maximum(jnp.sum(weights), 1e-8)
    result = conv_result / divisor
    return result


def transient_detector(
    hyperparameters: Hyperparameters,
    params: Params,
    chunk: Chunk,
    is_training: bool = True,
    return_aux: bool = False,
):
    def channel(window_size_s, weight) -> jnp.ndarray:
        power = chunk.audio**2
        avg = moving_average(
            power, window_size_s, chunk.sample_rate, is_training=is_training
        )
        rms = jnp.sqrt(avg)
        weighted_rms = weight * rms
        return weighted_rms

    channel_v = jax.vmap(channel)
    channels = channel_v(params.window_size_sec, params.weights)

    pre_activation = jnp.sum(channels, axis=0) + params.bias
    result = jax.nn.sigmoid(pre_activation)
    if not return_aux:
        return result
    else:
        return result, {"channel_outputs": channels, "pre_activation": pre_activation}


transient_detector_j = jax.jit(
    transient_detector, static_argnames=("hyperparameters", "is_training", "return_aux")
)


def train(hyperparameters: Hyperparameters, chunks: List[Chunk]) -> Params:
    import optax

    # Flatten/unflatten Params for optax
    num_channels = hyperparameters.num_channels

    def loss(params, batch):
        losses = []
        for c in batch:
            predictions = transient_detector_j(
                hyperparameters, params, c, is_training=True
            )
            losses.append(
                optax.losses.sigmoid_binary_cross_entropy(predictions, c.labels).mean()
            )
        losses = jnp.array(losses)
        return jnp.sum(losses) / len(batch)

    # SGD setup
    learning_rate = 1e-2
    optimizer = optax.sgd(learning_rate)
    # Initial params
    params = Params(
        window_size_sec=jnp.ones(num_channels) * 0.01,
        weights=jnp.ones(num_channels),
        bias=0.0,
    )

    opt_state = optimizer.init(params)

    # Training loop
    batch_size = min(2, len(chunks))
    num_steps = 100
    for step in range(num_steps):
        # Simple batching
        batch_idx = np.random.choice(len(chunks), batch_size, replace=False)
        batch = [chunks[i] for i in batch_idx]
        loss_val, grads = jax.value_and_grad(loss)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        if step % 10 == 0:
            print(f"Step {step}: loss={loss_val}")

    return params


def main():
    logging.basicConfig(level=logging.INFO)
    hyperparameters = Hyperparameters()
    params = Params(
        window_size_sec=jnp.array([0.001, 0.01]),
        weights=jnp.array([100.0, -100.0]),
        bias=-10.0,
    )

    # Clear out plots folder
    # shutil.rmtree(hyperparameters.plots_dir, ignore_errors=True)

    # Load data
    chunks = load_data(hyperparameters, filter={"DarkIllusion_Kick"})[:1]

    # params = train(hyperparameters, chunks)

    for i, chunk in enumerate(chunks):
        logger.info("Processing chunk %d", i)
        predictions, aux = transient_detector_j(
            hyperparameters, params, chunk, is_training=True, return_aux=True
        )
        plot_chunk(
            hyperparameters,
            "chunks",
            f"chunk_{i}",
            chunk,
            show_labels=True,
            show_transients=True,
            predictions=predictions,
            channel_outputs=aux["channel_outputs"],
            preactivation=aux["pre_activation"],
        )

    def oop(params):
        return jnp.sum(
            transient_detector_j(hyperparameters, params, chunks[0], is_training=False)
        )

    print(jax.value_and_grad(oop)(params))


if __name__ == "__main__":
    jax.config.update("jax_debug_nans", True)
    main()

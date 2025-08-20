import shutil
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import numpy as np
import os
import logging
import optax
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Tuple
from filters import design_biquad_bandpass, biquad_apply, apply_fir_filter

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

    chunk_length_sec: float = 5.0
    """Length of snippets used for training"""

    label_width_sec: float = 0.01
    """Width of pulse generated, centered on a transient"""

    num_channels: int = 4
    """Number of channels to use in transient detector architecture"""

    train_dataset_size: int = 5
    """Number of chunks to include in the training dataset"""

    enable_filters: bool = True
    """Whether to apply a bandpass filter to the beginning of each channel"""

    prenormalize_audio: bool = True
    """Normalize all audio clips so their peak is at 0 dBFS"""

    optimization_method: Literal["basinhopping", "differential_evolution"] = (
        "differential_evolution"
    )


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

    filter_f0s: jnp.ndarray
    """Per channel bandpass filter center frequency"""

    filter_qs: jnp.ndarray
    """Per channel bandpass filter q"""

    bias: float
    """Channel sum bias"""

    post_gain: float
    """Multiplier for channel sum. Should be included in channel weights, but adding this really helps training."""

    post_bias: float
    """Bias for channel sum"""


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
    ax_main.set_xlim((0, 0.5 * 48000))
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

            # Normalize chunk to [-1, 1]
            if hyperparameters.prenormalize_audio:
                max_val = jnp.max(jnp.abs(audio_chunk))
                audio_chunk = audio_chunk / max_val

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
    def channel(window_size_s, weight, f0, q) -> jnp.ndarray:
        if hyperparameters.enable_filters:
            b, a = design_biquad_bandpass(f0, q, chunk.sample_rate)
            # if is_training:
            if False:
                filtered = apply_fir_filter(chunk.audio, b, a)
            else:
                filtered = biquad_apply(chunk.audio, b, a)
        else:
            filtered = chunk.audio
        power = filtered**2
        avg = moving_average(
            power, window_size_s, chunk.sample_rate, is_training=is_training
        )
        rms = jnp.sqrt(avg)
        weighted_rms = weight * rms
        return weighted_rms

    channel_v = jax.vmap(channel)
    channels = channel_v(
        params.window_size_sec, params.weights, params.filter_f0s, params.filter_qs
    )

    pre_activation = jnp.sum(channels, axis=0) + params.bias
    result = jax.nn.sigmoid(params.post_gain * pre_activation + params.post_bias)
    if not return_aux:
        return result
    else:
        return result, {"channel_outputs": channels, "pre_activation": pre_activation}


transient_detector_j = jax.jit(
    transient_detector, static_argnames=("hyperparameters", "is_training", "return_aux")
)


def optimize(hyperparameters: Hyperparameters, chunks: List[Chunk]) -> Params:
    import scipy.optimize

    num_channels = hyperparameters.num_channels

    def params_to_flat(params):
        return jnp.concatenate(
            [
                jnp.array(params.window_size_sec),
                jnp.array(params.weights),
                jnp.array(params.filter_f0s),
                jnp.array(params.filter_qs),
                jnp.array([params.bias]),
                jnp.array([params.post_gain]),
                jnp.array([params.post_bias]),
            ]
        )

    def flat_to_params(flat):
        window_size_sec = jnp.array(flat[:num_channels])
        weights = jnp.array(flat[num_channels : 2 * num_channels])
        filter_f0s = jnp.array(flat[2 * num_channels : 3 * num_channels])
        filter_qs = jnp.array(flat[3 * num_channels : 4 * num_channels])
        bias = flat[4 * num_channels]
        post_gain = flat[4 * num_channels + 1]
        post_bias = flat[4 * num_channels + 2]
        return Params(
            window_size_sec=window_size_sec,
            weights=weights,
            filter_f0s=filter_f0s,
            filter_qs=filter_qs,
            bias=bias,
            post_gain=post_gain,
            post_bias=post_bias,
        )

    def loss(flat_params):
        params = flat_to_params(flat_params)
        losses = []
        for c in chunks:
            predictions = transient_detector_j(
                hyperparameters, params, c, is_training=True
            )
            this_loss = optax.losses.sigmoid_focal_loss(predictions, c.labels).mean()
            losses.append(this_loss)
        losses = jnp.array(losses)
        return jnp.sum(losses) / len(chunks)

    loss_and_grad = jax.value_and_grad(loss)

    def scipy_loss_and_grad(flat_params):
        val, grad = loss_and_grad(flat_params)
        return float(val), np.array(grad, dtype=np.float64)

    # Initial guess
    init_params = Params(
        window_size_sec=jnp.array([0.1] * num_channels),
        weights=jnp.array([10.0] * num_channels)
        * jnp.where(
            jnp.arange(num_channels) % 2 == 0, 1, -1
        ),  # Alternating sign helps optimizer
        filter_f0s=jnp.array([2000.0] * num_channels),
        filter_qs=jnp.array([1.0] * num_channels),
        bias=0.0,
        post_gain=10.0,
        post_bias=0.0,
    )
    x0 = params_to_flat(init_params)

    bounds = (
        [(0.0001, 0.5)] * num_channels  # window_size_sec
        + [(-200, 200)] * num_channels  # weights
        + [(20.0, 20000.0)] * num_channels  # filter_f0s (audio band)
        + [(0.1, 5.0)] * num_channels  # filter_qs (typical Q range)
        + [(-20, 20)]  # bias
        + [(0.0, 100.0)]  # post_gain
        + [(-20, 20)]  # post_bias
    )

    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "jac": True,
        "bounds": bounds,
        "options": {"maxiter": 100, "disp": True},
    }
    logger.info("Starting optimization with initial params: %s", init_params)
    if hyperparameters.optimization_method == "basinhopping":
        result = scipy.optimize.basinhopping(
            scipy_loss_and_grad,
            x0,
            minimizer_kwargs=minimizer_kwargs,
            niter=10,
            disp=True,
        )
        best_params = flat_to_params(result.x)
        logger.info("Basinhopping optimization result: %s", result)
        return best_params
    elif hyperparameters.optimization_method == "differential_evolution":
        result = scipy.optimize.differential_evolution(
            loss,
            bounds,
            callback=lambda intermediate_result=None: print(intermediate_result),
            maxiter=100,
            disp=True,
            polish=True,
        )
        best_params = flat_to_params(result.x)
        logger.info("Differential evolution optimization result: %s", result)
        return best_params
    else:
        raise ValueError(
            f"Unknown optimization_method: {hyperparameters.optimization_method}"
        )


def main():
    logging.basicConfig(level=logging.INFO)
    random.seed(42)

    hyperparameters = Hyperparameters()
    params = Params(
        window_size_sec=jnp.array([0.001, 0.01]),
        weights=jnp.array([10.0, -10.0]),
        filter_f0s=jnp.array([200.0, 2000.0]),
        filter_qs=jnp.array([1.0, 1.0]),
        bias=-10,
        post_gain=10,
        post_bias=0.0,
    )
    # params = Params(
    #     window_size_sec=jnp.array([0.2736782, 0.28159073, 0.12303665, 0.00095833]),
    #     weights=jnp.array([9.862206, 81.88036, 135.50629, 134.74747]),
    #     filter_f0s=jnp.array([15414.739, 7322.059, 14259.259, 8319.544]),
    #     filter_qs=jnp.array([1.8756838, 4.07662, 2.9322958, 3.0069635]),
    #     bias=np.float64(-1.842190948911615),
    #     post_gain=np.float64(43.4716256547256),
    #     post_bias=np.float64(-5.700840007924359),
    # )

    # Clear out plots folder
    shutil.rmtree(hyperparameters.plots_dir, ignore_errors=True)

    # Load data
    chunks = load_data(
        hyperparameters, filter={"DarkIllusion_Kick", "DarkIllusion_ElecGtr5DI"}
    )
    chunks = random.sample(chunks, hyperparameters.train_dataset_size)

    # Display pre-optimized solution
    # for i, chunk in enumerate(chunks[:10]):
    #     logger.info("Processing chunk %d", i)
    #     predictions, aux = transient_detector_j(
    #         hyperparameters, params, chunk, is_training=True, return_aux=True
    #     )
    #     plot_chunk(
    #         hyperparameters,
    #         "pre_optimized",
    #         f"chunk_{i}",
    #         chunk,
    #         show_labels=True,
    #         show_transients=True,
    #         predictions=predictions,
    #     )

    # Optimize
    params = optimize(hyperparameters, chunks)
    logger.info("Optimized params: %s", params)

    # Display post-optimized solution
    for i, chunk in enumerate(chunks):
        logger.info("Processing chunk %d", i)
        predictions, aux = transient_detector_j(
            hyperparameters, params, chunk, is_training=False, return_aux=True
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


if __name__ == "__main__":
    jax.config.update("jax_debug_nans", True)
    main()

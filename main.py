from pprint import pprint
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

FORCE_SAMPLE_RATE = 48000
MAX_WINDOW_SIZE = int(0.1 * FORCE_SAMPLE_RATE)


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

    label_front_porch: float = 0.001
    """Width of label before transient event (seconds)"""

    label_back_porch: float = 0.01
    """Width of label after transient event (seconds)"""

    num_channels: int = 2
    """Number of channels to use in transient detector architecture"""

    train_dataset_size: int = 10
    """Number of chunks to include in the training dataset"""

    enable_filters: bool = True
    """Whether to apply a bandpass filter to the beginning of each channel"""

    prenormalize_audio: bool = False
    """Normalize all audio clips so their peak is at 0 dBFS"""

    optimization_method: Literal["basinhopping", "differential_evolution"] = (
        "differential_evolution"
    )

    enable_compressor: bool = True

    ignore_window_sec: float = 0.01
    """Seconds to ignore after a detection (for evaluation)"""

    match_tolerance_sec: float = 0.01
    """Seconds to match ground truth transient (for evaluation)"""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Chunk:
    audio: jnp.ndarray
    """Mono audio data, scaled to +/- 1.0"""

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

    compressor_window_size_sec: float

    compressor_gain: float


@dataclass
class EvaluationResult:
    threshold: float
    """The threshold used for evaluation"""

    true_positives: int
    """Number of true positive detections"""

    false_positives: int
    """Number of false positive detections"""

    false_negatives: int
    """Number of false negative detections"""

    precision: float
    """Precision of the detections"""

    recall: float
    """Recall of the detections"""

    f1_score: float
    """F1 score of the detections"""


@dataclass
class ResultsSummary:
    hyperparameters: Hyperparameters
    """Hyperparameters used for this training run"""

    parameters: Params
    """Parameters used for this training run"""

    training_results: List[EvaluationResult]
    """List of evaluation results for each threshold"""

    validation_results: List[EvaluationResult]
    """List of evaluation results for each threshold"""

    def get_best_result(self) -> EvaluationResult:
        """Returns the best evaluation result based on F1 score"""
        return max(self.validation_results, key=lambda r: r.f1_score)


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
    sample_rate = FORCE_SAMPLE_RATE
    chunk_len = len(chunk.audio)
    seconds = int(np.ceil(chunk_len / sample_rate))
    for sec in range(seconds):
        start = sec * sample_rate
        end = min((sec + 1) * sample_rate, chunk_len)
        fig = None
        ax_channels = None
        if channel_outputs is not None:
            fig, (ax_main, ax_channels) = plt.subplots(
                2, 1, figsize=(12, 8), sharex=True
            )
        else:
            fig, ax_main = plt.subplots(1, 1, figsize=(12, 6), sharex=True)

        ax_main.plot(np.arange(start, end), chunk.audio[start:end], label="Audio")
        if show_labels:
            ax_main.plot(np.arange(start, end), chunk.labels[start:end], label="Labels")
        if show_transients:
            # Only plot transients in this window
            trans_samples = chunk.transient_times_sec * sample_rate
            trans_mask = (trans_samples >= start) & (trans_samples < end)
            trans_samples_in_window = trans_samples[trans_mask]
            ax_main.vlines(
                trans_samples_in_window,
                -1,
                1,
                color="r",
                linestyles="dotted",
                label="Transients",
            )
        if predictions is not None:
            ax_main.plot(
                np.arange(start, end), predictions[start:end], label="Predictions"
            )
        ax_main.set_title(f"{title}_sec{sec}")
        ax_main.legend()
        ax_main.set_xlim((start, end))
        ax_main.set_ylim((-1.1, 1.1))

        if ax_channels is not None:
            for i in range(channel_outputs.shape[0]):
                ax_channels.plot(
                    np.arange(start, end),
                    channel_outputs[i][start:end],
                    label=f"Channel {i}",
                )
            if preactivation is not None:
                ax_channels.plot(
                    np.arange(start, end),
                    preactivation[start:end],
                    label="Pre-activation",
                    linestyle="--",
                    color="gray",
                )
            ax_channels.set_title("Channel Outputs")
            ax_channels.legend()

        plt.savefig(hyperparameters.plots_dir / folder / f"{title}_sec{sec}.png")
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
        assert sample_rate == FORCE_SAMPLE_RATE, (
            f"Sample rate {sample_rate} != FORCE_SAMPLE_RATE {FORCE_SAMPLE_RATE} for {wav_path}"
        )
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

        chunk_len = int(hyperparameters.chunk_length_sec * FORCE_SAMPLE_RATE)
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
            chunk_start_sec = start / FORCE_SAMPLE_RATE
            chunk_end_sec = end / FORCE_SAMPLE_RATE
            transients_in_chunk = [
                t for t in transient_times if chunk_start_sec <= t < chunk_end_sec
            ]
            # Make transients relative to chunk start
            transients_rel = [t - chunk_start_sec for t in transients_in_chunk]
            # Generate label signal
            labels = jnp.zeros(end - start, dtype=jnp.float32)
            front_porch = int(hyperparameters.label_front_porch * FORCE_SAMPLE_RATE)
            back_porch = int(hyperparameters.label_back_porch * FORCE_SAMPLE_RATE)
            for t_rel in transients_rel:
                center = int(t_rel * FORCE_SAMPLE_RATE)
                left = max(center - front_porch, 0)
                right = min(center + back_porch, end - start)
                labels = labels.at[left:right].set(1.0)
            chunk = Chunk(
                audio=audio_chunk,
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
    audio: jnp.ndarray,
    is_training: bool = True,
    return_aux: bool = False,
):
    if hyperparameters.enable_compressor:
        compressor_env = moving_average(
            audio**2, params.compressor_window_size_sec, FORCE_SAMPLE_RATE
        )
        compressor_env = jnp.sqrt(compressor_env)
        audio = audio * (1 - compressor_env) + 1e-8

    def channel(window_size_s, weight, f0, q) -> jnp.ndarray:
        if hyperparameters.enable_filters:
            b, a = design_biquad_bandpass(f0, q, FORCE_SAMPLE_RATE)
            if is_training:
                filtered = apply_fir_filter(audio, b, a)
            else:
                filtered = biquad_apply(audio, b, a)
        else:
            filtered = audio
        power = filtered**2
        avg = moving_average(
            power, window_size_s, FORCE_SAMPLE_RATE, is_training=is_training
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


# vmapped version for batch audio
def _transient_detector_v(hyperparameters, params, audio_batch, is_training=True):
    return jax.vmap(
        lambda audio: transient_detector(
            hyperparameters, params, audio, is_training=is_training, return_aux=False
        )
    )(audio_batch)


transient_detector_v = jax.jit(
    _transient_detector_v, static_argnames=("hyperparameters", "is_training")
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
        compressor_window_size_sec = flat[4 * num_channels + 3]
        compressor_gain = flat[4 * num_channels + 4]
        return Params(
            window_size_sec=window_size_sec,
            weights=weights,
            filter_f0s=filter_f0s,
            filter_qs=filter_qs,
            bias=bias,
            post_gain=post_gain,
            post_bias=post_bias,
            compressor_window_size_sec=compressor_window_size_sec,
            compressor_gain=compressor_gain,
        )

    def pad_to_length(arr, target_len):
        pad_len = target_len - arr.shape[0]
        if pad_len > 0:
            arr = jnp.pad(arr, (pad_len, 0))
        return arr

    max_len = max(c.audio.shape[0] for c in chunks)
    audio_batch = jnp.stack([pad_to_length(c.audio, max_len) for c in chunks])
    label_batch = jnp.stack([pad_to_length(c.labels, max_len) for c in chunks])

    # Move batches to GPU if available
    device = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices()[0]
    audio_batch = jax.device_put(audio_batch, device)
    label_batch = jax.device_put(label_batch, device)

    def loss(flat_params):
        params = flat_to_params(flat_params)
        predictions_batch = transient_detector_v(
            hyperparameters, params, audio_batch, is_training=True
        )
        # Compute per-chunk loss and average
        losses = optax.losses.squared_error(predictions_batch, label_batch)
        return losses.mean(axis=(0, 1))

    loss_v = jax.jit(jax.vmap(loss, in_axes=1))

    # Vectorized loss for differential_evolution (flat_params shape: (num_params, S))
    def loss_vectorized(flat_params_batch, batch_size=32):
        flat_params_batch = jnp.asarray(flat_params_batch)
        flat_params_batch = jax.device_put(flat_params_batch, device)
        S = flat_params_batch.shape[1]
        results = []
        for i in range(0, S, batch_size):
            sub_batch = flat_params_batch[:, i : i + batch_size]
            results.append(loss_v(sub_batch))
        return jnp.concatenate(results)

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
        compressor_window_size_sec=0.01,
        compressor_gain=0.0,
    )
    # Move initial params to GPU
    x0 = jax.device_put(params_to_flat(init_params), device)

    bounds = (
        [(0.0001, 0.1)] * num_channels  # window_size_sec
        + [(-20, 20)] * num_channels  # weights
        + [(20.0, 20000.0)] * num_channels  # filter_f0s (audio band)
        + [(0.1, 5.0)] * num_channels  # filter_qs (typical Q range)
        + [(-2, 2)]  # bias
        + [(0.0, 200.0)]  # post_gain
        + [(-20, 20)]  # post_bias
        + [(0.0001, 0.5)]  # compressor_window_size_sec
        + [(0.0, 100.0)]  # compressor_gain
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

        def print_intermediate_result(intermediate_result: Any = None):
            print(intermediate_result)
            print(flat_to_params(intermediate_result.x))

        result = scipy.optimize.differential_evolution(
            loss_vectorized,
            bounds,
            callback=print_intermediate_result,
            maxiter=100,
            popsize=15,
            disp=True,
            polish=True,
            vectorized=True,
            updating="deferred",  # needed for vectorized
        )
        best_params = flat_to_params(result.x)
        logger.info("Differential evolution optimization result: %s", result)
        return best_params
    else:
        raise ValueError(
            f"Unknown optimization_method: {hyperparameters.optimization_method}"
        )


def get_predicted_transient_times(
    predictions: jnp.ndarray,
    threshold: float,
    sample_rate: int,
    ignore_window_sec: float,
) -> jnp.ndarray:
    """
    Returns predicted transient times (in seconds) from model output, using upward threshold crossings and ignore window.
    """
    above = predictions > threshold
    crossings = jnp.where((~above[:-1]) & (above[1:]))[0] + 1
    pred_times = []
    last_pred = -float("inf")
    for idx in crossings:
        t = idx / sample_rate
        if t - last_pred >= ignore_window_sec:
            pred_times.append(t)
            last_pred = t
    return jnp.array(pred_times)


def evaluate_at_threshold(
    hyperparameters: Hyperparameters,
    params: Params,
    chunks: List[Chunk],
    threshold: float,
) -> EvaluationResult:
    """
    Evaluates transient detection over a list of chunks, returning aggregate metrics.
    """
    sample_rate = FORCE_SAMPLE_RATE
    ignore_window_sec = hyperparameters.ignore_window_sec
    match_tolerance_sec = hyperparameters.match_tolerance_sec

    logger.debug("evaluate_at_threshold: start")
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for i, chunk in enumerate(chunks):
        logger.debug("evaluate_at_threshold: model prediction for chunk %d", i)
        predictions = transient_detector_j(
            hyperparameters, params, chunk.audio, is_training=False, return_aux=False
        )
        logger.debug("evaluate_at_threshold: threshold crossing detection")
        pred_times = get_predicted_transient_times(
            predictions, threshold, sample_rate, ignore_window_sec
        )

        logger.debug("evaluate_at_threshold: ground truth extraction")
        gt_times = chunk.transient_times_sec

        logger.debug("evaluate_at_threshold: matching predicted to ground truth")
        matched_gt = set()
        tp = 0
        for pt in pred_times:
            diffs = jnp.abs(gt_times - pt)
            min_diff = jnp.min(diffs) if gt_times.size > 0 else float("inf")
            if min_diff <= match_tolerance_sec:
                idx = int(jnp.argmin(diffs))
                if idx not in matched_gt:
                    tp += 1
                    matched_gt.add(idx)
        fp = len(pred_times) - tp
        fn = len(gt_times) - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn

    logger.debug("evaluate_at_threshold: metrics calculation")
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    logger.debug("evaluate_at_threshold: end")
    return EvaluationResult(
        threshold=threshold,
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=total_fn,
        precision=precision,
        recall=recall,
        f1_score=f1,
    )


def main():
    logging.basicConfig(level=logging.INFO)
    random.seed(42)

    hyperparameters = Hyperparameters()
    params = Params(
        window_size_sec=jnp.array([0.0001, 0.005]),
        weights=jnp.array([10.0, -8.0]),
        filter_f0s=jnp.array([200.0, 2000.0]),
        filter_qs=jnp.array([1.0, 1.0]),
        bias=-1.0,
        post_gain=100,
        post_bias=0.0,
        compressor_gain=1.0,
        compressor_window_size_sec=0.1,
    )

    # Clear out plots folder
    shutil.rmtree(hyperparameters.plots_dir, ignore_errors=True)

    # Load data
    chunks = load_data(
        hyperparameters, filter={"DarkIllusion_Kick", "DragMeDown_ElecGtr3DI"}
    )
    chunks = random.sample(chunks, hyperparameters.train_dataset_size)

    results = evaluate_at_threshold(hyperparameters, params, chunks, 0.7)
    pprint(results)

    # Display pre-optimized solution
    for i, chunk in enumerate(chunks[:2]):
        logger.info("Processing chunk %d", i)
        predictions, aux = transient_detector_j(
            hyperparameters, params, chunk.audio, is_training=True, return_aux=True
        )
        plot_chunk(
            hyperparameters,
            "pre_optimized",
            f"chunk_{i}",
            chunk,
            show_labels=True,
            show_transients=True,
            predictions=predictions,
            channel_outputs=aux["channel_outputs"],
            preactivation=aux["pre_activation"],
        )
    return

    # Optimize
    params = optimize(hyperparameters, chunks)
    logger.info("Optimized params: %s", params)

    # Display post-optimized solution
    for i, chunk in enumerate(chunks):
        logger.info("Processing chunk %d", i)
        predictions, aux = transient_detector_j(
            hyperparameters, params, chunk.audio, is_training=False, return_aux=True
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
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "all")
    main()

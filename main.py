from dataclasses import dataclass, astuple, field
import numpy as np
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt
import random
from typing import Dict, List, Union, Optional
import scipy.optimize
import jax.numpy as jnp
import jax


# Maximum window size for moving average (in seconds)
MAX_WINDOW_SIZE = 0.5  # seconds, must match optimizer bounds
# Assumed sample rate for kernel size (must be global for MAX_KERNEL_SIZE)
ASSUME_SAMPLE_RATE = 48000
# Kernel size for moving average (must be odd)
MAX_KERNEL_SIZE = int(round(MAX_WINDOW_SIZE * ASSUME_SAMPLE_RATE))
MAX_KERNEL_SIZE = MAX_KERNEL_SIZE + (MAX_KERNEL_SIZE + 1) % 2


@dataclass
class TransientExample:
    audio: np.ndarray  # 1D numpy array, mono, normalized to [-1.0, 1.0]
    label_array: np.ndarray  # 1D numpy array, 0.0/1.0, same length as audio
    transient_times: list  # List of transient times in seconds
    sample_rate: int  # Sample rate of the audio


def load_transient_example(
    base_path: str,
    hyperparams: "ExperimentHyperparameters",
    window_s: Optional[float] = None,
) -> TransientExample:
    """
    Given a base file path (without extension), load the wav and txt, generate label array, and return a TransientExample.
    Example: base_path='data/export/DarkIllusion_ElecGtr5DI' will load .wav and .txt
    """
    audio_path = base_path + ".wav"
    label_path = base_path + "_Labels.txt"
    audio, sr = load_wav_mono_normalized(audio_path)
    transient_times = load_transient_times(label_path)
    if window_s is None:
        window_s = hyperparams.window_s
    label_array = generate_label_array(
        transient_times, len(audio), sr, window_s=window_s, hyperparams=hyperparams
    )
    return TransientExample(
        audio=audio,
        label_array=label_array,
        transient_times=transient_times,
        sample_rate=sr,
    )


def load_transient_times(txt_path: str) -> list[float]:
    """Load transient times (in seconds) from a label .txt file (first column only) using numpy."""
    times = np.loadtxt(txt_path, usecols=0, comments="#", ndmin=1)
    return times.tolist()


def load_wav_mono_normalized(filepath: str) -> tuple[np.ndarray, int]:
    """Load a wav file, convert to mono if needed, and normalize to [-1.0, 1.0]."""
    sr, data = wavfile.read(filepath)
    if data.ndim > 1:
        data = data.mean(axis=1)  # Convert to mono
    # Normalize to [-1.0, 1.0] depending on dtype
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128) / 128.0
    else:
        data = data.astype(np.float32)
    return data, sr


def generate_label_array(
    transient_times: list[float],
    audio_length: int,
    sr: int,
    window_s: Optional[float] = None,
    hyperparams: Optional["ExperimentHyperparameters"] = None,
) -> np.ndarray:
    """
    Generate a label array (0.0/1.0) for the given transient times.
    Each transient is marked as 1.0 for ±window_s/2 around the transient time.
    """
    label = np.zeros(audio_length, dtype=np.float32)
    if window_s is None:
        if hyperparams is not None:
            window_s = hyperparams.window_s
        else:
            window_s = 0.04
    half_window = window_s / 2.0
    for t in transient_times:
        start = int(max(0, (t - half_window) * sr))
        end = int(min(audio_length, (t + half_window) * sr))
        label[start:end] = 1.0
    return label


def plot_transient_example(
    example: TransientExample, out_path: str, duration_s: float = 1.0
) -> None:
    """Plot the audio and label array of a TransientExample and save to out_path. Plots up to duration_s seconds."""
    audio = example.audio
    label = example.label_array
    sr = example.sample_rate

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
    ax1.set_title(f"Audio and Transient Label (First {duration_s} Second(s))")
    ax1.set_xlim(0, duration_s)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def chunkify_examples(
    example: "TransientExample",
    hyperparams: "ExperimentHyperparameters",
    min_length_s: Optional[float] = None,
    max_length_s: Optional[float] = None,
    overlap_s: Optional[float] = None,
) -> List["TransientExample"]:
    """
    Split a TransientExample into overlapping chunks of random length.
    Each chunk is a new TransientExample with adjusted transient_times.
    """
    audio = example.audio
    label = example.label_array
    sr = example.sample_rate
    transients = example.transient_times
    total_len = len(audio)
    chunks = []
    start_sample = 0
    if min_length_s is None:
        min_length_s = hyperparams.min_length_s
    if max_length_s is None:
        max_length_s = hyperparams.max_length_s
    if overlap_s is None:
        overlap_s = hyperparams.overlap_s
    while start_sample < total_len:
        chunk_length_s = random.uniform(min_length_s, max_length_s)
        chunk_length = int(chunk_length_s * sr)
        end_sample = min(start_sample + chunk_length, total_len)
        # Slice audio and label
        audio_chunk = audio[start_sample:end_sample]
        label_chunk = label[start_sample:end_sample]
        # Find transients in this chunk, adjust to chunk-relative time
        chunk_start_time = start_sample / sr
        chunk_end_time = end_sample / sr
        chunk_transients = [
            t - chunk_start_time
            for t in transients
            if chunk_start_time <= t < chunk_end_time
        ]
        # Create chunk TransientExample
        chunk = TransientExample(
            audio=audio_chunk,
            label_array=label_chunk,
            transient_times=chunk_transients,
            sample_rate=sr,
        )
        chunks.append(chunk)
        # Advance start_sample with overlap
        if end_sample == total_len:
            break
        start_sample = end_sample - int(overlap_s * sr)
        if start_sample < 0:
            start_sample = 0
    return chunks


def plot_chunk_with_prediction(
    chunk: TransientExample,
    prediction: jnp.ndarray,
    out_path: str = "data/plots/chunk_with_prediction.png",
    duration_s: float = 1.0,
    hyperparams: Optional["ExperimentHyperparameters"] = None,
) -> None:
    import jax.numpy as jnp

    """Plot the audio, label, and prediction for a chunk."""
    import matplotlib.pyplot as plt

    audio = chunk.audio
    label = chunk.label_array
    sr = chunk.sample_rate
    max_samples = min(len(audio), int(duration_s * sr), len(prediction))
    t = np.arange(max_samples) / sr
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(t, audio[:max_samples], label="Audio", color="C0", linewidth=0.8)
    ax1.set_ylabel("Audio")
    ax1.set_xlabel("Time (s)")
    ax2 = ax1.twinx()
    ax2.plot(
        t, label[:max_samples], label="Label", color="C1", alpha=0.5, linewidth=1.5
    )
    ax2.plot(
        t,
        np.asarray(prediction[:max_samples]),
        label="Prediction",
        color="C3",
        alpha=0.7,
        linewidth=1.5,
    )

    # Plot vertical lines for true transient times (ground truth)
    for tt in chunk.transient_times:
        if 0 <= tt <= duration_s:
            ax2.axvline(tt, color="g", linestyle="--", alpha=0.7, label="True Event" if tt == chunk.transient_times[0] else None)

    # Plot vertical lines for predicted transient times
    # Use detect_events_from_prediction with threshold 0.5 and window_s from hyperparams
    if hyperparams is not None:
        window_s = hyperparams.window_s
    else:
        window_s = 0.04
    pred_times = detect_events_from_prediction(np.asarray(prediction), 0.5, sr, window_s)
    for i, pt in enumerate(pred_times):
        if 0 <= pt <= duration_s:
            ax2.axvline(pt, color="r", linestyle=":", alpha=0.7, label="Pred Event" if i == 0 else None)

    ax2.set_ylabel("Label / Prediction")
    ax1.set_title(f"Audio, Label, and Prediction (First {duration_s} Second(s))")
    ax1.set_xlim(0, duration_s)
    handles, labels = ax2.get_legend_handles_labels()
    # Remove duplicate labels
    from collections import OrderedDict
    by_label = OrderedDict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc="upper right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def internal_debug(i: int, data: Dict[str, Union[np.ndarray, jnp.ndarray, float]]):
    import matplotlib.pyplot as plt
    import os

    os.makedirs("data/plots/debug", exist_ok=True)
    keys = list(data.keys())
    n = len(keys)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        val = data[key]
        if isinstance(val, (np.ndarray, jnp.ndarray)):
            arr = np.asarray(val)
            ax.plot(arr)
        else:
            ax.plot([val])
        ax.set_ylabel(key)
        ax.set_title(key)
    axes[-1].set_xlabel("Index")
    fig.tight_layout()
    out_path = f"data/plots/debug/{i}.png"
    plt.savefig(out_path)
    plt.close(fig)


def plot_chunks(
    chunks: List[TransientExample], out_dir: str = "data/plots/chunks"
) -> None:
    """Plot the first 5 chunks using plot_transient_example."""
    os.makedirs(out_dir, exist_ok=True)
    for i, chunk in enumerate(chunks[:5]):
        out_path = os.path.join(out_dir, f"chunk_{i + 1}.png")
        plot_transient_example(chunk, out_path)
    print(f"Plotted {min(5, len(chunks))} chunks to {out_dir}")


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
        print(f"{print_prefix}Plotted prediction for chunk {i + 1} to {out_path}")


@jax.tree_util.register_dataclass
@dataclass
class TransientDetectorParameters:
    fast_window: float = 0.01  # in seconds (can be fractional for differentiability)
    slow_window: float = 0.1  # in seconds (can be fractional for differentiability)
    w0: float = 0.0  # bias term
    w1: float = 20.0  # fast_env weight
    w2: float = -20.0  # slow_env weight
    post_gain: float = 1.0
    post_bias: float = 0.0


# Experiment-level hyperparameters dataclass
@dataclass
class ExperimentHyperparameters:
    device: str = "gpu"  # or 'cpu'
    min_length_s: float = 5.0
    max_length_s: float = 10.0
    overlap_s: float = 1.0
    window_s: float = 0.04  # for label generation
    loss_epsilon: float = 1e-7
    # Bounds for optimization (in order of TransientDetectorParameters fields)
    bounds: tuple = (
        (1e-3, 0.5),  # fast_window
        (1e-3, 0.5),  # slow_window
        (-500.0, 500.0),  # w0
        (-500.0, 500.0),  # w1
        (-500.0, 500.0),  # w2
        (-1e4, 1e4),  # post_gain
        (-1.0, 1.0),  # post_bias
    )
    detector_defaults: TransientDetectorParameters = field(
        default_factory=TransientDetectorParameters
    )


def softmax_kernel(window_size: float) -> jnp.ndarray:
    half = MAX_KERNEL_SIZE // 2
    idxs = jnp.arange(-half, half + 1)
    mask = idxs <= 0
    causal_idxs = idxs * mask

    # Causal softmax kernel: only current and past
    # Set future weights to -inf so softmax is zero there
    logits = -((causal_idxs / window_size) ** 2)
    logits = jnp.where(mask, logits, -jnp.inf)
    kernel = jax.nn.softmax(logits)
    return kernel / kernel.sum()


def rectangle_kernel(window_size: float) -> jnp.ndarray:
    half = MAX_KERNEL_SIZE // 2
    idxs = jnp.arange(-half, half + 1)

    width = jnp.clip(window_size, 1.0, MAX_KERNEL_SIZE)
    rect_mask = (idxs >= -width + 1) & (idxs <= 0)
    kernel = rect_mask.astype(jnp.float32)
    return kernel / kernel.sum()


def moving_average(
    x: jnp.ndarray, window_size: float, is_training: bool = True
) -> jnp.ndarray:
    """
    Differentiable, fractional window moving average with a softmax or rectangle kernel.
    """

    if is_training:
        kernel = softmax_kernel(window_size)
    else:
        kernel = rectangle_kernel(window_size)

    return jnp.convolve(x, kernel, mode="same")


_dbg_n = 0


def transient_detector(
    params: TransientDetectorParameters,
    audio: jnp.ndarray,
    sample_rate: float,
    do_debug: bool = False,
    is_training: bool = True,
    hyperparams: Optional["ExperimentHyperparameters"] = None,
) -> jnp.ndarray:
    # Convert window sizes from seconds to samples
    envelop1_window_samples = params.fast_window * sample_rate
    envelop2_window_samples = params.slow_window * sample_rate

    power = jnp.abs(audio)

    envelop1 = moving_average(power, envelop1_window_samples, is_training=is_training)
    envelop2 = moving_average(power, envelop2_window_samples, is_training=is_training)

    inner = params.w0 + params.w1 * envelop1 + params.w2 * envelop2
    outer = params.post_gain * inner + params.post_bias
    result = jax.nn.sigmoid(outer)

    if do_debug:
        global _dbg_n
        # Internal debug plotting
        internal_debug(
            _dbg_n,
            {
                "audio": audio,
                "power": power,
                "envelop1": envelop1,
                "envelop2": envelop2,
                "inner": inner,
                "outer": outer,
                "result": result,
            },
        )
        _dbg_n += 1

    return result


def loss_function(
    target: jnp.ndarray,
    prediction: jnp.ndarray,
    hyperparams: "ExperimentHyperparameters",
) -> jnp.ndarray:
    # Binary cross entropy loss
    eps = hyperparams.loss_epsilon
    prediction = jnp.clip(prediction, eps, 1 - eps)
    loss = -(target * jnp.log(prediction) + (1 - target) * jnp.log(1 - prediction))
    return jnp.mean(loss)


def optimize_transient_detector(
    chunks: List[TransientExample],
    hyperparams: "ExperimentHyperparameters",
) -> TransientDetectorParameters:
    # Prepare arrays for vmap
    audio_arrs = [jnp.asarray(chunk.audio) for chunk in chunks]
    label_arrs = [jnp.asarray(chunk.label_array) for chunk in chunks]
    sample_rates = [chunk.sample_rate for chunk in chunks]

    # Pad all arrays to the same length for batching
    max_len = max(len(a) for a in audio_arrs)

    def pad(arr, length):
        return jnp.pad(arr, (0, length - len(arr)), constant_values=0)

    audio_batch = jnp.stack([pad(a, max_len) for a in audio_arrs])
    label_batch = jnp.stack([pad(l, max_len) for l in label_arrs])
    sample_rate_batch = jnp.array(sample_rates)
    valid_lengths = jnp.array([len(a) for a in audio_arrs])

    # Move all batch arrays to the selected device
    device = jax.devices(hyperparams.device)[0]
    audio_batch = jax.device_put(audio_batch, device)
    label_batch = jax.device_put(label_batch, device)
    sample_rate_batch = jax.device_put(sample_rate_batch, device)
    valid_lengths = jax.device_put(valid_lengths, device)

    def chunk_loss(params_array, audio, label, sample_rate, valid_len):
        fast_window, slow_window, w0, w1, w2, post_gain, post_bias = params_array
        params = TransientDetectorParameters(
            fast_window=fast_window,
            slow_window=slow_window,
            w0=w0,
            w1=w1,
            w2=w2,
            post_gain=post_gain,
            post_bias=post_bias,
        )
        pred = transient_detector(
            params, audio, sample_rate, is_training=True, hyperparams=hyperparams
        )
        # Create mask for valid (unpadded) region
        mask = jnp.arange(label.shape[0]) < valid_len
        # Compute loss over all elements, mask out padded region
        loss = loss_function(label, pred, hyperparams)
        # Masked mean: sum only valid elements
        masked_loss = jnp.sum(loss * mask) / jnp.maximum(1, jnp.sum(mask))
        return masked_loss * valid_len, valid_len

    v_chunk_loss = jax.vmap(chunk_loss, in_axes=(None, 0, 0, 0, 0))

    @jax.jit
    def loss_for_params(param_array):
        losses, counts = v_chunk_loss(
            param_array, audio_batch, label_batch, sample_rate_batch, valid_lengths
        )
        total_loss = jnp.sum(losses)
        total_count = jnp.sum(counts)
        return total_loss / jnp.maximum(1, total_count)

    # JAX grad for the loss function
    loss_grad = jax.grad(lambda p: loss_for_params(p))

    default_params = TransientDetectorParameters()
    x0 = np.array(
        [
            default_params.fast_window,
            default_params.slow_window,
            default_params.w0,
            default_params.w1,
            default_params.w2,
            default_params.post_gain,
            default_params.post_bias,
        ],
        dtype=np.float32,
    )
    bounds = list(hyperparams.bounds)

    # Progress display callback
    def progress_callback(xk):
        print(f"Current: {TransientDetectorParameters(*xk)}")

    result = scipy.optimize.minimize(
        loss_for_params,
        x0,
        method="L-BFGS-B",
        jac=loss_grad,
        bounds=bounds,
        options={"disp": True},
        callback=progress_callback,
    )

    return TransientDetectorParameters(*result.x)

@dataclass
class EvaluationResult:
    params: TransientDetectorParameters
    loss: float
    threshold: float
    false_positives: int
    false_negatives: int
    true_positives: int
    true_negatives: int
    accuracy: float
    recall: float


def detect_events_from_prediction(
    preds_np,
    threshold: float,
    sample_rate: float,
    window_s: float,
) -> list[float]:
    """
    Detect event times from a prediction array using upward threshold crossings with latching and a refractory period.
    Returns a list of event times (in seconds).
    """
    # Ensure numpy is available for type and array ops
    import numpy as np
    refractory_s = window_s
    pred_times: list[float] = []
    last_det_time = -1e9
    went_below_since_last = True  # allow first crossing
    for i in range(1, len(preds_np)):
        t_prev = (i - 1) / sample_rate
        t_cur = i / sample_rate
        if preds_np[i] <= threshold:
            went_below_since_last = True
        if (
            preds_np[i - 1] <= threshold
            and preds_np[i] > threshold
            and went_below_since_last
            and (t_cur - last_det_time) >= refractory_s
        ):
            pred_times.append(t_cur)
            last_det_time = t_cur
            went_below_since_last = False
    return pred_times

def evaluate_model_for_threshold(
    hparams: ExperimentHyperparameters,
    params: TransientDetectorParameters,
    data: List[TransientExample],
    threshold: float,
) -> EvaluationResult:
    # Event-based evaluation: use detection function for event times
    tol_s = hparams.window_s / 2.0

    total_event_tp = 0
    total_event_fp = 0
    total_event_fn = 0

    # Sample-based confusion (for optional accuracy reporting)
    sample_tp = 0
    sample_fp = 0
    sample_tn = 0
    sample_fn = 0

    total_loss_sum = 0.0
    total_count = 0

    for ex in data:
        sr = ex.sample_rate
        preds = transient_detector(
            params,
            jnp.asarray(ex.audio),
            sr,
            do_debug=False,
            is_training=False,
            hyperparams=hparams,
        )
        preds_np = np.asarray(preds)

        # Sample-wise confusion
        y_true = ex.label_array > 0.5
        y_pred = preds_np > threshold
        sample_tp += int(np.sum(np.logical_and(y_true, y_pred)))
        sample_fp += int(np.sum(np.logical_and(~y_true, y_pred)))
        sample_tn += int(np.sum(np.logical_and(~y_true, ~y_pred)))
        sample_fn += int(np.sum(np.logical_and(y_true, ~y_pred)))

        # Loss (weighted by sample count)
        loss_val = float(loss_function(jnp.asarray(ex.label_array), preds, hparams))
        total_loss_sum += loss_val * len(ex.label_array)
        total_count += len(ex.label_array)

        # Use new detection function
        pred_times = detect_events_from_prediction(
            preds_np, threshold, sr, hparams.window_s
        )

        # Event matching to ground truth
        gt_times = list(ex.transient_times)
        matched_gt = [False] * len(gt_times)
        event_tp = 0
        event_fp = 0

        for pt in pred_times:
            # find nearest unmatched gt within tolerance
            best_j = -1
            best_dt = float("inf")
            for j, gt in enumerate(gt_times):
                if matched_gt[j]:
                    continue
                dt = abs(pt - gt)
                if dt < best_dt:
                    best_dt = dt
                    best_j = j
            if best_j != -1 and best_dt <= tol_s:
                matched_gt[best_j] = True
                event_tp += 1
            else:
                event_fp += 1

        event_fn = matched_gt.count(False)

        total_event_tp += event_tp
        total_event_fp += event_fp
        total_event_fn += event_fn

    # Aggregate metrics
    mean_loss = (total_loss_sum / max(1, total_count)) if total_count > 0 else 0.0

    # Event-based recall and a simple "accuracy" notion over events
    recall = (
        total_event_tp / max(1, (total_event_tp + total_event_fn))
        if (total_event_tp + total_event_fn) > 0
        else 0.0
    )
    accuracy = (
        total_event_tp / max(1, (total_event_tp + total_event_fp + total_event_fn))
        if (total_event_tp + total_event_fp + total_event_fn) > 0
        else 0.0
    )

    return EvaluationResult(
        params=params,
        loss=mean_loss,
        threshold=threshold,
        false_positives=total_event_fp,
        false_negatives=total_event_fn,
        true_positives=total_event_tp,
        true_negatives=0,  # TN not well-defined for event detection; leave as 0
        accuracy=accuracy,
        recall=recall,
    )


def evaluate_model(
    hparams: ExperimentHyperparameters,
    params: TransientDetectorParameters,
    data: List[TransientExample],
) -> List[EvaluationResult]:
    # Evaluate across a sweep of thresholds
    thresholds = np.linspace(0.1, 0.9, 9, dtype=np.float32)
    results: List[EvaluationResult] = []
    for th in thresholds:
        results.append(evaluate_model_for_threshold(hparams, params, data, float(th)))
    return results

def main() -> None:
    hyperparams = ExperimentHyperparameters()
    # Load and plot an example
    base_path = "data/export/DarkIllusion_ElecGtr5DI"
    example = load_transient_example(base_path, hyperparams)
    chunks = chunkify_examples(example, hyperparams)
    chunks = chunks[:5]  # Limit to first 5 chunks for testing

    params = TransientDetectorParameters()

    # Plot predictions before optimization (eval kernel)
    plot_predictions(
        chunks,
        params,
        output_dir="data/plots/chunk_preds",
        is_training=False,
        hyperparams=hyperparams,
        do_debug=True,
        prefix="",
        print_prefix="",
    )

    # Optimize parameters
    print("Optimizing transient detector parameters...")
    opt_params = optimize_transient_detector(chunks, hyperparams)
    print(f"Optimized parameters: {opt_params}")

    # Evaluate optimized model across thresholds
    eval_results = evaluate_model(hyperparams, opt_params, chunks)
    # Print concise summary
    print("Evaluation results by threshold:")
    for r in eval_results:
        print(
            f"  th={r.threshold:.2f} | loss={r.loss:.4f} | TP={r.true_positives} FP={r.false_positives} FN={r.false_negatives} | acc={r.accuracy:.3f} rec={r.recall:.3f}"
        )
    # Select best by accuracy and recall
    best_by_acc = max(eval_results, key=lambda r: r.accuracy) if eval_results else None
    best_by_rec = max(eval_results, key=lambda r: r.recall) if eval_results else None
    if best_by_acc:
        print(
            f"Best accuracy: th={best_by_acc.threshold:.2f}, acc={best_by_acc.accuracy:.3f}, rec={best_by_acc.recall:.3f}"
        )
    if best_by_rec:
        print(
            f"Best recall:   th={best_by_rec.threshold:.2f}, acc={best_by_rec.accuracy:.3f}, rec={best_by_rec.recall:.3f}"
        )

    # Plot predictions after optimization
    plot_predictions(
        chunks,
        opt_params,
        output_dir="data/plots/chunk_preds_optimized",
        is_training=False,
        hyperparams=hyperparams,
        do_debug=True,
        prefix="",
        print_prefix="Optimized: ",
    )


if __name__ == "__main__":
    main()

from dataclasses import dataclass
import numpy as np
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt
import random
from typing import Dict, List, Union
import jax.numpy as jnp
import jax

# Maximum window size for moving average (in seconds)
MAX_WINDOW_SIZE = 0.5  # seconds, must match optimizer bounds

@dataclass
class TransientExample:
    audio: np.ndarray  # 1D numpy array, mono, normalized to [-1.0, 1.0]
    label_array: np.ndarray  # 1D numpy array, 0.0/1.0, same length as audio
    transient_times: list  # List of transient times in seconds
    sample_rate: int  # Sample rate of the audio


def load_transient_example(base_path: str, window_s: float = 0.04) -> TransientExample:
    """
    Given a base file path (without extension), load the wav and txt, generate label array, and return a TransientExample.
    Example: base_path='data/export/DarkIllusion_ElecGtr5DI' will load .wav and .txt
    """
    audio_path = base_path + ".wav"
    label_path = base_path + "_Labels.txt"
    audio, sr = load_wav_mono_normalized(audio_path)
    transient_times = load_transient_times(label_path)
    label_array = generate_label_array(
        transient_times, len(audio), sr, window_s=window_s
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
    window_s: float = 0.01
) -> np.ndarray:
    """
    Generate a label array (0.0/1.0) for the given transient times.
    Each transient is marked as 1.0 for ±window_s/2 around the transient time.
    """
    label = np.zeros(audio_length, dtype=np.float32)
    half_window = window_s / 2.0
    for t in transient_times:
        start = int(max(0, (t - half_window) * sr))
        end = int(min(audio_length, (t + half_window) * sr))
        label[start:end] = 1.0
    return label


def plot_transient_example(
    example: TransientExample,
    out_path: str,
    duration_s: float = 1.0
) -> None:
    """Plot the audio and label array of a TransientExample and save to out_path. Plots up to duration_s seconds."""
    audio = example.audio
    label = example.label_array
    sr = example.sample_rate

    max_samples = min(len(audio), int(duration_s * sr))
    t = np.arange(max_samples) / sr

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(t, audio[:max_samples], label='Audio', color='C0', linewidth=0.8)
    ax1.set_ylabel('Audio')
    ax1.set_xlabel('Time (s)')
    ax2 = ax1.twinx()
    ax2.plot(t, label[:max_samples], label='Label', color='C1', alpha=0.5, linewidth=1.5)
    ax2.set_ylabel('Label (0/1)')
    ax1.set_title(f'Audio and Transient Label (First {duration_s} Second(s))')
    ax1.set_xlim(0, duration_s)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)

def chunkify_examples(
    example: "TransientExample",
    min_length_s: float = 3.0,
    max_length_s: float = 5.0,
    overlap_s: float = 1.0
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
        chunk_transients = [t - chunk_start_time for t in transients if chunk_start_time <= t < chunk_end_time]
        # Create chunk TransientExample
        chunk = TransientExample(
            audio=audio_chunk,
            label_array=label_chunk,
            transient_times=chunk_transients,
            sample_rate=sr
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
    duration_s: float = 1.0
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
    ax2.plot(t, label[:max_samples], label="Label", color="C1", alpha=0.5, linewidth=1.5)
    ax2.plot(t, np.asarray(prediction[:max_samples]), label="Prediction", color="C3", alpha=0.7, linewidth=1.5)
    ax2.set_ylabel("Label / Prediction")
    ax1.set_title(f"Audio, Label, and Prediction (First {duration_s} Second(s))")
    ax1.set_xlim(0, duration_s)
    ax2.legend(loc="upper right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)

def internal_debug(i: int, data: Dict[str, Union[np.ndarray, jnp.ndarray, float]]):
    import matplotlib.pyplot as plt
    import os
    os.makedirs('data/plots/debug', exist_ok=True)
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
    axes[-1].set_xlabel('Index')
    fig.tight_layout()
    out_path = f'data/plots/debug/{i}.png'
    plt.savefig(out_path)
    plt.close(fig)


def plot_chunks(
    chunks: List[TransientExample],
    out_dir: str = "data/plots/chunks"
) -> None:
    """Plot the first 5 chunks using plot_transient_example."""
    os.makedirs(out_dir, exist_ok=True)
    for i, chunk in enumerate(chunks[:5]):
        out_path = os.path.join(out_dir, f"chunk_{i+1}.png")
        plot_transient_example(chunk, out_path)
    print(f"Plotted {min(5, len(chunks))} chunks to {out_dir}")

@jax.tree_util.register_dataclass
@dataclass
class TransientDetectorParameters:
    fast_window: float = 0.01  # in seconds (can be fractional for differentiability)
    slow_window: float = 0.1   # in seconds (can be fractional for differentiability)
    w0: float = 0.0            # bias term
    w1: float = 20.0            # fast_env weight
    w2: float = -20.0           # slow_env weight

def moving_average(x: jnp.ndarray, window_size: float, sample_rate: float) -> jnp.ndarray:
    """
    Differentiable, fractional window moving average with a softmax kernel.
    max_win is set to cover the largest window size allowed by optimizer bounds (5.0s) at the given sample rate.
    """
    max_win = int(MAX_WINDOW_SIZE * sample_rate)
    # Ensure max_win is odd for symmetry
    if max_win % 2 == 0:
        max_win += 1
    half = max_win // 2
    idxs = jnp.arange(-half, half + 1)
    # Softmax kernel centered at 0, width controlled by window_size
    kernel = jax.nn.softmax(-((idxs / window_size) ** 2))
    kernel = kernel / kernel.sum()
    return jnp.convolve(x, kernel, mode='same')

_dbg_n = 0

def transient_detector(
    params: TransientDetectorParameters,
    audio: jnp.ndarray,
    sample_rate: float,
    do_debug: bool = False
) -> jnp.ndarray:
    # Convert window sizes from seconds to samples
    fast_window_samples = params.fast_window * sample_rate
    slow_window_samples = params.slow_window * sample_rate

    power = jnp.abs(audio)

    fast_env = moving_average(power, fast_window_samples, sample_rate)
    slow_env = moving_average(power, slow_window_samples, sample_rate)
    # New formula: sigmoid(w0 + w1 * fast_env + w2 * slow_env)
    inner = params.w0 + params.w1 * fast_env + params.w2 * slow_env
    result = jax.nn.sigmoid(inner)

    if do_debug:
        global _dbg_n
        # Internal debug plotting
        internal_debug(_dbg_n, {
            'audio': audio,
            'power': power,
            'fast_env': fast_env,
            'slow_env': slow_env,
            'inner': inner,
            'result': result
        })
        _dbg_n += 1

    return result

def loss_function(target: jnp.ndarray, prediction: jnp.ndarray) -> jnp.ndarray:
    # Binary cross entropy loss
    eps = 1e-7
    prediction = jnp.clip(prediction, eps, 1 - eps)
    loss = - (target * jnp.log(prediction) + (1 - target) * jnp.log(1 - prediction))
    return jnp.mean(loss)


def optimize_transient_detector(chunks: List[TransientExample]) -> TransientDetectorParameters:
    import scipy.optimize
    import numpy as np


    def loss_for_params(param_array):
        # param_array: [fast_window, slow_window, w0, w1, w2]
        fast_window, slow_window, w0, w1, w2 = param_array
        params = TransientDetectorParameters(fast_window=fast_window, slow_window=slow_window, w0=w0, w1=w1, w2=w2)
        total_loss = 0.0
        total_count = 0
        for chunk in chunks:
            audio_jnp = jnp.asarray(chunk.audio)
            label_jnp = jnp.asarray(chunk.label_array)
            pred = transient_detector(params, audio_jnp, chunk.sample_rate)
            # Ensure label and pred are same length
            n = min(len(label_jnp), len(pred))
            total_loss += loss_function(label_jnp[:n], pred[:n]) * n
            total_count += n
        return total_loss / max(1, total_count)

    # JAX grad for the loss function
    loss_grad = jax.grad(lambda p: loss_for_params(p))

    # Initial guess: [fast_window, slow_window, w0, w1, w2] from dataclass defaults
    default_params = TransientDetectorParameters()
    x0 = np.array([
        default_params.fast_window,
        default_params.slow_window,
        default_params.w0,
        default_params.w1,
        default_params.w2
    ], dtype=np.float32)
    bounds = [
        (1e-3, MAX_WINDOW_SIZE),  # fast_window
        (1e-3, MAX_WINDOW_SIZE),  # slow_window
        (-100.0, 100.0),              # w0 (bias)
        (-100.0, 100.0),            # w1 (fast_env weight)
        (-100.0, 100.0),            # w2 (slow_env weight)
    ]


    # Progress display callback
    def progress_callback(xk):
        print(f"Current params: fast_window={xk[0]:.4f}, slow_window={xk[1]:.4f}, w0={xk[2]:.4f}, w1={xk[3]:.4f}, w2={xk[4]:.4f}")

    result = scipy.optimize.minimize(
        loss_for_params,
        x0,
        method='L-BFGS-B',
        jac=loss_grad,
        bounds=bounds,
        options={'disp': True},
        callback=progress_callback
    )

    fast_window, slow_window, w0, w1, w2 = result.x
    return TransientDetectorParameters(fast_window=fast_window, slow_window=slow_window, w0=w0, w1=w1, w2=w2)

def main() -> None:
    # Load and plot an example
    base_path = 'data/export/DarkIllusion_ElecGtr5DI'
    example = load_transient_example(base_path)
    chunks = chunkify_examples(example)
    chunks = chunks[:5]  # Limit to first 10 chunks for testing


    # Plot predictions before optimization
    params = TransientDetectorParameters()
    os.makedirs('data/plots/chunk_preds', exist_ok=True)
    for i, chunk in enumerate(chunks[:5]):
        audio_jnp = jnp.asarray(chunk.audio)
        pred = transient_detector(params, audio_jnp, chunk.sample_rate, do_debug=True)
        out_path = f'data/plots/chunk_preds/chunk_{i+1}_pred.png'
        plot_chunk_with_prediction(chunk, pred, out_path, duration_s=min(5.0, len(chunk.audio)/chunk.sample_rate))
        print(f"Plotted prediction for chunk {i+1} to {out_path}")

    # Optimize parameters
    print("Optimizing transient detector parameters...")
    opt_params = optimize_transient_detector(chunks)
    print(f"Optimized parameters: {opt_params}")

    # Plot predictions after optimization
    os.makedirs('data/plots/chunk_preds_optimized', exist_ok=True)
    for i, chunk in enumerate(chunks[:5]):
        audio_jnp = jnp.asarray(chunk.audio)
        pred = transient_detector(opt_params, audio_jnp, chunk.sample_rate, do_debug=True)
        out_path = f'data/plots/chunk_preds_optimized/chunk_{i+1}_pred.png'
        plot_chunk_with_prediction(chunk, pred, out_path, duration_s=min(5.0, len(chunk.audio)/chunk.sample_rate))
        print(f"Plotted optimized prediction for chunk {i+1} to {out_path}")


if __name__ == "__main__":
    main()

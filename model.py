"""
Model definition, kernels, and optimization for transient detection.
"""

from __future__ import annotations


import logging
from dataclasses import dataclass, field
from typing import List, Optional, Self

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize

from data import TransientExample
from filter import design_biquad_bandpass, biquad_apply

logger = logging.getLogger(__name__)


# Maximum window size for moving average (in seconds)
MAX_WINDOW_SIZE = 0.25  # seconds, must match optimizer bounds
# Assumed sample rate for kernel size (must be global for MAX_KERNEL_SIZE)
ASSUME_SAMPLE_RATE = 48000
# Kernel size for moving average (must be odd)
MAX_KERNEL_SIZE = int(round(MAX_WINDOW_SIZE * ASSUME_SAMPLE_RATE))
MAX_KERNEL_SIZE = MAX_KERNEL_SIZE + (MAX_KERNEL_SIZE + 1) % 2


@dataclass
class TransientDetectorParameters:
    window_sizes: jnp.ndarray = field(
        default_factory=lambda: jnp.array([0.01, 0.1, 0.001], dtype=jnp.float32)
    )  # in seconds, per channel
    weights: jnp.ndarray = field(
        default_factory=lambda: jnp.array([20.0, -20.0, -10.0], dtype=jnp.float32)
    )  # per channel
    f0s: jnp.ndarray = field(
        default_factory=lambda: jnp.array([1000.0, 2000.0, 4000.0], dtype=jnp.float32)
    )  # Hz, per channel
    qs: jnp.ndarray = field(
        default_factory=lambda: jnp.array([2.0, 2.0, 2.0], dtype=jnp.float32)
    )  # Q, per channel
    bias: float = 0.0
    post_gain: float = 1.0
    post_bias: float = 0.0

    def to_array(self, hyperparams: "ExperimentHyperparameters") -> jnp.ndarray:
        arr = jnp.concatenate(
            [
                jnp.asarray(self.window_sizes, dtype=jnp.float32),
                jnp.asarray(self.weights, dtype=jnp.float32),
                jnp.asarray(self.f0s, dtype=jnp.float32),
                jnp.asarray(self.qs, dtype=jnp.float32),
                jnp.array(
                    [self.bias, self.post_gain, self.post_bias], dtype=jnp.float32
                ),
            ]
        )
        return arr

    @classmethod
    def from_array(
        cls, arr: jnp.ndarray, hyperparams: "ExperimentHyperparameters"
    ) -> Self:
        arr = jnp.asarray(arr)
        n = hyperparams.num_channels
        window_sizes = arr[:n]
        weights = arr[n : 2 * n]
        f0s = arr[2 * n : 3 * n]
        qs = arr[3 * n : 4 * n]
        bias = arr[4 * n]  # type: ignore
        post_gain = arr[4 * n + 1]  # type: ignore
        post_bias = arr[4 * n + 2]  # type: ignore
        return cls(
            window_sizes=window_sizes,
            weights=weights,
            f0s=f0s,
            qs=qs,
            bias=bias,  # type: ignore
            post_gain=post_gain,  # type: ignore
            post_bias=post_bias,  # type: ignore
        )


@jax.tree_util.register_dataclass
@dataclass
class ExperimentHyperparameters:
    device: str = "gpu"  # or 'cpu'
    min_length_s: float = 5.0
    max_length_s: float = 10.0
    overlap_s: float = 1.0
    window_s: float = 0.04  # for label generation
    loss_epsilon: float = 1e-7
    num_channels: int = 3
    window_bounds: tuple = (1e-3, MAX_WINDOW_SIZE)
    weight_bounds: tuple = (-500.0, 500.0)
    f0_bounds: tuple = (20.0, 20000.0)  # Hz, sensible audio range
    q_bounds: tuple = (0.1, 20.0)       # Q, typical bandpass range
    bias_bounds: tuple = (-500.0, 500.0)
    post_gain_bounds: tuple = (-1e4, 1e4)
    post_bias_bounds: tuple = (-1.0, 1.0)
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


def internal_debug(i, data):
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
        try:
            arr = np.asarray(val)
            ax.plot(arr)
        except Exception:
            ax.plot([val])
        ax.set_ylabel(key)
        ax.set_title(key)
    axes[-1].set_xlabel("Index")
    fig.tight_layout()
    out_path = f"data/plots/debug/{i}.png"
    plt.savefig(out_path)
    plt.close(fig)


def compute_channel_output(
    audio: jnp.ndarray,
    window_size: float,
    weight: float,
    f0: float,
    q: float,
    sample_rate: float,
    is_training: bool,
) -> jnp.ndarray:
    """Compute a single channel's weighted moving average of the power envelope, with bandpass filter."""
    b, a = design_biquad_bandpass(f0, q, sample_rate)
    filtered = biquad_apply(audio, b, a)
    power = jnp.abs(filtered)
    window_samples = window_size * sample_rate
    env = moving_average(power, window_samples, is_training=is_training)
    return weight * env


def compute_all_channels(
    audio: jnp.ndarray,
    window_sizes: jnp.ndarray,
    weights: jnp.ndarray,
    f0s: jnp.ndarray,
    qs: jnp.ndarray,
    sample_rate: float,
    is_training: bool,
) -> jnp.ndarray:
    """Compute all channel outputs as a JAX array."""
    n = window_sizes.shape[0]
    assert n == weights.shape[0] == f0s.shape[0] == qs.shape[0], "All per-channel parameter arrays must have same length"
    def channel_fn(i):
        return compute_channel_output(
            audio,
            window_sizes[i], # type: ignore
            weights[i], # type: ignore
            f0s[i], # type: ignore
            qs[i], # type: ignore
            sample_rate,
            is_training,
        ) # type: ignore
    return jnp.stack([channel_fn(i) for i in range(n)], axis=0)


def transient_detector(
    params: TransientDetectorParameters,
    audio: jnp.ndarray,
    sample_rate: float,
    do_debug: bool = False,
    is_training: bool = True,
    hyperparams: Optional[ExperimentHyperparameters] = None,
) -> jnp.ndarray:
    channel_outputs = compute_all_channels(
        audio, params.window_sizes, params.weights, params.f0s, params.qs, sample_rate, is_training
    )
    summed = jnp.sum(channel_outputs, axis=0)
    inner = params.bias + summed
    outer = params.post_gain * inner + params.post_bias
    result = jax.nn.sigmoid(outer)

    if do_debug:
        global _dbg_n
        n = params.window_sizes.shape[0]
        debug_data = {
            "audio": audio,
            "power": jnp.abs(audio),
            **{
                f"env_{i}": channel_outputs[i] / params.weights[i]
                if params.weights[i] != 0
                else channel_outputs[i]
                for i in range(n)
            },
            **{f"weighted_env_{i}": channel_outputs[i] for i in range(n)},
            "summed": summed,
            "inner": inner,
            "outer": outer,
            "result": result,
        }
        internal_debug(_dbg_n, debug_data)
        _dbg_n += 1

    return result


def loss_function(
    target: jnp.ndarray,
    prediction: jnp.ndarray,
    hyperparams: ExperimentHyperparameters,
) -> jnp.ndarray:
    # Binary cross entropy loss
    eps = hyperparams.loss_epsilon
    prediction = jnp.clip(prediction, eps, 1 - eps)
    loss = -(target * jnp.log(prediction) + (1 - target) * jnp.log(1 - prediction))
    return jnp.mean(loss)


def optimize_transient_detector(chunks, hyperparams: ExperimentHyperparameters):
    """
    Optimize the transient detector parameters using a set of labeled audio chunks.
    """
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
        params = TransientDetectorParameters.from_array(params_array, hyperparams)
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
    x0 = np.array(default_params.to_array(hyperparams), dtype=np.float32)

    # Build bounds for all parameters
    n = hyperparams.num_channels
    bounds = (
        [hyperparams.window_bounds] * n
        + [hyperparams.weight_bounds] * n
        + [hyperparams.f0_bounds] * n
        + [hyperparams.q_bounds] * n
        + [hyperparams.bias_bounds]
        + [hyperparams.post_gain_bounds]
        + [hyperparams.post_bias_bounds]
    )

    # Progress display callback
    def progress_callback(xk):
        logger.info(
            f"Current: {TransientDetectorParameters.from_array(xk, hyperparams)}"
        )

    result = scipy.optimize.minimize(
        loss_for_params,
        x0,
        method="L-BFGS-B",
        jac=loss_grad,
        bounds=bounds,
        options={"disp": True},
        callback=progress_callback,
    )

    # Log optimization result details
    logger.info(f"""
Optimization finished.
  Success: {result.success}
  Status: {result.status}
  Message: {result.message}
  Function evaluations: {result.nfev}
  Gradient evaluations: {getattr(result, "njev", "N/A")}
  Final loss: {result.fun}
  Optimized parameters: {result.x}
""")

    return TransientDetectorParameters.from_array(result.x, hyperparams)



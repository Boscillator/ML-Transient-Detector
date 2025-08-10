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

logger = logging.getLogger(__name__)


# Maximum window size for moving average (in seconds)
MAX_WINDOW_SIZE = 0.5  # seconds, must match optimizer bounds
# Assumed sample rate for kernel size (must be global for MAX_KERNEL_SIZE)
ASSUME_SAMPLE_RATE = 48000
# Kernel size for moving average (must be odd)
MAX_KERNEL_SIZE = int(round(MAX_WINDOW_SIZE * ASSUME_SAMPLE_RATE))
MAX_KERNEL_SIZE = MAX_KERNEL_SIZE + (MAX_KERNEL_SIZE + 1) % 2


@dataclass
class TransientDetectorParameters:
    fast_window: float = 0.01  # in seconds (can be fractional for differentiability)
    slow_window: float = 0.1  # in seconds (can be fractional for differentiability)
    w0: float = 0.0  # bias term
    w1: float = 20.0  # fast_env weight
    w2: float = -20.0  # slow_env weight
    post_gain: float = 1.0
    post_bias: float = 0.0


    def to_array(self, hyperparams: 'ExperimentHyperparameters') -> jnp.ndarray:
        """Convert parameters to a JAX array in canonical order. hyperparams is reserved for future use."""
        return jnp.array([
            self.fast_window,
            self.slow_window,
            self.w0,
            self.w1,
            self.w2,
            self.post_gain,
            self.post_bias,
        ], dtype=jnp.float32)

    @classmethod
    def from_array(cls, arr: jnp.ndarray, hyperparams: 'ExperimentHyperparameters') -> Self:
        """Create a TransientDetectorParameters from a JAX or numpy array. hyperparams is reserved for future use."""
        arr = jnp.asarray(arr)

        # We know that this array contains floats, but the type checker doesn't.
        # However, we can't use `float()` here because the array might be a JAX tracer.
        return cls(
            fast_window=arr[0], # type: ignore
            slow_window=arr[1], # type: ignore
            w0=arr[2], #type: ignore
            w1=arr[3], #type: ignore
            w2=arr[4], #type: ignore
            post_gain=arr[5], #type: ignore
            post_bias=arr[6], #type: ignore
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


def transient_detector(
    params: TransientDetectorParameters,
    audio: jnp.ndarray,
    sample_rate: float,
    do_debug: bool = False,
    is_training: bool = True,
    hyperparams: Optional[ExperimentHyperparameters] = None,
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
    hyperparams: ExperimentHyperparameters,
) -> jnp.ndarray:
    # Binary cross entropy loss
    eps = hyperparams.loss_epsilon
    prediction = jnp.clip(prediction, eps, 1 - eps)
    loss = -(target * jnp.log(prediction) + (1 - target) * jnp.log(1 - prediction))
    return jnp.mean(loss)


def optimize_transient_detector(
    chunks: List[TransientExample],
    hyperparams: ExperimentHyperparameters,
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
    bounds = list(hyperparams.bounds)

    # Progress display callback
    def progress_callback(xk):
        logger.info(f"Current: {TransientDetectorParameters.from_array(xk, hyperparams)}")

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
  Gradient evaluations: {getattr(result, 'njev', 'N/A')}
  Final loss: {result.fun}
  Optimized parameters: {result.x}
""")

    return TransientDetectorParameters.from_array(result.x, hyperparams)

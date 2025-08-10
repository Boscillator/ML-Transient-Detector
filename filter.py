import jax.numpy as jnp
import jax.lax as lax


def design_biquad_bandpass(f0_hz, q, fs) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Differentiable RBJ-style bandpass (constant skirt gain, peak gain = Q) biquad.
    Returns normalized coefficients b=[b0,b1,b2], a=[1,a1,a2].
    """
    # Clamp to valid ranges to aid stability
    nyq = 0.5 * fs
    f0 = jnp.clip(f0_hz, 1.0, nyq - 1.0)
    q = jnp.clip(q, 1e-3, 1e3)

    w0 = 2.0 * jnp.pi * (f0 / fs)
    alpha = jnp.sin(w0) / (2.0 * q)
    cosw0 = jnp.cos(w0)

    b0 = q * alpha
    b1 = 0.0
    b2 = -q * alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * cosw0
    a2 = 1.0 - alpha

    # Normalize by a0
    b = jnp.array([b0 / a0, b1 / a0, b2 / a0], dtype=jnp.float32)
    a = jnp.array([1.0, a1 / a0, a2 / a0], dtype=jnp.float32)
    return b, a


def biquad_apply(x: jnp.ndarray, b: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
    """Causal IIR biquad evaluation via lax.scan; differentiable in x,b,a."""
    b0, b1, b2 = b
    _, a1, a2 = a

    def step(carry, xn):
        x1, x2, y1, y2 = carry
        yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        return (xn, x1, yn, y1), yn

    init = (0.0, 0.0, 0.0, 0.0)
    _, y = lax.scan(step, init, x)
    return y

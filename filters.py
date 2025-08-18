import jax.numpy as jnp
import jax.lax as lax

def design_biquad_bandpass(f0_hz, q, fs) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Differentiable RBJ-style bandpass (constant skirt gain, peak gain = Q) biquad.
    Returns normalized coefficients b=[b0,b1,b2], a=[1,a1,a2].
    """
    # # Clamp to valid ranges to aid stability
    # nyq = 0.5 * fs
    # f0 = jnp.clip(f0_hz, 1.0, nyq - 1.0)
    # q = jnp.clip(q, 1e-3, 1e3)

    # w0 = 2.0 * jnp.pi * (f0 / fs)
    # alpha = jnp.sin(w0) / (2.0 * q)
    # cosw0 = jnp.cos(w0)

    # b0 = q * alpha
    # b1 = 0.0
    # b2 = -q * alpha
    # a0 = 1.0 + alpha
    # a1 = -2.0 * cosw0
    # a2 = 1.0 - alpha

    # # Normalize by a0
    # b = jnp.array([b0 / a0, b1 / a0, b2 / a0], dtype=jnp.float32)
    # a = jnp.array([1.0, a1 / a0, a2 / a0], dtype=jnp.float32)
    # return b, a

    w0 = 2.0 * jnp.pi * f0_hz / fs
    alpha = jnp.sin(w0) / (2.0 * q)
    b0 = alpha
    b1 = 0
    b2 = -alpha
    a0 = 1 + alpha
    a1 = -2  * jnp.cos(w0)
    a2 = 1 - alpha
    b = jnp.array([b0 / a0, b1 / a0, b2 / a0], dtype=jnp.float32)
    a = jnp.array([1.0, a1 / a0, a2 / a0], dtype=jnp.float32)
    return b, a


def convert_to_fir_filter(b: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
    """
    Convert biquad (IIR) coefficients to a causal FIR filter by matching the frequency response.
    Uses frequency sampling: computes the frequency response of the biquad, then uses IFFT to get FIR taps.
    The result is a real, causal, differentiable FIR filter.
    """
    # Number of FIR taps (should be odd for symmetry, and long enough for accuracy)
    num_taps = 129
    # Frequency grid (linear, up to Nyquist)
    w = jnp.linspace(0, jnp.pi, num_taps)
    # Evaluate biquad frequency response H(e^{jw})
    ejw = jnp.exp(1j * w)
    ejw2 = jnp.exp(2j * w)
    num = b[0] + b[1] * ejw**-1 + b[2] * ejw**-2
    den = a[0] + a[1] * ejw**-1 + a[2] * ejw**-2
    H = num / den
    # IFFT to get impulse response (FIR taps)
    # Mirror to get full spectrum for real IFFT
    H_full = jnp.concatenate([H, jnp.conj(H[-2:0:-1])])
    h = jnp.fft.ifft(H_full).real
    # Center the impulse response (causal: shift right)
    h = jnp.roll(h, num_taps // 2)
    # Truncate to num_taps (causal, all zeros before index 0)
    fir = h[:num_taps]
    return fir.astype(jnp.float32)


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


def apply_fir_filter(x: jnp.ndarray, b: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
    """
    Apply a causal FIR filter (converted from biquad) to x using convolution.
    This is differentiable and causal.
    """
    fir = convert_to_fir_filter(b, a)
    # Reshape for JAX conv: [batch, length, channels]
    x_ = x[None, :, None]  # [N, L, C] = [1, length, 1]
    fir_ = fir[:, None, None]  # [W, I, O] = [kernel, 1, 1]
    y = lax.conv_general_dilated(
        x_,
        fir_,
        window_strides=(1,),
        padding=[(fir.shape[0] - 1, 0)],  # causal: pad left only
        dimension_numbers=("NWC", "WIO", "NWC"),
    )[0, :, 0]
    return y

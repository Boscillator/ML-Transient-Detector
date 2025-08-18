import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../.."))

from filters import design_biquad_bandpass, biquad_apply, apply_fir_filter

# Settings for grid
f0s = jnp.array([500.0, 1000.0, 5000.0, 10_000.0])  # Hz
qs = jnp.array([0.1, 0.5, 1.0, 1.5, 2.0])
sample_rate = 48000
N = 4096  # FFT size
impulse = jnp.zeros(N)
impulse = impulse.at[0].set(1.0)

# Frequency axis
freqs = jnp.fft.rfftfreq(N, 1.0 / sample_rate)

# Prepare plot grid
grid_rows = len(f0s)
grid_cols = len(qs)
fig, axes = plt.subplots(
    grid_rows,
    grid_cols,
    figsize=(4 * grid_cols, 3 * grid_rows),
    sharex=True,
    sharey=True,
)


for i, f0 in enumerate(f0s):
    for j, q in enumerate(qs):
        print(f"Processing filter: f0={f0:.1f} Hz, Q={q:.2f}")
        b, a = design_biquad_bandpass(f0, q, sample_rate)
        # IIR response
        iir_out = biquad_apply(impulse, b, a)
        iir_fft = jnp.fft.rfft(iir_out)
        # FIR response
        fir_out = apply_fir_filter(impulse, b, a)
        fir_fft = jnp.fft.rfft(fir_out)
        # Plot
        ax = axes[i, j] if grid_rows > 1 else axes[j]
        ax.plot(freqs, 20 * jnp.log10(jnp.abs(iir_fft) + 1e-10), label="IIR (biquad)")
        ax.plot(
            freqs,
            20 * jnp.log10(jnp.abs(fir_fft) + 1e-10),
            label="FIR (converted)",
            linestyle="--",
        )
        ax.set_title(f"f0={f0:.0f} Hz, Q={q:.1f}")
        ax.set_xlim(0, sample_rate / 2)
        ax.set_ylim(-40, 10)
        if i == grid_rows - 1:
            ax.set_xlabel("Frequency (Hz)")
        if j == 0:
            ax.set_ylabel("Magnitude (dB)")
        ax.legend()

plt.tight_layout()
print("Saving plot to data/plots/compare_frequency_responses.png ...")
os.makedirs("data/plots", exist_ok=True)
plt.savefig("data/plots/compare_frequency_responses.png")
plt.close(fig)
print("Saved plot to data/plots/compare_frequency_responses.png")
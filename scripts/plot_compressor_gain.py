
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Parameters
threshold = 0.2
makeup_gain = 1.0

envelope = jnp.linspace(0.001, 1.5, 500)  # avoid log(0)
gain = jnp.where(envelope > threshold, threshold / envelope, 1.0)
gain_db = 20 * jnp.log10(gain)

plt.figure(figsize=(8, 5))
plt.plot(envelope, gain_db, label="Gain Reduction (dB)")
plt.axvline(threshold, color="r", linestyle="--", label="Threshold")
plt.xlabel("Envelope")
plt.ylabel("Gain (dB)")
plt.title("Hard-Knee Compressor Gain (dB) vs Envelope")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/plots/compressor.png")
plt.close()
print("Saved plot to data/plots/compressor.png")
